import logging
import sys
import torch
import Levenshtein
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def setup_logger(fname):
    logger = logging.getLogger(__name__)  # Custom logger for this module
    handler = logging.FileHandler(fname, mode='w')  # Log to file
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

def log_data_stats(logger, config, train_dataset, train_loader, val_loader, test_loader):
    logger.info("Data Stats: ")
    logger.info(f"No. of Train MFCCs   : {train_dataset.__len__()}")
    logger.info(f"Batch Size           : {config['batch_size']}")
    logger.info(f"Train Batches        : {train_loader.__len__()}")
    logger.info(f"Val Batches          : {val_loader.__len__()}")
    logger.info(f"Test Batches         : {test_loader.__len__()}")
    
    padded_input, padded_target, input_lengths, target_lengths = None, None, None, None
    logger.info("Checking the Shapes of the Data --\n")
    for batch in train_loader:
        x_pad, y_shifted_pad, y_golden_pad, x_len, y_len, = batch
        padded_input, padded_target, input_lengths, target_lengths = x_pad, y_shifted_pad, x_len, y_len
        logger.info(f"x_pad shape:\t\t{x_pad.shape}")
        logger.info(f"x_len shape:\t\t{x_len.shape}\n")

        logger.info(f"y_shifted_pad shape:\t{y_shifted_pad.shape}")
        logger.info(f"y_golden_pad shape:\t{y_golden_pad.shape}")
        logger.info(f"y_len shape:\t\t{y_len.shape}\n")

        # convert one transcript to text
        transcript = train_dataset.tokenizer.decode(y_shifted_pad[0].tolist())
        logger.info(f"Transcript Shifted: {transcript}")
        transcript = train_dataset.tokenizer.decode(y_golden_pad[0].tolist())
        logger.info(f"Transcript Golden: {transcript}")
        break
    return padded_input, padded_target, input_lengths, target_lengths


def verify_dataset(dataloader, partition):
    print("Loaded Path: ", partition)

    max_len_mfcc = 0
    max_len_t    = 0  # To track the maximum length of transcripts

    # Iterate through the dataloader
    for batch in tqdm(dataloader, desc=f"Verifying {partition} Dataset"):
        x_pad, y_shifted_pad, y_golden_pad, x_len, y_len = batch

        len_x = x_pad.shape[1]
        if len_x > max_len_mfcc:
            max_len_mfcc = len_x

        # Update the maximum transcript length
        # transcript length is dim 1 of y_shifted_pad
        len_y = y_shifted_pad.shape[1]
        if len_y > max_len_t:
            max_len_t = len_y

    print(f"Maximum MFCC Length in Dataset: {max_len_mfcc}\n")
    print(f"Maximum Transcript Length in Dataset: {max_len_t}\n")
    return max_len_mfcc, max_len_t

def save_model(model, optimizer, scheduler, metric, epoch, path):
    torch.save(
        {"model_state_dict"         : model.state_dict(),
         "optimizer_state_dict"     : optimizer.state_dict(),
         "scheduler_state_dict"     : scheduler.state_dict() if scheduler is not None else {},
         metric[0]                  : metric[1],
         "epoch"                    : epoch},
         path
    )

def load_model(path, model, metric= "valid_acc", optimizer= None, scheduler= None):

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer != None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler != None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch   = checkpoint["epoch"]
    metric  = checkpoint[metric]

    return [model, optimizer, scheduler, epoch, metric]

def num_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params / 1E6

def indices_to_chars(indices, tokenizer):
    tokens = tokenizer.decode(indices)
    return tokens


''' utility function for Levenshtein Distantce quantification '''
def calc_edit_distance(predictions, y, y_len, tokenizer, calc_lev=False, print_example=False):

    dist = 0.0
    batch_size, seq_len = predictions.shape

    for batch_idx in range(batch_size):
        
        # Trim upto the EOS_TOKEN
        pad_indices = torch.where(predictions[batch_idx] == tokenizer.EOS_TOKEN)[0]
        if pad_indices.numel() > 0:
            lowest_pad_idx = pad_indices.min().item()
        else:
            lowest_pad_idx = 0
        if lowest_pad_idx == len(predictions[batch_idx]):
            pred_trimmed = predictions[batch_idx]
        else:
            pred_trimmed = predictions[batch_idx, :lowest_pad_idx+1]
        
        y_string = indices_to_chars(y[batch_idx, 0 : y_len[batch_idx]], tokenizer)
        pred_string = indices_to_chars(pred_trimmed, tokenizer)

        if calc_lev:
            curr_dist   = Levenshtein.distance(pred_string, y_string)
            dist += curr_dist
    
    if print_example:
        print("\nGround Truth : ", y_string)
        print("Prediction   : ", pred_string)

    dist /= batch_size
    return dist, y_string, pred_string


def train_model(model, train_loader, loss_func, optimizer, scaler, pad_token, device):

    model.train()
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc="Train")

    total_loss          = 0
    running_loss        = 0.0
    running_perplexity  = 0.0

    for i, (inputs, targets_shifted, targets_golden, inputs_lengths, targets_lengths) in enumerate(train_loader):

        optimizer.zero_grad()

        inputs          = inputs.to(device)
        targets_shifted = targets_shifted.to(device)
        targets_golden  = targets_golden.to(device)


        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # passing the minibatch through the model
            raw_predictions, attention_weights = model(inputs, inputs_lengths, targets_shifted, targets_lengths)

            padding_mask = torch.logical_not(torch.eq(targets_shifted, pad_token))

            # cast the mask
            loss = loss_func(raw_predictions.transpose(1,2), targets_golden)*padding_mask
            loss = loss.sum() / padding_mask.sum()

        scaler.scale(loss).backward()   # This is a replacement for loss.backward()
        scaler.step(optimizer)          # This is a replacement for optimizer.step()
        scaler.update()                 # This is something added just for FP16

        running_loss        += float(loss.item())
        perplexity          = torch.exp(loss)
        running_perplexity  += perplexity.item()

        # online training monitoring
        batch_bar.set_postfix(
            loss = "{:.04f}".format(float(running_loss / (i + 1))),
            perplexity = "{:.04f}".format(float(running_perplexity / (i + 1)))
        )

        batch_bar.update()

        del inputs, targets_shifted, targets_golden, inputs_lengths, targets_lengths
        torch.cuda.empty_cache()

    running_loss        = float(running_loss / len(train_loader))
    running_perplexity  = float(running_perplexity / len(train_loader))

    batch_bar.close()

    return running_loss, running_perplexity, attention_weights


def validate_fast(model, dataloader, tokenizer, device, calc_lev=False):
    model.eval()

    # progress bar
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc="Val", ncols=5)

    # Remove the padding tokens from the predictions and targets
    running_distance = 0.0
    json_output = {}
    for i, (inputs, targets_shifted, targets_golden, inputs_lengths, targets_lengths) in enumerate(dataloader):

        inputs  = inputs.to(device)
        targets_golden = targets_golden.to(device)

        with torch.inference_mode():
            greedy_predictions = model.recognize(inputs, inputs_lengths)
        
        # calculating Levenshtein Distance
        # @NOTE: modify the print_example to print more or less validation examples
        dist, y_string, pred_string = calc_edit_distance(greedy_predictions, targets_golden, targets_lengths, tokenizer, print_example=False, calc_lev=calc_lev)
        running_distance += dist
        json_output[i] = {
            "Input": y_string,
            "Output": pred_string
        }

        # online validation distance monitoring
        batch_bar.set_postfix(
            running_distance = "{:.04f}".format(float(running_distance / (i + 1)))
        )

        batch_bar.update()

        del inputs, targets_shifted, targets_golden, inputs_lengths, targets_lengths
        torch.cuda.empty_cache()

        if i==4: break      # validating only upon first five batches

    batch_bar.close()
    running_distance /= 5

    return running_distance, json_output

def validate_full(model, dataloader, tokenizer, device):
    model.eval()

    # progress bar
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc="Val", ncols=5)

    running_distance = 0.0

    for i, (inputs, targets_shifted, targets_golden, inputs_lengths, targets_lengths) in enumerate(dataloader):

        inputs  = inputs.to(device)
        targets_golden = targets_golden.to(device)

        with torch.inference_mode():
            greedy_predictions = model.recognize(inputs, inputs_lengths)

        # calculating Levenshtein Distance
        # @NOTE: modify the print_example to print more or less validation examples
        running_distance += calc_edit_distance(greedy_predictions, targets_golden, targets_lengths, tokenizer, print_example=True)

        # online validation distance monitoring
        batch_bar.set_postfix(
            running_distance = "{:.04f}".format(float(running_distance / (i + 1)))
        )

        batch_bar.update()

        del inputs, targets_shifted, targets_golden, inputs_lengths, targets_lengths
        torch.cuda.empty_cache()


    batch_bar.close()
    running_distance /= len(dataloader)

    return running_distance

def save_attention_plot(attention_weights, epoch=0):
    ''' function for saving attention weights plot to a file

        @NOTE: default starter code set to save cross attention
    '''

    plt.clf()  # Clear the current figure
    sns.heatmap(attention_weights, cmap="GnBu")  # Create heatmap

    # Save the plot to a file. Specify the directory if needed.
    plt.savefig(f"cross_attention-epoch{epoch}.png")