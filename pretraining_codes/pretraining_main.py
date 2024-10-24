from utils2 import *
from data_processing2 import *
from hw_tokenizers2 import *
import yaml
import os
from transformer2 import *
import json
import argparse
import gc

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: ", device)




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="HW4P2_Basic_Tests")
    parser.add_argument('config_path', type=str, help='path to experiment config file')
    args = parser.parse_args()
    
    #### SETUP ----------------------------------------------------------------------------------------------------------------
    EXPT_CONFIG_PATH = args.config_path
    #### ----------------------------------------------------------------------------------------------------------------------


    #### Expt Config ----------------------------------------------------------------------------------------------------------
    with open(EXPT_CONFIG_PATH) as file:
        config = yaml.safe_load(file)

    EXPT_NAME = "{}_Transformer_TOK-{}_Embed-{}_ENC-{}_{}_DEC-{}_{}_{}_{}_{}_{}".format(
                            config["Name"],
                            config['token_type'],
                            config['embed_type'],
                            config["enc_num_layers"],       
                            config["enc_num_heads"],        
                            config["dec_num_layers"],
                            config["dec_num_heads"],
                            config["d_model"],
                            config["d_ff"],
                            config["optimizer"],
                            config["scheduler"])
    #### ----------------------------------------------------------------------------------------------------------------------


    #### Init Tokenizer -------------------------------------------------------------------------------------------------------
    tokenizer = GTokenizer(config['token_type'])
    print('')
    #### ----------------------------------------------------------------------------------------------------------------------


    #### Data -----------------------------------------------------------------------------------------------------------------
    train_dataset   = SpeechDataset(root='../hw4p2', partition=config['train_partition'], config=config, tokenizer=tokenizer, isTrainPartition=True,  subset=config['subset'])
    val_dataset     = SpeechDataset(root='../hw4p2', partition=config['val_partition'],   config=config, tokenizer=tokenizer, isTrainPartition=False, subset=config['subset'])
    test_dataset    = SpeechDataset(root='../hw4p2', partition=config['test_partition'],   config=config, tokenizer=tokenizer, isTrainPartition=False, subset=config['subset'])

    train_loader    = torch.utils.data.DataLoader(
        dataset     = train_dataset,
        batch_size  = config["batch_size"],
        shuffle     = True,
        num_workers = config['NUM_WORKERS'],
        pin_memory  = True,
        collate_fn  = train_dataset.collate_fn
    )

    val_loader      = torch.utils.data.DataLoader(
        dataset     = val_dataset,
        batch_size  = config["batch_size"],
        shuffle     = False,
        num_workers = config['NUM_WORKERS'],
        pin_memory  = True,
        collate_fn  = val_dataset.collate_fn
    )

    test_loader     = torch.utils.data.DataLoader(
        dataset     = test_dataset,
        batch_size  = config["batch_size"],
        shuffle     = False,
        num_workers = config['NUM_WORKERS'],
        pin_memory  = True,
        collate_fn  = test_dataset.collate_fn
    )

    print('')
    print("Data Stats: ")
    print(f"No. of Train MFCCs   : {train_dataset.__len__()}")
    print(f"Batch Size           : {config['batch_size']}")
    print(f"Train Batches        : {train_loader.__len__()}")
    print(f"Val Batches          : {val_loader.__len__()}")
    print(f"Test Batches         : {test_loader.__len__()}")
    print('')
    print("Checking the Shapes of the Data --\n")
    for batch in train_loader:
        x_pad, y_shifted_pad, y_golden_pad, x_len, y_len, = batch
        print(f"x_pad shape:\t\t{x_pad.shape}")
        print(f"x_len shape:\t\t{x_len.shape}")
        print(f"y_shifted_pad shape:\t{y_shifted_pad.shape}")
        print(f"y_golden_pad shape:\t{y_golden_pad.shape}")
        print(f"y_len shape:\t\t{y_len.shape}\n")

        # convert one transcript to text
        transcript = train_dataset.tokenizer.decode(y_shifted_pad[0].tolist())
        print(f"Transcript Shifted: {transcript}\n")
        transcript = train_dataset.tokenizer.decode(y_golden_pad[0].tolist())
        print(f"Transcript Golden: {transcript}\n")
        break
    print('')

    print("\n\nVerifying Train Dataset")
    max_train_mfcc, max_train_transcript = verify_dataset(train_loader, config['train_partition'])
    max_val_mfcc, max_val_transcript     =  verify_dataset(val_loader, config['val_partition'])
    max_test_mfcc, max_test_transcript   = verify_dataset(test_loader, config['test_partition'])

    MAX_MFCC_LEN  = max(max_train_mfcc, max_val_mfcc, max_test_mfcc)
    MAX_TRANS_LEN = max(max_train_transcript, max_val_transcript, max_test_transcript)
    print(f"Maximum MFCC Length in Entire Dataset: {MAX_MFCC_LEN}")
    print(f"Maximum Transcript Length in Entire Dataset: {MAX_TRANS_LEN}")
    print('')
    gc.collect()
    #### ----------------------------------------------------------------------------------------------------------------------

    #### Model ----------------------------------------------------------------------------------------------------------------
    model = Transformer(
        input_dim                   = config['input_dim'],
        enc_num_layers              = config['enc_num_layers'],
        dec_num_layers              = config['dec_num_layers'],
        enc_num_heads               = config['enc_num_heads'],
        dec_num_heads               = config['dec_num_heads'],
        d_model                     = config['d_model'],
        d_ff                        = config['d_ff'],
        target_vocab_size           = tokenizer.VOCAB_SIZE,
        eos_token                   = tokenizer.EOS_TOKEN,
        sos_token                   = tokenizer.SOS_TOKEN,
        pad_token                   = tokenizer.PAD_TOKEN,
        enc_dropout                 = config['enc_dropout'],
        dec_dropout                 = config['dec_dropout'],
        trans_max_seq_length        = MAX_TRANS_LEN,
        mfcc_max_seq_length         = MAX_MFCC_LEN,
        embed_type                  = config['embed_type']
    ).to(device)

    para = num_parameters(model)
    print("#"*10)
    print(f"Model Parameters:\n {para}M")
    print("#"*10)
    print('')
    #### ----------------------------------------------------------------------------------------------------------------------


    #### Loss | Optim | Sched -------------------------------------------------------------------------------------------------
    loss_func   = nn.CrossEntropyLoss(ignore_index = tokenizer.PAD_TOKEN)
    scaler      = torch.amp.GradScaler()

    if config["optimizer"] == "SGD":
        # feel free to change any of the initializations you like to fit your needs
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=config["learning_rate"],
                                    momentum=config["momentum"],
                                    weight_decay=1E-4,
                                    nesterov=config["nesterov"])

    elif config["optimizer"] == "Adam":
        # feel free to change any of the initializations you like to fit your needs
        optimizer = torch.optim.Adam(model.parameters(),
                                    lr=float(config["learning_rate"]),
                                    weight_decay=1e-6,betas= (0.9, 0.999),maximize= False,amsgrad= False,capturable= False ,fused= None )

    elif config["optimizer"] == "AdamW":
        # feel free to change any of the initializations you like to fit your needs
        optimizer = torch.optim.AdamW(model.parameters(),
                                        lr=float(config["learning_rate"]),
                                        weight_decay=0.01)


    if config["scheduler"] == "ReduceLR":
        #Feel Free to change any of the initializations you like to fit your needs
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                        factor=config["factor"], patience=config["patience"], min_lr=1E-8, threshold=1E-1)

    elif config["scheduler"] == "CosineAnnealing":
        #Feel Free to change any of the initializations you like to fit your needs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                        T_max = config["epochs"], eta_min=1E-8)
    

    ###  Now let's train the encoder to master the encoder input ranges
    e                   = 0
    best_loss           = 10.0
    best_dist = 60
    RESUME_LOGGING = False
    checkpoint_root = os.path.join(os.getcwd(), 'checkpoints')
    os.makedirs(checkpoint_root, exist_ok=True)
    os.makedirs('attention_imgs', exist_ok=True)
    os.makedirs('out_text', exist_ok=True)
    checkpoint_root_lm = os.path.join(os.getcwd(), 'checkpoints_lm')
    os.makedirs(checkpoint_root_lm, exist_ok=True)
    os.makedirs('attention_imgs_lm', exist_ok=True)
    os.makedirs('out_text_lm', exist_ok=True)
    checkpoint_best_loss_model_filename     = 'checkpoint-best-loss-modelfull.pth'
    checkpoint_last_epoch_filename          = 'checkpoint-epochfull-'
    best_loss_model_path                    = os.path.join(checkpoint_root, checkpoint_best_loss_model_filename)

    lm_epochs = config['lm_epochs']

    for epoch in range(e, lm_epochs):

        print("\nEpoch LM {}/{}".format(epoch+1, config["epochs"]))

        curr_lr = float(optimizer.param_groups[0]["lr"])

        train_loss, train_perplexity, attention_weights = train_model(model, train_loader, loss_func, optimizer, scaler, tokenizer.PAD_TOKEN, device,True)
    
        print("\nEpoch {}/{}: \nTrain Loss {:.04f}\t Train Perplexity {:.04f}\t Learning Rate {:.04f}".format(
            epoch + 1, config["epochs"], train_loss, train_perplexity, curr_lr))


        # res = validate_full_lm(model, val_loader, epoch,tokenizer,  tokenizer.PAD_TOKEN)

        # with open(f"out_text/{epoch+1}_out.json", "w") as f:
        #     json.dump(res, f, indent=4)
        
        # print("Average NLL: {:.04f}".format(res))
        attention_keys = list(attention_weights.keys())

        attention_weights_decoder_self       = attention_weights[attention_keys[0]][0].cpu().detach().numpy()
        attention_weights_decoder_cross      = attention_weights[attention_keys[-1]][0].cpu().detach().numpy()
       
        save_attention_plot_lm(attention_weights_decoder_cross, epoch)
        
        if config["scheduler"] == "ReduceLR":
            scheduler.step(train_loss)
        else:
            scheduler.step()

        ### Highly Recommended: Save checkpoint in drive and/or wandb if accuracy is better than your current best
        epoch_model_path = os.path.join(checkpoint_root, (checkpoint_last_epoch_filename + str(epoch) + '.pth'))
        save_model(model, optimizer, scheduler, ['train_loss', train_loss], epoch, epoch_model_path)

        # if best_dist >= res:
        #     best_loss = train_loss
        #     best_dist = res
        #     save_model(model, optimizer, scheduler, ['train_loss', train_loss], epoch, best_loss_model_path)
        #     print("Saved best distance model")


    epochs = config["epochs"]

    for epoch in range(e, epochs):

        print("\nEpoch {}/{}".format(epoch+1, config["epochs"]))

        curr_lr = float(optimizer.param_groups[0]["lr"])

        train_loss, train_perplexity, attention_weights = train_model(model, train_loader, loss_func, optimizer, scaler, tokenizer.PAD_TOKEN, device,False)
    
        print("\nEpoch {}/{}: \nTrain Loss {:.04f}\t Train Perplexity {:.04f}\t Learning Rate {:.04f}".format(
            epoch + 1, config["epochs"], train_loss, train_perplexity, curr_lr))


        levenshtein_distance, json_out = validate_fast(model, val_loader, tokenizer, device, config['calc_lev'])

        with open(f"out_text/{epoch+1}_out.json", "w") as f:
            json.dump(json_out, f, indent=4)
        
        print("Levenshtein Distance : {:.04f}".format(levenshtein_distance))
        attention_keys = list(attention_weights.keys())

        attention_weights_decoder_self       = attention_weights[attention_keys[0]][0].cpu().detach().numpy()
        attention_weights_decoder_cross      = attention_weights[attention_keys[-1]][0].cpu().detach().numpy()
       
        save_attention_plot(attention_weights_decoder_cross, epoch)
        save_attention_plot(attention_weights_decoder_self, epoch+100)
        if config["scheduler"] == "ReduceLR":
            scheduler.step(levenshtein_distance)
        else:
            scheduler.step()

        ### Highly Recommended: Save checkpoint in drive and/or wandb if accuracy is better than your current best
        epoch_model_path = os.path.join(checkpoint_root, (checkpoint_last_epoch_filename + str(epoch) + '.pth'))
        save_model(model, optimizer, scheduler, ['train_loss', train_loss], epoch, epoch_model_path)

        if best_dist >= levenshtein_distance:
            best_loss = train_loss
            best_dist = levenshtein_distance
            save_model(model, optimizer, scheduler, ['train_loss', train_loss], epoch, best_loss_model_path)
            print("Saved best distance model")
