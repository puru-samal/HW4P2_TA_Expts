from log_utils import *
from data_processing import *
from hw_tokenizers import *
import yaml

#### SETUP ----------------------------------------------------------------------------------------------------------------
# Step 1: Setup the path to your expt config
EXPT_CONFIG_PATH = "config_base.yaml"
# Step 2: Set Expt Name
EXPT_NAME       = "TEST"
# Step 3: Set Dataset Paths, Num_workers
DATASET_ROOT    = "hw4p2"
TRAIN_PARTITION = "train-clean-100"
VAL_PARTITION   = "dev-clean"
TEST_PARTITION  = "test-clean"
NUM_WORKERS     = 0 
#### ----------------------------------------------------------------------------------------------------------------------


#### Expt Config ----------------------------------------------------------------------------------------------------------
with open(EXPT_CONFIG_PATH) as file:
    config = yaml.safe_load(file)
logger = setup_logger(EXPT_NAME + ".log")
#### ----------------------------------------------------------------------------------------------------------------------


#### Init Tokenizer -------------------------------------------------------------------------------------------------------
tokenizer = GTokenizer(config['token_type'], logger=logger)
#### ----------------------------------------------------------------------------------------------------------------------


#### Data -----------------------------------------------------------------------------------------------------------------
train_dataset   = SpeechDataset(root=DATASET_ROOT, partition=TRAIN_PARTITION, config=config, tokenizer=tokenizer, isTrainPartition=True,  subset=0.1)
val_dataset     = SpeechDataset(root=DATASET_ROOT, partition=VAL_PARTITION,   config=config, tokenizer=tokenizer, isTrainPartition=False, subset=0.1)
test_dataset    = SpeechDataset(root=DATASET_ROOT, partition=TEST_PARTITION,  config=config, tokenizer=tokenizer, isTrainPartition=False, subset=0.1)

train_loader    = torch.utils.data.DataLoader(
    dataset     = train_dataset,
    batch_size  = config["batch_size"],
    shuffle     = True,
    num_workers = NUM_WORKERS,
    pin_memory  = True,
    collate_fn  = train_dataset.collate_fn
)

val_loader      = torch.utils.data.DataLoader(
    dataset     = val_dataset,
    batch_size  = config["batch_size"],
    shuffle     = False,
    num_workers = NUM_WORKERS,
    pin_memory  = True,
    collate_fn  = val_dataset.collate_fn
)

test_loader     = torch.utils.data.DataLoader(
    dataset     = test_dataset,
    batch_size  = config["batch_size"],
    shuffle     = False,
    num_workers = NUM_WORKERS,
    pin_memory  = True,
    collate_fn  = test_dataset.collate_fn
)

log_data_stats(logger=logger, config=config, train_dataset=train_dataset, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)
#### ----------------------------------------------------------------------------------------------------------------------

