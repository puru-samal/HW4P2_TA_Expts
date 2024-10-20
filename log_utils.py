import logging
import sys


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
    logger.info(f"Batch Size           : {config["batch_size"]}")
    logger.info(f"Train Batches        : {train_loader.__len__()}")
    logger.info(f"Val Batches          : {val_loader.__len__()}")
    logger.info(f"Test Batches         : {test_loader.__len__()}")
    
    logger.info("Checking the Shapes of the Data --\n")
    for batch in train_loader:
        x_pad, y_shifted_pad, y_golden_pad, x_len, y_len, = batch

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