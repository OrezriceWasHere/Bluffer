# Data parameters:
import os
SOURCE_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")
SOURCE_FOLDER = os.path.join(SOURCE_FOLDER, "Dataset2")
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
SUBMIT_FILE_NAME = "submit.csv"
DATASET_FORMAT = "CSV"

# Model Parameters:
BERT_TOKENIZER_NAME = "bert-base-uncased"
MAX_SEQ_LEN = 128
BATCH_SIZE = 16


