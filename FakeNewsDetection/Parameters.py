import os
import torch
from pathlib import Path

# Data parameters:
SOURCE_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")
OUTPUT_FOLDER = os.path.join(SOURCE_FOLDER, "Data/Output")
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
SOURCE_FOLDER = os.path.join(SOURCE_FOLDER, "Dataset2")
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
SUBMIT_FILE_NAME = "submit.csv"
DATASET_FORMAT = "CSV"
MODEL_OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, "model.pt")
METRICS_OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, "metrics.pt")

# Model Parameters:
BERT_TOKENIZER_NAME = "bert-base-uncased"
BATCH_SIZE = 16

# Read only first 128 tokens
MAX_SEQ_LEN = 128

# Running environment Parameters
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Running over {DEVICE}')




