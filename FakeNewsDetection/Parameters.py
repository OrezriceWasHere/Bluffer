import os
import torch
from pathlib import Path

# Data parameters:
SOURCE_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")
SOURCE_1_FOLDER = os.path.join(SOURCE_FOLDER, "Dataset1")
SOURCE_2_FOLDER = os.path.join(SOURCE_FOLDER, "Dataset2")
SOURCE_3_FOLDER = os.path.join(SOURCE_FOLDER, "Dataset3")
SOURCE_4_FOLDER = os.path.join(SOURCE_FOLDER, "Dataset4")
SOURCE_5_FOLDER = os.path.join(SOURCE_FOLDER, "Dataset5")
OUTPUT_1_FOLDER = os.path.join(SOURCE_1_FOLDER, "output")
OUTPUT_2_FOLDER = os.path.join(SOURCE_2_FOLDER, "output")
OUTPUT_3_FOLDER = os.path.join(SOURCE_3_FOLDER, "output")
OUTPUT_4_FOLDER = os.path.join(SOURCE_4_FOLDER, "output")
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
SUBMIT_FILE_NAME = "submit.csv"
TWEETS_FILE_NAME = "tweets.tsv"
DATASET_FORMAT = "TSV"
SOURCE_2_FILE = "news2.csv"
MODEL_FILE_NAME = "model.pt"
METRICS_FILE_NAME = "metrics.pt"
LOSS_GRAPH_EXTENSION = ".lossgraph.png"
MODEL_FILE_EXTENSION = MODEL_FILE_NAME.split(".")[1]

OUTPUT_FOLDER = OUTPUT_3_FOLDER
MODEL_OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, MODEL_FILE_NAME)
METRICS_OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, METRICS_FILE_NAME)
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

# Model Parameters:
BERT_TOKENIZER_NAME = "bert-base-uncased"
BATCH_SIZE = 64 if torch.cuda.is_available() else 4
THRESHOLD = 0
LR = 1e-3
TRUE_LABEL_INDEX = 1

# Read only first 128 tokens
MAX_SEQ_LEN = 128

# Running environment Parameters
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Running over {DEVICE}')
