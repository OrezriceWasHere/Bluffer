from torchtext.data import Field, TabularDataset, BucketIterator, Iterator
from transformers import BertTokenizer
import torch
import Parameters

# Use BERT tokenizer (the method to use words the same way they are used in BERT model)
tokenizer = BertTokenizer.from_pretrained(Parameters.BERT_TOKENIZER_NAME)

# Define padding token and unknown word token
PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

int_field = Field(sequential=False,
                  use_vocab=False,
                  batch_first=True,
                  dtype=torch.int)

text_field = Field(use_vocab=False,
                   tokenize=tokenizer.encode,
                   lower=False,
                   include_lengths=False,
                   batch_first=True,
                   fix_length=Parameters.MAX_SEQ_LEN,
                   pad_token=PAD_INDEX,
                   unk_token=UNK_INDEX)
# id,title,author,text,label
fields = [('id', int_field), ('title', text_field), ('author', text_field), ('text', text_field), ('label', int_field)]

train = TabularDataset(path=Parameters.SOURCE_FOLDER + "/" + Parameters.TRAIN_FILE_NAME,
                       format=Parameters.DATASET_FORMAT,
                       fields=fields,
                       skip_header=True)

test = TabularDataset.splits(path=Parameters.SOURCE_FOLDER + "/" + Parameters.TEST_FILE_NAME,
                             format=Parameters.DATASET_FORMAT,
                             fields=fields,
                             skip_header=True)

print(f'running on {Parameters.DEVICE}')

train_iter = BucketIterator(train,
                            batch_size=Parameters.BATCH_SIZE,
                            sort_key=lambda x: len(x.text),
                            device=Parameters.DEVICE,
                            train=True,
                            sort=True,
                            sort_within_batch=True)
test_iter = Iterator(test,
                     batch_size=Parameters.BATCH_SIZE,
                     device=Parameters.DEVICE,
                     train=False,
                     shuffle=False,
                     sort=False)

print("Finish dataset prepare")
