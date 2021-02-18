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
                   fix_length=64,
                   pad_token=PAD_INDEX,
                   unk_token=UNK_INDEX)
# id,title,author,text,label
fields = [('id', int_field), ('title', text_field), ('author', None), ('label', int_field)]

train, test = TabularDataset(path=Parameters.SOURCE_2_FOLDER + "/" + Parameters.SOURCE_2_FILE,
                       format=Parameters.DATASET_FORMAT,
                       fields=fields,
                       skip_header=True).split()

test_iter = Iterator(test,
                     batch_size=Parameters.BATCH_SIZE,
                     device=Parameters.DEVICE,
                     train=False,
                     shuffle=False,
                     sort=False)

print("Finish dataset prepare")
