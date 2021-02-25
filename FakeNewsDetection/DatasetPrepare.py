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

long_field = Field(sequential=False,
                   use_vocab=False,
                   batch_first=True,
                   dtype=torch.long)

text_field = Field(use_vocab=False,
                   tokenize=tokenizer.encode,
                   lower=False,
                   include_lengths=False,
                   batch_first=True,
                   fix_length=Parameters.MAX_SEQ_LEN,
                   pad_token=PAD_INDEX,
                   unk_token=UNK_INDEX)

# label is required to be long field by Net
fields = [('title', text_field), ('label', long_field)]
#
# test = TabularDataset(path=Parameters.SOURCE_4_FOLDER + "/" + "output.tsv",
#                       format=Parameters.DATASET_FORMAT,
#                       fields=fields,
#                       skip_header=True)
#
#
# train_iter = BucketIterator(test,
#                             batch_size=Parameters.BATCH_SIZE,
#                             device=Parameters.DEVICE,
#                             train=True,
#                             shuffle=True)


def create_iterators(data_file_location):
    train, test = TabularDataset(path=data_file_location,
                                 format="TSV",
                                 fields=fields,
                                 skip_header=True).split()

    train_iter = BucketIterator(train,
                                batch_size=Parameters.BATCH_SIZE,
                                device=Parameters.DEVICE,
                                train=True,
                                shuffle=True)

    test_iter = BucketIterator(test,
                               batch_size=Parameters.BATCH_SIZE,
                               device=Parameters.DEVICE,
                               train=True,
                               shuffle=True,
                               sort=False)
    print("Finish dataset prepare")

    return train_iter, test_iter



