from transformers import BertForSequenceClassification
import torch.nn as nn
import Parameters


class BERT(nn.Module):

    def __init__(self):
        super(BERT, self).__init__()
        self.encoder = BertForSequenceClassification.from_pretrained(Parameters.BERT_TOKENIZER_NAME)

    def forward(self, text, label):
        loss, text_fea = self.encoder(text, labels=label)[:2]

        return loss, text_fea
