from transformers import BertForSequenceClassification
import torch.nn as nn
import Parameters


class BERT(nn.Module):

    def __init__(self):
        super(BERT, self).__init__()
        self.encoder = BertForSequenceClassification.from_pretrained(Parameters.BERT_TOKENIZER_NAME, num_labels=2)

    def forward(self, text, label):
        output = self.encoder(text, labels=label)
        loss, text_fea = output[:2]

        return loss, text_fea
