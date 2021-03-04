from transformers import BertForSequenceClassification
import torch
import torch.nn as nn
import Parameters

class BERT(nn.Module):

    def __init__(self):
        super(BERT, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(Parameters.BERT_TOKENIZER_NAME)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.bert.classifier = nn.Linear(in_features=768, out_features=2)
        self.bert.classifier.requires_grad = True


    def forward(self, text):
        cls_hs = self.bert(text)[0]
        return cls_hs

