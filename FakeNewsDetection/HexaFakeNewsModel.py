from transformers import BertForSequenceClassification
import torch
import torch.nn as nn
import Parameters


class HexaFakeNewsModel(nn.Module):

    def __init__(self, bert_model):
        super(HexaFakeNewsModel, self).__init__()
        self.bert_model = bert_model
        for param in self.bert_model.parameters():
            param.requires_grad = False
        self.bert_model.fc.requires_grad = True
        self.hexa_fc = nn.Linear(in_features=2, out_features=6)
        self.hexa_fc.requires_grad = True

    def forward(self, text):
        x = self.bert_model(text)
        x = torch.sigmoid(x)
        x = self.hexa_fc(x)
        return x
