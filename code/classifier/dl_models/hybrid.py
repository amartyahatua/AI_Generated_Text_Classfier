import torch
import torch.nn as nn
from torch.autograd import Variable
from AI_Generated_Text_Classfier.code.classifier import config


class HybridClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, label_size, extracted_feature_size, batch_size, dropout=0.2):
        super(HybridClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True)
        self.hidden2label = nn.Linear(hidden_dim * 2, extracted_feature_size)
        self.hybridLayer = nn.Linear(extracted_feature_size, label_size)
        self.hidden = self.init_hidden()
        self.softmax = nn.Softmax(dim=1)

    def init_hidden(self):
        # first is the hidden h
        # second is the cell c

        return (Variable(torch.zeros(2, self.batch_size, self.hidden_dim)),
                Variable(torch.zeros(2, self.batch_size, self.hidden_dim)))

    def forward(self, x):
        lstm_out, self.hidden = self.lstm(x)
        hidden_out = self.hidden2label(lstm_out)
        hybrid_output = self.hybridLayer(hidden_out)
        output = self.dropout(hybrid_output)
        return output


hybrid_model = HybridClassifier(embedding_dim=config.FEATURE_DIMENSION, hidden_dim=config.HIDDEN_LAYER,
                                label_size=1, extracted_feature_size=config.EXTRACTED_FEATURES,
                                batch_size=config.BATCH_SIZE)
hybrid_model.hybridLayer.requires_grad_(False)
print(hybrid_model)
