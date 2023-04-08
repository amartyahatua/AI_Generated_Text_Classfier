import torch
import torch.nn as nn
from torch.autograd import Variable

class BiLSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, label_size, batch_size, dropout=0.2):
        super(BiLSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True)
        self.hidden2label = nn.Linear(hidden_dim*2, label_size)
        self.hidden = self.init_hidden()
        self.softmax = nn.Softmax(dim=1)

    def init_hidden(self):
        # first is the hidden h
        # second is the cell c

        return (Variable(torch.zeros(2, self.batch_size, self.hidden_dim)),
                Variable(torch.zeros(2, self.batch_size, self.hidden_dim)))

    def forward(self, x):
        lstm_out, self.hidden = self.lstm(x)
        output = self.hidden2label(lstm_out)
        output = self.dropout(output)
        return output