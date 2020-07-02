import torch
from torch import nn


class LSTMClassifier(nn.Module):
    def __init__(self, embedding_size: int, hidden_size: int):
        super(LSTMClassifier, self).__init__()

        self.rnn = nn.LSTM(embedding_size, hidden_size)
        self.linear = nn.Linear(hidden_size, 1)
        self.out =  nn.Sigmoid()

        # self.model = nn.Sequential(
        #   nn.LSTM(embedding_size, hidden_size),
        #   nn.Linear(hidden_size, 1),
        #   nn.Sigmoid()
        # )

    def forward(self, reviews):
        output, hidden = self.rnn(reviews)
        last_output = output[:,-1,:]
        o = self.out(self.linear(last_output))
        return o
