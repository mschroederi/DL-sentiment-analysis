import torch
from torch import nn


class BowClassifier(nn.Module):
    
    def __init__(self, vocab_size: int):
        super(BowClassifier, self).__init__()
        self.linear = nn.Linear(vocab_size, 1, bias=False)

    def forward(self, bow):
        o = self.linear(bow)
        return o


class DeepBowClassifier(nn.Module):

    def __init__(self, vocab_size: int, hidden_size: int):
        super(DeepBowClassifier, self).__init__()
        self.out = nn.Sequential(
            nn.Linear(vocab_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, bow):
        o = self.out(bow)
        return o
