import torch
from torch import nn


class BowClassifier(nn.Module):
    
    def __init__(self, vocab_size: int):
        super(BowClassifier, self).__init__()
        self.linear = nn.Linear(vocab_size, 1)

    def forward(self, bow):
        o = self.linear(bow)
        return o
