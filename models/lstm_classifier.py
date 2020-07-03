import torch
from torch import nn


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, padding_size: int, embedding_size: int, hidden_size: int):
        super(LSTMClassifier, self).__init__()
        self.vocab_size = vocab_size
        self.padding_size = padding_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.embbedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size, padding_idx=0)
        self.rnn = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size)

        self.out = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.ReLU(),
            nn.Linear(32, 8), nn.ReLU(),
            nn.Linear(8, 1), nn.Sigmoid()
        )

    def forward(self, reviews):
        reviews_embedded = self.embbedding(reviews)
        reviews_embedded_permuted = reviews_embedded.permute(1, 0, 2)
        lstm_output, lstm_hidden = self.rnn(reviews_embedded_permuted)
        return self.out(lstm_hidden[0].reshape(-1, self.hidden_size))
