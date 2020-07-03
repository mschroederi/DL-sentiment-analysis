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
        # print("len(reviews): ", len(reviews))
        # print("reviews.shape: ", reviews.shape)
        reviews_embedded = self.embbedding(reviews)
        # print("reviews_embedded.shape: ", reviews_embedded.shape)
        reviews_embedded_permuted = reviews_embedded.permute(1, 0, 2)
        # print("reviews_embedded_permuted.shape: ", reviews_embedded_permuted.shape)
        lstm_output, lstm_hidden = self.rnn(reviews_embedded_permuted)
        # print("lstm_output.shape: ", lstm_output.shape)
        # print("lstm_hidden[0].shape: ", lstm_hidden[0].shape)
        #last_output = output[-1,:,:]
        #print("last_output.shape: ", last_output.shape)
        return self.out(lstm_hidden[0].reshape(-1, self.hidden_size))
