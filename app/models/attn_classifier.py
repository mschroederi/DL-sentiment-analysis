import torch
from torch import nn

class AttentionRNNClassifier(nn.Module):
    """
        This is just an experimental implementation of an attention based RNN classifier and not referenced in our paper.
        But first experiments showed promising results in performance on test data.
    """
    def __init__(self, vocab_size: int, padding_size: int, embedding_size: int, hidden_size: int, attn_encoder_hidden_size: int=32):
        super(AttentionRNNClassifier, self).__init__()
        self.vocab_size = vocab_size
        self.padding_size = padding_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.attn_encoder_hidden_size = attn_encoder_hidden_size

        self.embbedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size, padding_idx=0)
        self.seq_encoder = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, dropout=1, bidirectional=True)
        self.seq_encorder_attn = nn.Sequential(nn.Linear(2*self.hidden_size, self.padding_size), nn.Softmax())

        self.encoder_with_attn = nn.GRU(input_size=2*self.hidden_size, hidden_size=self.attn_encoder_hidden_size, dropout=0, bidirectional=False)
        self.out = nn.Sequential(
            nn.Linear(self.attn_encoder_hidden_size + 2 * self.hidden_size, 40), nn.ReLU(),
            nn.Linear(40, 4), nn.ReLU(),
            nn.Linear(4, 1), nn.Sigmoid())

    def forward(self, reviews):
        reviews_embedded = self.embbedding(reviews)
        reviews_embedded_permuted = reviews_embedded.permute(1, 0, 2)
        seq_encoder_output, seq_encoder_hidden = self.seq_encoder(reviews_embedded_permuted)

        seq_encoder_hidden = seq_encoder_hidden.permute([1,0,2]).reshape(-1, 2*self.hidden_size)

        attention_weights = self.seq_encorder_attn(seq_encoder_hidden)
        attention_weights = attention_weights.permute(1, 0).unsqueeze(dim=2)

        seq_encoder_output = seq_encoder_output * self.padding_size
        outputs_attn_applied = torch.mul(attention_weights, seq_encoder_output)

        _, attn_encoded_hidden = self.encoder_with_attn(outputs_attn_applied)
        attn_encoded_hidden = attn_encoded_hidden[0]

        seq_encoder_hidden = seq_encoder_hidden.reshape(-1, 2*self.hidden_size)

        seq_hidden_attn_hidden = torch.cat((attn_encoded_hidden, seq_encoder_hidden), dim=1)

        return self.out(seq_hidden_attn_hidden)