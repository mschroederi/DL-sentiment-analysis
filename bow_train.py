import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List

from data_loader import MovieSentimentDataset
from embeddings.bag_of_words import BagOfWords
from models.bow_classifier import BowClassifier


def train_epoch(model: nn.Module, dataloader: DataLoader, embedding, loss_function, optimizer: optim.Optimizer) -> float:
    train_loss_epoch, n = 0.0, 0
    l1_lambda = 0.01
    model.train()
    for i_batch, sample_batched in enumerate(dataloader):
        y = sample_batched["sentiment"].type(torch.FloatTensor)
        y = torch.unsqueeze(y, 1)
        bow = embedding.embed(sample_batched["review"])
        y_hat = model(bow)
        
        # Integrate an L1 loss
        epoch_weights = next(model.parameters())
        l1_loss = torch.norm(epoch_weights, 1)
        l = loss(y_hat, y) + l1_lambda * l1_loss
        train_loss_epoch += l.item()
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        n += len(sample_batched)

    train_loss_epoch /= n
    return train_loss_epoch


def test(model: nn.Module, dataloader: DataLoader, embedding, loss_function, optimizer: optim.Optimizer) -> Dict[str, float]:
    pass

def evaluation(model: nn.Module, embedding) -> None:
    # Find the words that are responsible for sentiment
    param = next(model.parameters())
    transform = embedding.embedding.inverse_transform

    def words_to_prob(k: int, largest: bool, absolute: bool = False) -> List[str]:
        my_param = param.abs() if absolute else param
        values, indices = torch.topk(my_param, k, largest=largest)
        values = values.squeeze(dim=0).tolist()
        words = transform(indices[0].tolist())
        word_to_prob = ["{} ({})".format(word, np.round(prob, 4)) for (word, prob) in zip(words, values)]
        return word_to_prob
    
    k = 10
    pos_sentiment = words_to_prob(k, True)
    neg_sentiment = words_to_prob(k, False)
    no_sentiment = words_to_prob(k, False, True)
    print("Words that speak for a good review:")
    print(", ".join(pos_sentiment))
    print("Words that speak for a bad review:")
    print(", ".join(neg_sentiment))
    print("Words that have no sentiment:")
    print(", ".join(no_sentiment))


if __name__ == "__main__":
    dataset = MovieSentimentDataset(csv_file="data/train.csv")
    # Restrict the number of reviews for now due to long run time
    dataset.movie_sentiments = dataset.movie_sentiments.sample(100)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=1)

    # Preprocess reviews and create a bag of words embedding
    reviews = dataset.movie_sentiments["review"]
    reviews = BagOfWords.preprocess(reviews)
    dataset.movie_sentiments["review"] = reviews
    embedding = BagOfWords.from_pandas(reviews)
    vocab_size = len(embedding.vocab)
    print("Vocab Size: {}".format(vocab_size))

    # Set up a bag of words model and training
    model = BowClassifier(vocab_size)
    loss = nn.BCELoss()
    num_epochs = 3
    lr = 0.1
    optimizer = optim.SGD(model.parameters(), lr)

    for epoch in range(num_epochs):
        train_loss_epoch = train_epoch(model, dataloader, embedding, loss, optimizer)
        print("Epoch: {}, Train Loss: {}".format(epoch+1, train_loss_epoch))

    evaluation(model, embedding)
