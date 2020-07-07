import ast
import torch
import numpy as np
import pandas as pd

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from sklearn.model_selection import train_test_split

from embeddings.bag_of_words import BagOfWords
from models.bow_classifier import BowClassifier, DeepBowClassifier

class BowMovieSentimentDataset(Dataset):
    """Movie sentiment dataset."""

    def __init__(self, movie_sentiments: pd.DataFrame, embedding: BagOfWords, with_count: bool = True):
        """
        Args:
            csv_file (string): Path to the csv file with sentiments.
        """
        self.movie_sentiments = movie_sentiments.copy()
        self.movie_sentiments["review"] = self.movie_sentiments["review"].apply(ast.literal_eval)
        self.embedding = embedding
        self.with_count = with_count

    @classmethod
    def from_csv(cls, csv_file: str, embedding: BagOfWords):
        df = pd.read_csv(csv_file)
        return cls(df, embedding)

    def __len__(self):
        return len(self.movie_sentiments)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        review = self.movie_sentiments.iloc[idx, 0]
        t = self.embedding.spread_indices(review, self.with_count)

        sentiment = self.movie_sentiments.iloc[idx, 1]

        sample = {'review': t, 'sentiment': sentiment}

        return sample


def train_epoch(model: nn.Module, dataloader: DataLoader, embedding, loss_function, optimizer: optim.Optimizer) -> float:
    train_loss_epoch, n = 0.0, 0
    l1_lambda = 0.001
    l1_lambda = 0
    model.train()
    for i, sample_batched in enumerate(dataloader):
        progress = np.round(i * 100 / len(dataloader), 2)
        print("Train Epoch Progress: {}%".format(progress), end="\r")
        y = sample_batched["sentiment"].type(torch.FloatTensor)
        y = torch.unsqueeze(y, 1)
        bow = sample_batched["review"].type(torch.FloatTensor)
        y_hat = model(bow)
        
        # Integrate an L1 loss
        epoch_weights = next(model.parameters())
        l1_loss = torch.norm(epoch_weights, 1)
        l = loss(y_hat, y) + l1_lambda * l1_loss
        train_loss_epoch += l.item()
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        n += len(sample_batched["sentiment"])

    train_loss_epoch /= n
    return train_loss_epoch


def test(model: nn.Module, dataloader: DataLoader, embedding, loss_function) -> Dict[str, float]:
    test_loss, matches, n = 0.0, 0, 0
    model.eval()
    with torch.no_grad():
        for _, sample_batched in enumerate(dataloader):
            y = sample_batched["sentiment"].type(torch.FloatTensor)
            y = torch.unsqueeze(y, 1)
            bow = sample_batched["review"].type(torch.FloatTensor)
            y_hat = model(bow)
            
            prediction = y_hat >= 0
            matches += (prediction == y).sum().item()
            l = loss_function(y_hat, y)
            test_loss += l.item()
            n += len(sample_batched["sentiment"])
            
    accuracy = matches / n
    test_loss /= n
    
    return {
        "loss": test_loss,
        "accuracy": accuracy
    }

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
    embedding = BagOfWords.from_vocab_file("data/bow_vocab.txt")
    vocab_size = len(embedding.vocab)
    print("Vocab Size: {}".format(vocab_size))

    df = pd.read_csv("data/bow_train.csv")
    train_df, validation_df = train_test_split(df, train_size=0.8)

    train_dataset = BowMovieSentimentDataset(train_df, embedding=embedding, with_count=False)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    print("Created Train Batches")
    validation_dataset = BowMovieSentimentDataset(validation_df, embedding=embedding, with_count=False)
    validation_loader = DataLoader(validation_dataset, batch_size=256, shuffle=True, num_workers=4)
    print("Created Validation Batches")
    test_dataset = BowMovieSentimentDataset.from_csv(csv_file="data/bow_test.csv", embedding=embedding, with_count=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True, num_workers=4)
    print("Created Test Batches")

    # Set up a bag of words model and training
    # model = BowClassifier(vocab_size)
    model = DeepBowClassifier(vocab_size, 128)

    # checkpoint_loc = "checkpoints/bow_model"
    # model = torch.load(checkpoint_loc)
    loss = nn.BCEWithLogitsLoss()
    num_epochs = 5
    lr = 0.1
    optimizer = optim.SGD(model.parameters(), lr)

    results = test(model, validation_loader, embedding, loss)
    print("Initial Validation Accuracy: {}".format(results["accuracy"]))
    for epoch in range(num_epochs):
        train_loss_epoch = train_epoch(model, train_loader, embedding, loss, optimizer)
        results = test(model, validation_loader, embedding, loss)
        print("Epoch: {}, Train Loss: {}, Validation Acc: {}".format(epoch+1, train_loss_epoch, results["accuracy"]))

    checkpoint_loc = "checkpoints/bow_model"
    torch.save(model, checkpoint_loc)
    print("Saved Model")
    final_results = test(model, test_loader, embedding, loss)
    print("Test Acc: {}".format(final_results["accuracy"]))
    # evaluation(model, embedding)
