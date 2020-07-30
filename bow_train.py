import ast
import torch
import numpy as np
import pandas as pd

from torch import nn, optim
from typing import Dict, List
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from app.embeddings.bag_of_words import BagOfWords
from app.models.bow_classifier import BowClassifier
from app.preprocessing.preprocessor import Preprocessor


class BowMovieSentimentDataset(Dataset):
    """Movie sentiment dataset."""

    def __init__(self, movie_sentiments: pd.DataFrame, embedding: BagOfWords, binary_vectorizer: bool = True):
        """
        Args:
            csv_file (string): Path to the csv file with sentiments.
        """
        self.movie_sentiments = movie_sentiments.copy()
        self.movie_sentiments["review"] = self.movie_sentiments["review"]
        self.embedding = embedding
        self.binary_vectorizer = binary_vectorizer

    @classmethod
    def from_csv(cls, csv_file: str, embedding: BagOfWords, binary_vectorizer: bool = True):
        df = pd.read_csv(csv_file).apply(ast.literal_eval)
        return cls(df, embedding, binary_vectorizer)

    def __len__(self):
        return len(self.movie_sentiments)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        review = self.movie_sentiments.iloc[idx, 0]
        t = self.embedding.spread_indices(review, self.binary_vectorizer)

        sentiment = self.movie_sentiments.iloc[idx, 1]

        sample = {'review': t, 'sentiment': sentiment}

        return sample


def train_epoch(model: nn.Module, dataloader: DataLoader, embedding, loss_function, optimizer: optim.Optimizer) -> float:
    train_loss_epoch, n = 0.0, 0
    # l1_lambda = 0.0001
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
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")

    train_df["review"] = Preprocessor.remove_symbols(train_df["review"])
    test_df["review"] = Preprocessor.remove_symbols(test_df["review"])
    print("Removed Symbols")

    embedding = BagOfWords.from_pandas(train_df["review"])
    vocab_size = len(embedding.vocab)
    print("Vocab Size: {}".format(vocab_size))

    train_df["review"] = embedding._embed_series(train_df["review"])
    test_df["review"] = embedding._embed_series(test_df["review"])

    train_df, validation_df = train_test_split(train_df, train_size=0.8)

    train_dataset = BowMovieSentimentDataset(train_df, embedding=embedding, binary_vectorizer=True)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    print("Created Train Batches")
    validation_dataset = BowMovieSentimentDataset(validation_df, embedding=embedding, binary_vectorizer=True)
    validation_loader = DataLoader(validation_dataset, batch_size=256, shuffle=True, num_workers=4)
    print("Created Validation Batches")
    test_dataset = BowMovieSentimentDataset(test_df, embedding=embedding, binary_vectorizer=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=True, num_workers=4)
    print("Created Test Batches")
    # validation_loader = test_loader

    # Set up a bag of words model and training
    model = BowClassifier(vocab_size)
    # model = DeepBowClassifier(vocab_size, 128)

    # checkpoint_loc = "checkpoints/bow_model"
    # model = torch.load(checkpoint_loc)
    loss = nn.BCEWithLogitsLoss()
    num_epochs = 3
    # lr = 0.1
    # optimizer = optim.SGD(model.parameters(), lr)
    lr = 1e-2
    optimizer = optim.Adam(model.parameters(), lr=lr)

    results = test(model, validation_loader, embedding, loss)
    print("Initial Validation Accuracy: {}".format(results["accuracy"]))
    for epoch in range(num_epochs):
        train_loss_epoch = train_epoch(model, train_loader, embedding, loss, optimizer)
        results = test(model, validation_loader, embedding, loss)
        print("Epoch: {}, Train Loss: {}, Validation Loss: {}, Validation Acc: {}".format(epoch+1, train_loss_epoch, results["loss"], results["accuracy"]))

    checkpoint_loc = "bow_model.pt"
    torch.save(model, checkpoint_loc)
    print("Saved Model")
    embedding.store_vocab("data/bow_vocab.txt")
    print("Stored vocab")
    final_results = test(model, test_loader, embedding, loss)
    print("Test Acc: {}".format(final_results["accuracy"]))
    # evaluation(model, embedding)
