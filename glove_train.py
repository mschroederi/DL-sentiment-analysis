from torch import nn, optim
from typing import Dict

from app.models.bow_classifier import BowClassifier

import torch
import pandas as pd
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
from app.preprocessing.preprocessor import Preprocessor

class GloveMovieSentimentDataset(Dataset):
    """Movie sentiment dataset."""

    def __init__(self, csv_file: str):
        """
        Args:
            csv_file (string): Path to the csv file with sentiments.
        """

        glove_path = 'data'
        words = pickle.load(open(f'{glove_path}/6B.50_words.pkl', 'rb'))
        word2idx = pickle.load(open(f'{glove_path}/6B.50_idx.pkl', 'rb'))
        vectors = pickle.load(open(f'{glove_path}/6B.50_vectors.pkl', 'rb'))
        glove = {w: vectors[word2idx[w]] for w in words}


        self.movie_sentiments = pd.read_csv(csv_file)
        reviews = Preprocessor.remove_symbols(self.movie_sentiments["review"])
        embeddingOfSentences = []


        for review in reviews:
            sentence_embedding = np.zeros(50)
            review_word_splitted = review.split(" ")
            for word in review_word_splitted:
                try: 
                    sentence_embedding = sentence_embedding + glove[word]
                except KeyError:
                    # lets ignore them, because they shouldnt be too important if they arent in the glove dataset...
                    sentence_embedding = sentence_embedding + 0
            sentence_embedding = sentence_embedding / len(review_word_splitted)
            embeddingOfSentences.append(sentence_embedding)
    
        self.embeddings = embeddingOfSentences


    def __len__(self):
        return len(self.movie_sentiments)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

    
        t = self.embeddings[idx]
        sentiment = self.movie_sentiments.iloc[idx, 1]
        sample = {'review': t, 'sentiment': sentiment}
        return sample


def train_epoch(model: nn.Module, dataloader: DataLoader, loss_function, optimizer: optim.Optimizer) -> float:
    train_loss_epoch, n = 0.0, 0
    l1_lambda = 0.001
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


def test(model: nn.Module, dataloader: DataLoader, loss_function) -> Dict[str, float]:
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



if __name__ == "__main__":

    train_dataset = GloveMovieSentimentDataset(csv_file="data/train.csv")
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    print("Created Train Batches")
    
    test_dataset = GloveMovieSentimentDataset(csv_file="data/test.csv")
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=4)
    print("Created Test Batches")

    model = BowClassifier(50)
    loss = nn.BCEWithLogitsLoss()
    num_epochs = 5
    lr = 0.3
    optimizer = optim.SGD(model.parameters(), lr)

    results = test(model, test_loader, loss)
    print("Initial Test Accuracy: {}".format(results["accuracy"]))
    for epoch in range(num_epochs):
        train_loss_epoch = train_epoch(model, train_loader, loss, optimizer)
        results = test(model, test_loader, loss)
        print("Epoch: {}, Train Loss: {}, Test Acc: {}".format(epoch+1, train_loss_epoch, results["accuracy"]))


