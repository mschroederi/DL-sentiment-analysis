from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader

from app.data_loading.bow_data_loading import BowMovieSentimentDataset
from app.embeddings.bag_of_words import BagOfWords
from app.models.bow_classifier import BowClassifier
from app.preprocessing.preprocessor import Preprocessor


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
        l = loss_function(y_hat, y) + l1_lambda * l1_loss
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


class BowClassifierTrainer:
    @staticmethod
    def train(train_data_path: str = 'data/train.csv',
              model_checkpoint_path: str = 'bow_model.pt',
              vocab_checkpoint_path: str = 'data/bow_vocab.txt',
              num_epochs: int = 3):
        train_df = pd.read_csv(train_data_path)
        train_df["review"] = Preprocessor.remove_symbols(train_df["review"])
        print("Removed Symbols")

        embedding = BagOfWords.from_pandas(train_df["review"])
        vocab_size = len(embedding.vocab)
        print("Vocab Size: {}".format(vocab_size))

        train_df["review"] = embedding._embed_series(train_df["review"])
        train_df, validation_df = train_test_split(train_df, train_size=0.8)

        train_dataset = BowMovieSentimentDataset(train_df, embedding=embedding, binary_vectorizer=True)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
        print("Created Train Batches")
        validation_dataset = BowMovieSentimentDataset(validation_df, embedding=embedding, binary_vectorizer=True)
        validation_loader = DataLoader(validation_dataset, batch_size=256, shuffle=True, num_workers=4)
        print("Created Validation Batches")

        # Set up a bag of words model and training
        model = BowClassifier(vocab_size)
        # model = DeepBowClassifier(vocab_size, 128)

        loss = nn.BCEWithLogitsLoss()
        lr = 1e-2
        optimizer = optim.Adam(model.parameters(), lr=lr)

        results = test(model, validation_loader, embedding, loss)
        print("Initial Validation Accuracy: {}".format(results["accuracy"]))
        for epoch in range(num_epochs):
            train_loss_epoch = train_epoch(model, train_loader, embedding, loss, optimizer)
            results = test(model, validation_loader, embedding, loss)
            print("Epoch: {}, Train Loss: {}, Validation Loss: {}, Validation Acc: {}".format(epoch+1, train_loss_epoch, results["loss"], results["accuracy"]))

        torch.save(model, model_checkpoint_path)
        print("Saved Model")
        embedding.store_vocab(vocab_checkpoint_path)
        print("Stored vocab")
        evaluation(model, embedding)