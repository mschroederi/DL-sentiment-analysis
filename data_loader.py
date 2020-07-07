import os
import io
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class MovieSentimentDataset(Dataset):
    """Movie sentiment dataset."""

    def __init__(self, data: pd.DataFrame, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with sentiments.
        """
        self.movie_sentiments = data

    def __len__(self):
        return len(self.movie_sentiments)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        review = self.movie_sentiments.iloc[idx, 0]
        #review  = np.array([review])

        sentiment = self.movie_sentiments.iloc[idx, 1]
        #sentiment  = np.array([sentiment])

        sample = {'review': review, 'sentiment': sentiment}

        return sample


class MovieSentimentDatasetBuilder:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.train_validation_split = None

    @staticmethod
    def from_csv(csv_file: str):
        return MovieSentimentDatasetBuilder(data=pd.read_csv(csv_file))
    
    def with_train_validation_split(self, splits: (float, float)=[.8, .2]):
        self.train_validation_split = splits
        return self
    
    def build(self):
        if self.train_validation_split is None:
            return MovieSentimentDataset(data=self.data)
        else:
            msk = np.random.rand(len(self.data)) < self.train_validation_split[0]
            train = self.data[msk]
            validation = self.data[~msk]
            return (MovieSentimentDataset(data=train), MovieSentimentDataset(data=validation))