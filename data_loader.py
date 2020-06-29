import os
import io
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class MovieSentimentDataset(Dataset):
    """Movie sentiment dataset."""

    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with sentiments.
        """
        self.movie_sentiments = pd.read_csv(csv_file)

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
