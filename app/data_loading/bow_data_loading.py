import ast

import pandas as pd
import torch
from torch.utils.data import Dataset

from app.embeddings.bag_of_words import BagOfWords


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