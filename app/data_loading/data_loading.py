import torch
import numpy as np
import pandas as pd
from os import listdir, path

from typing import List
import torch
from torch.utils.data import Dataset


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
        # review  = np.array([review])

        sentiment = self.movie_sentiments.iloc[idx, 1]
        # sentiment  = np.array([sentiment])

        sample = {'review': review, 'sentiment': sentiment}

        return sample


class MovieSentimentDatasetBuilder:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.train_validation_split = None

    @staticmethod
    def from_csv(csv_file: str):
        return MovieSentimentDatasetBuilder(data=pd.read_csv(csv_file))

    def with_train_validation_split(self, splits: (float, float) = [.8, .2]):
        self.train_validation_split = splits
        return self

    def build(self):
        if self.train_validation_split is None:
            return MovieSentimentDataset(data=self.data)
        else:
            msk = np.random.rand(len(self.data)) < self.train_validation_split[0]
            train = self.data[msk]
            validation = self.data[~msk]
            return MovieSentimentDataset(data=train), MovieSentimentDataset(data=validation)


# Read individual review files
def read_raw_reviews(data_path: str) -> List[str]:
    reviews = []
    for file_name in listdir(data_path):
        file = open(path.join(data_path, file_name), "r+")
        reviews.append(file.read())
    return reviews


# Create pd.DataFrame from reviews with the provided label
def reviews_to_df(reviews: List[str], label: int) -> pd.DataFrame:
    labels = [label for _ in range(len(reviews))]
    return pd.DataFrame(list(zip(reviews, labels)), columns=['review', 'sentiment'])


# Build pd.DataFrame containing positive and negative reviews
def load_reviews(data_path: str) -> pd.DataFrame:
    pos_reviews = reviews_to_df(read_raw_reviews(path.join(data_path, 'pos')), 1)
    neg_reviews = reviews_to_df(read_raw_reviews(path.join(data_path, 'neg')), 0)
    return pd.concat([pos_reviews, neg_reviews])


if __name__ == "__main__":
    # load train data
    train_df = load_reviews('./aclImdb_v1/train')
    # load test data
    test_df = load_reviews('./aclImdb_v1/test')

    # Write to CSV files
    train_df.to_csv('./data/train.csv', index=False)
    test_df.to_csv('./data/test.csv', index=False)
