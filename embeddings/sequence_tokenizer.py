import torch
import numpy as np
import pandas as pd
from torch import Tensor
from sklearn import preprocessing
from typing import List, Set

class SequenceTokenizer:

    def __init__(self):
        self.vocab = dict({"EOS": 0})
        self.padding_size = -1

    def fit(self, reviews: pd.Series):
        for review in reviews:
            words_in_review = review.split()
            if len(words_in_review) > self.padding_size:
                self.padding_size = len(words_in_review)
            for word in words_in_review:
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)
        self.vocab_size = len(self.vocab)

    def __transform_single_review(self, review: str):
        tokenized = np.array([self.vocab[word] for word in review.split() if word in self.vocab])
        if len(tokenized) < self.padding_size:
            tokenized = np.append(np.array([0 for _ in range(self.padding_size - len(tokenized))]), tokenized)
        return torch.from_numpy(tokenized).long()

    def transform(self, reviews: pd.Series):
        return reviews.apply(self.__transform_single_review)

    def fit_transform(self, reviews: pd.Series) -> pd.Series:
        self.fit(reviews)
        return self.transform(reviews)