import torch
import numpy as np
import pandas as pd
from torch import Tensor
from sklearn import preprocessing
from typing import List, Set

class OneHotSequenceEncoder:

    def __init__(self, max_vocab_size: int = -1):
        self.vocab = dict({"EOS": 0})
        self.max_vocab_size = max_vocab_size # TODO: not respected yet
        self.embedding_size = -1
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
        self.embedding_size = 1# TODO: len(self.vocab)

    def __to_one_hot(self, idx: int):
        return idx # TODO: currently using to much RAM -> use sparse matrix
        #return np.eye(self.vocab_size)[idx]

    def __transform_single_review(self, review: str):
        tokenized = np.array([self.__to_one_hot(self.vocab[word]) for word in review.split()]).reshape(-1, self.embedding_size)
        if len(tokenized) < self.padding_size:
            tokenized = np.append(tokenized, np.array([self.__to_one_hot(0) for _ in range(self.padding_size - len(tokenized))]))
        return torch.from_numpy(tokenized).float().reshape(self.padding_size, self.embedding_size)

    def transform(self, reviews: pd.Series):
        return reviews.apply(self.__transform_single_review)

    def fit_transform(self, reviews: pd.Series) -> pd.Series:
        self.fit(reviews)
        return self.transform(reviews)
