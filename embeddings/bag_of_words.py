import torch
import numpy as np
import pandas as pd
from torch import Tensor
from sklearn import preprocessing
from typing import List, Set

SYMBOLS_TO_REMOVE = [".", "\"", "(", ")", ",", "?", "!", "'", ";", "{", "}", "-", "*", "=", ":", "\x91", "\x97", "<br />", "/", "<", ">"]


class BagOfWords:

    def __init__(self, vocab: Set[str], embedding):
        self.vocab = vocab
        self.embedding = embedding

    @classmethod
    def from_vocab(cls, vocab: Set[str]):
        embedding = preprocessing.LabelEncoder()
        embedding.fit_transform(list(vocab))
        return cls(vocab, embedding)

    @classmethod
    def from_pandas(cls, *args: List[pd.Series]):
        vocab = set()
        for texts in args:
            texts.str.lower().str.split().apply(vocab.update)
        return cls.from_vocab(vocab)

    def padding(self, text: str) -> np.array:
        indices = self.embedding.transform(text.split())
        pad = np.zeros(len(self.vocab))
        pad[indices] = 1
        return pad

    def embed(self, texts: List[str]) -> Tensor:
        columns = np.arange(0, len(self.vocab))
        df = pd.DataFrame(columns=columns)

        for i, text in enumerate(texts):
            df.loc[i] = self.padding(text)
        
        return torch.Tensor(df.values)

    @staticmethod
    def preprocess(texts: pd.Series) -> pd.Series:
        def preprocess_text(text: str):
            for symbol in SYMBOLS_TO_REMOVE:
                text = text.replace(symbol, " ")
            text = " ".join([w for w in text.split() if w])
            return text.lower()

        return texts.str.lower().apply(preprocess_text)
