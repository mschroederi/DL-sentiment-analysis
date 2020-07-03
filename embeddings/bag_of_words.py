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
    def from_vocab_file(cls, file: str):
        with open(file, "r") as f:
            vocab = set(f.read().splitlines())
        return cls.from_vocab(vocab)

    @classmethod
    def from_pandas(cls, *args: List[pd.Series]):
        vocab = set()
        for texts in args:
            texts.str.lower().str.split().apply(vocab.update)
        return cls.from_vocab(vocab)

    def store_vocab(self, file: str):
        with open(file, "w") as f:
            f.write("\n".join(list(self.vocab)))

    def spread_indices(self, indices: List[int]) -> np.array:
        pad = np.zeros(len(self.vocab))
        pad[indices] = 1
        return pad

    def _embed_series(self, texts: pd.Series, show_progress: bool = False) -> pd.Series:
        if show_progress:
            return texts.str.split().progress_apply(self.embedding.transform)
        
        return texts.str.split().apply(self.embedding.transform)

    def embedding_to_tensor(self, embedded_texts: pd.Series) -> Tensor:
        df = pd.DataFrame(columns=[0])
        tensors = embedded_texts.map(lambda x: Tensor(spread_indices(x)))
        return torch.stack(tensors)

    def _embed_to_dataframe(self, texts: List[str]) -> pd.DataFrame:
        columns = np.arange(0, len(self.vocab))
        df = pd.DataFrame(columns=[0])

        for i, text in enumerate(texts):
            df.loc[i, 0] = self.embedding.transform(text.split()) # self.padding(text)
        return df

    def embed(self, texts: List[str]) -> Tensor:
        df = self._embed_to_dataframe(texts)
        return torch.Tensor(df.values)

    @staticmethod
    def preprocess(texts: pd.Series) -> pd.Series:
        def preprocess_text(text: str):
            for symbol in SYMBOLS_TO_REMOVE:
                text = text.replace(symbol, " ")
            text = " ".join([w for w in text.split() if w])
            return text.lower()

        return texts.str.lower().apply(preprocess_text)
