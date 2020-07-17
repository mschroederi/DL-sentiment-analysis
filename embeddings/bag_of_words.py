import torch
import numpy as np
import pandas as pd
from torch import Tensor
from typing import List, Set
from collections import Counter

from embeddings.label_encoder import LabelEncoderExt

SYMBOLS_TO_REMOVE = [".", "\"", "(", ")", ",", "?", "!", "'", ";", "{", "}", "-", "*", "=", ":", "\x91", "\x97", "<br />", "/", "<", ">"]


class BagOfWords:

    def __init__(self, vocab: Set[str], embedding):
        self.vocab = vocab
        self.embedding = embedding

    @classmethod
    def from_vocab(cls, vocab: Set[str]):
        embedding = LabelEncoderExt()
        embedding.fit(list(vocab))
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

    def spread_indices(self, indices: List[int], with_count: bool = True) -> np.array:
        pad = np.zeros(len(self.vocab))
        if len(indices) == 0:
            return pad

        if with_count:
            count = Counter(indices)
            pad[list(count.keys())] = np.array(list(count.values()))
        else:
            pad[indices] = 1
            
        return pad

    def _embed_series(self, texts: pd.Series, show_progress: bool = False) -> pd.Series:
        if show_progress:
            return texts.str.split().progress_apply(self.embedding.transform)
        
        return texts.str.split().apply(self.embedding.transform)

    def create_tensor(self, texts: List[str]) -> Tensor:
        s = pd.Series(texts)
        preprocessed = self.preprocess(s)
        embedded = self._embed_series(preprocessed)
        self.spread_indices()

    @staticmethod
    def preprocess_text(text: str):
        for symbol in SYMBOLS_TO_REMOVE:
            text = text.replace(symbol, " ")
        text = " ".join([w for w in text.split() if w])
        return text.lower()

    @staticmethod
    def preprocess(texts: pd.Series) -> pd.Series:
        return texts.str.lower().apply(BagOfWords.preprocess_text)
