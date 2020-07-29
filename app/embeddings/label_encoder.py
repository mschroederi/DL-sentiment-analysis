import numpy as np

from sklearn.preprocessing import LabelEncoder
from typing import List


class LabelEncoderExt(object):
    def __init__(self):
        self.label_encoder = LabelEncoder()

    def fit(self, words: List[str]):
        """
        """
        self.label_encoder = self.label_encoder.fit(words)
        self.le_dict = dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))
        return self


    def transform(self, words: List[str]):
        """
        This will transform the words and skips every word that hasn't been seen by the label encoder before.
        :param words:
        :return:
        """
        # known_words = [w for w in words if w in self.le_dict]
        # return self.label_encoder.transform(known_words)
        encoded = [self.le_dict[w] for w in words if w in self.le_dict]
        return encoded

    def __getattr__(self, name):
        """
        Pass every method to the underlying label_encoder.
        """
        return getattr(self.label_encoder, name)
