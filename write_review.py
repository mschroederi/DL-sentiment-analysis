import argparse
from tkinter import *

import numpy as np
import torch

from app.embeddings.sequence_tokenizer import SequenceTokenizer
from app.models.lstm_classifier import LSTMClassifier
from app.preprocessing.preprocessor import Preprocessor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-checkpoint', type=str, default="model.pt", dest="model_checkpoint",
                        help='File location where the model from training is stored.')
    parser.add_argument('--vocab-checkpoint', type=str, default="data/vocab.txt", dest="vocab_checkpoint",
                        help='File location where the vocabulary of the sequence tokenizer from training is stored.')

    return parser.parse_args()


def evaluate(event):
    review = entry.get("1.0", END)
    preprocessed = Preprocessor.preprocess_text(review)
    tokenized = tokenizer._SequenceTokenizer__transform_single_review(preprocessed)

    with torch.no_grad():
        X = torch.tensor(tokenized).reshape(-1, padding_size)
        y = model(X).reshape(-1, 1)
        prob = y.flatten().item()
    percentage = np.round(prob * 100, 2)

    res.config(text="Positive Review: {}%".format(percentage))


args = parse_args()

model: LSTMClassifier = torch.load(args.model_checkpoint)
padding_size = model.padding_size
tokenizer = SequenceTokenizer.from_vocab_file(args.vocab_checkpoint, padding_size)
model.eval()

w = Tk()
Label(w, text="Type your review:").pack()
entry = Text(w)
entry.bind("<KeyRelease>", evaluate)
entry.pack()
res = Label(w)
res.pack()
w.mainloop()
