import argparse

import torch
import numpy as np
from tkinter import *

from app.embeddings.bag_of_words import BagOfWords


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-checkpoint', type=str, default="bow_model.pt", dest="model_checkpoint",
                        help='File location where the model from training is stored.')
    parser.add_argument('--vocab-checkpoint', type=str, default="data/bow_vocab.txt", dest="vocab_checkpoint",
                        help='File location where the vocabulary of the tokenizer from training is stored.')

    return parser.parse_args()


def evaluate(event):
    review = entry.get("1.0", END)
    preprocessed = BagOfWords.preprocess_text(review).split()

    known_words = list(filter(lambda x: x in le_dict, preprocessed))
    unknown_words = list(set(preprocessed) - set(known_words))

    embedded = embedding.embedding.transform(known_words)
    spread = embedding.spread_indices(embedded)

    X = torch.Tensor(spread)
    y = model(X)
    prob = torch.sigmoid(y).item()
    percentage = np.round(prob * 100, 2)

    res.config(text = "Positive Review: {}%".format(percentage))
    unknown.config(text = "Unknown Words: " + ", ".join(unknown_words))
    

args = parse_args()
embedding = BagOfWords.from_vocab_file(args.vocab_checkpoint)
le = embedding.embedding
le_dict = dict(zip(le.classes_, le.transform(le.classes_)))
model = torch.load(args.model_checkpoint)
model.eval()

w = Tk()
Label(w, text="Type your review:").pack()
entry = Text(w)
entry.bind("<KeyRelease>", evaluate)
entry.pack()
res = Label(w)
res.pack()
unknown = Label(w)
unknown.pack()
w.mainloop()
