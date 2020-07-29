import torch
import numpy as np
from tkinter import *

from app.embeddings.bag_of_words import BagOfWords


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
    

embedding = BagOfWords.from_vocab_file("data/bow_vocab.txt")
le = embedding.embedding
le_dict = dict(zip(le.classes_, le.transform(le.classes_)))
checkpoint_loc = "checkpoints/bow_model"
model = torch.load(checkpoint_loc)
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

# if __name__ == "__main__":
#     x = np.zeros(10)
#     x[[1,5,9]] = [1, 2, 3]
#     print(x)
