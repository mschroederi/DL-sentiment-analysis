import pandas as pd


df = pd.read_csv("data/train.csv")

with open("data/train_corpus.txt", "w") as f:
    for review in df["review"]:
        f.write("{}\n".format(review))
