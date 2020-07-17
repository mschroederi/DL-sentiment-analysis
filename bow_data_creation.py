import pandas as pd
import numpy as np
from tqdm import tqdm

from embeddings.bag_of_words import BagOfWords
from preprocessing.preprocessor import Preprocessor


def store(df: pd.DataFrame, embedding, file: str):
    df["review"] = embedding._embed_series(df["review"], show_progress=True)
    df["review"] = df["review"].apply(lambda indices: "[{}]".format(",".join(np.array(indices, dtype=str))))
    df.to_csv(file, index=False)

if __name__ == "__main__":
    train = pd.read_csv("./data/train.csv")
    test = pd.read_csv("./data/test.csv")
    
    train["review"] = Preprocessor.remove_symbols(train["review"])
    test["review"] = Preprocessor.remove_symbols(test["review"])
    print("Removed Symbols")

    embedding = BagOfWords.from_pandas(train["review"])
    vocab_size = len(embedding.vocab)
    print("Vocab Size: {}".format(vocab_size))

    embedding.store_vocab("./data/bow_vocab2.txt")

    tqdm.pandas()
    store(train, embedding, "./data/bow_train2.csv")
    print("Stored Train Embedding")
    store(test, embedding, "./data/bow_test2.csv")
    print("Stored Test Embedding")
