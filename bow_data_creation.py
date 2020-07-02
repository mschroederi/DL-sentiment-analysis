import pandas as pd
from tqdm import tqdm

from embeddings.bag_of_words import BagOfWords

def store(df: pd.DataFrame, embedding, file: str):
    df["review"] = embedding._embed_series(df["review"], show_progress=True)
    df["review"] = df["review"].apply(lambda indices: "[{}]".format(",".join(indices.astype(str))))
    train.to_csv(file, index=False)

if __name__ == "__main__":
    train = pd.read_csv("./data/train.csv")
    test = pd.read_csv("./data/test.csv")
    
    train["review"] = BagOfWords.preprocess(train["review"])
    test["review"] = BagOfWords.preprocess(test["review"])

    embedding = BagOfWords.from_pandas(train["review"], test["review"])
    vocab_size = len(embedding.vocab)
    print("Vocab Size: {}".format(vocab_size))

    embedding.store_vocab("./data/bow_vocab.txt")

    tqdm.pandas()
    store(train, embedding, "./data/bow_train.csv")
    print("Stored Train Embedding")
    store(test, embedding, "./data/bow_test.csv")
    print("Stored Test Embedding")
