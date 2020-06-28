import pandas as pd
from os import listdir, path

from typing import List

# Read individual review files
def read_raw_reviews(data_path: str) -> List[str]:
    reviews = []
    for file_name in listdir(data_path):
        file = open(path.join(data_path, file_name), "r+")
        reviews.append(file.read())
    return reviews

# Create pd.DataFrame from reviews with the provided label
def reviews_to_df(reviews: List[str], label: int) -> pd.DataFrame:
    labels = [1 for _ in range(len(reviews))]
    return pd.DataFrame(list(zip(reviews, labels)), columns =['review', 'label'])

# Build pd.DataFrame containing positive and negative reviews
def load_reviews(data_path: str) -> pd.DataFrame:
    pos_reviews = reviews_to_df(read_raw_reviews(path.join(data_path, 'pos')), 1)
    neg_reviews = reviews_to_df(read_raw_reviews(path.join(data_path, 'neg')), 0)
    return pd.concat([pos_reviews, neg_reviews])

# load train data
train_df = load_reviews('./data/train')
# load test data
test_df = load_reviews('./data/test')

# Write to CSV files
train_df.to_csv('./train.csv', index=False)
test_df.to_csv('./test.csv', index=False)