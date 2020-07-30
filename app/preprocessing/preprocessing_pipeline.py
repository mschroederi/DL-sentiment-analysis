from app.data_loading.data_loading import MovieSentimentDataset
from app.preprocessing.preprocessor import Preprocessor
from app.embeddings.sequence_tokenizer import SequenceTokenizer


def execute_preprocessing_pipeline(dataset: MovieSentimentDataset, tokenizer=None):
    reviews = dataset.movie_sentiments["review"]
    dataset.movie_sentiments["review"] = Preprocessor.remove_symbols(dataset.movie_sentiments["review"])
    dataset.movie_sentiments = Preprocessor.remove_long_sequences(dataset.movie_sentiments, max_len=1000)

    if tokenizer is None:
        tokenizer = SequenceTokenizer()
        tokenizer.fit(dataset.movie_sentiments["review"])
    dataset.movie_sentiments["review"] = tokenizer.transform(dataset.movie_sentiments["review"])
    return tokenizer