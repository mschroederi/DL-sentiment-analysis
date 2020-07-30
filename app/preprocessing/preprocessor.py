import pandas as pd

SYMBOLS_TO_REMOVE = [".", "\"", "(", ")", ",", "?", "!", "'", ";", "{", "}", "-", "*", "=", ":", "\x91", "\x97", "<br />", "/", "<", ">"]


class Preprocessor:
    @staticmethod
    def preprocess_text(text: str):
        for symbol in SYMBOLS_TO_REMOVE:
            text = text.replace(symbol, " ")
        text = " ".join([w for w in text.split() if w])
        return text.lower()

    @staticmethod
    def remove_symbols(review_texts: pd.Series) -> pd.Series:
        return review_texts.str.lower().apply(Preprocessor.preprocess_text)
    
    @staticmethod
    def remove_long_sequences(df: pd.DataFrame, max_len: int) -> pd.Series:
        seq_lengths = df["review"].apply(lambda text: len(text.split()))
        return df[seq_lengths <= max_len]

