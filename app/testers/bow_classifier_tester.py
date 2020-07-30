import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from app.data_loading.bow_data_loading import BowMovieSentimentDataset
from app.embeddings.bag_of_words import BagOfWords
from app.models.bow_classifier import BowClassifier
from app.preprocessing.preprocessor import Preprocessor
from app.trainers.bow_classifier_trainer import test


class BowClassifierTester:
    @staticmethod
    def test(test_data_path: str = 'data/test.csv',
             model_checkpoint_path: str = "bow_model.pt",
             vocab_path: str = "data/bow_vocab.txt"):
        test_df = pd.read_csv(test_data_path)
        test_df["review"] = Preprocessor.remove_symbols(test_df["review"])
        embedding = BagOfWords.from_vocab_file(vocab_path)
        vocab_size = len(embedding.vocab)
        print("Vocab Size: {}".format(vocab_size))

        test_df["review"] = embedding._embed_series(test_df["review"])
        test_dataset = BowMovieSentimentDataset(test_df, embedding=embedding, binary_vectorizer=True)
        test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=True, num_workers=4)
        print("Created Test Batches")

        loss = nn.BCEWithLogitsLoss()
        model: BowClassifier = torch.load(model_checkpoint_path)
        final_results = test(model, test_loader, embedding, loss)
        print("Test Acc: {}".format(final_results["accuracy"]))