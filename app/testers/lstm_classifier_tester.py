import torch
from torch import nn
from torch.utils.data import DataLoader

from app.data_loading.data_loading import MovieSentimentDataset, MovieSentimentDatasetBuilder
from app.embeddings.sequence_tokenizer import SequenceTokenizer
from app.models.lstm_classifier import LSTMClassifier
from app.preprocessing.preprocessing_pipeline import execute_preprocessing_pipeline

class LSTMClassifierTester:
    def __init__(self):
        super().__init__()

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("Running on the GPU")
        else:
            self.device = torch.device("cpu")
            print("Running on the CPU")

        self.dataset_test = None
        self.padding_size = None


    def __proprocess_dataset(self, vocab_path: str):
        print("Starting preprocessing pipeline...")
        if self.dataset_test is None or self.padding_size is None:
            raise Exception("No data available for preprocessing.")

        tokenizer = SequenceTokenizer.from_vocab_file(vocab_path, self.padding_size)
        execute_preprocessing_pipeline(self.dataset_test, tokenizer=tokenizer)
        print("Completed preprocessing pipeline.")


    def __evaluate(self, model):
        dataloader_test = DataLoader(self.dataset_test, batch_size=256, shuffle=False, num_workers=1)

        test_loss, test_acc, n = 0, 0, 0
        loss = nn.BCELoss()
        model.eval()
        with torch.no_grad():
            for i_batch, sample_batched in enumerate(dataloader_test):
                y = sample_batched["sentiment"].type(torch.FloatTensor).to(self.device).reshape(-1, 1)
                y_hat = model(sample_batched["review"].to(self.device).reshape(-1, self.padding_size))
                l = loss(y_hat, y)
                test_loss += l.item()
                test_acc += (y == (y_hat > .5).type(torch.FloatTensor).to(self.device)).sum().item()
                n += len(y)
            
        test_loss /= n
        test_acc /= n
        print("Test Loss: {}, Test acc: {}".format(test_loss, test_acc))


    def test(self, test_data_path: str = 'data/test.csv', model_checkpoint_path: str = "best_model.pt", vocab_path: str = "vocab.txt"):
        self.dataset_test = MovieSentimentDatasetBuilder\
            .from_csv(csv_file=test_data_path)\
            .build()

        model: LSTMClassifier = torch.load(model_checkpoint_path)
        self.padding_size = model.padding_size

        self.__proprocess_dataset(vocab_path)
        self.__evaluate(model)

    