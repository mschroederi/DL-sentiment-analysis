import torch
import pandas as pd

from torch import nn, optim
from torch.utils.data import DataLoader

from app.data_loading.data_loading import MovieSentimentDataset, MovieSentimentDatasetBuilder
from app.embeddings.sequence_tokenizer import SequenceTokenizer
from app.preprocessing.preprocessor import Preprocessor
from app.hyperparameters.grid_search import GridSearch
from app.regularizer.early_stopping import EarlyStopping
from app.models.lstm_classifier import LSTMClassifier
from app.preprocessing.preprocessing_pipeline import execute_preprocessing_pipeline

default_param_grid = {
    'batch_size': [256, 512],
    'embedding_size': [40, 80, 100],
    'hidden_size': [32, 64, 128],
    'learning_rate': [1e-2],
    'weight_decay': [3e-3, 1e-3]
}

best_param_config = {
    'batch_size': 256,
    'embedding_size': 100,
    'hidden_size': 64,
    'learning_rate': 1e-2,
    'weight_decay': 1e-3
}

class LSTMClassifierTrainer:
    def __init__(self, use_grid_search: bool = False, param_grid: dict = default_param_grid):
        super().__init__()
        self.use_grid_search = use_grid_search
        self.param_grid = param_grid

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.enable_sampling = False
            print("Running on the GPU")
        else:
            self.device = torch.device("cpu")
            self.enable_sampling = True
            print("Running on the CPU")
            print("Notice: As your current runtime is CPU only, the model will only be trained on a subset of the data. Please switch to a GPU runtime.")

        self.dataset_train, self.dataset_validation = None, None
        self.vocab_size, self.padding_size = None, None


    def __proprocess_datasets(self, vocab_checkpoint_path: str):
        print("Starting preprocessing pipeline...")
        if self.dataset_train is None or self.dataset_validation is None:
            raise Exception("No data available for preprocessing.")

        tokenizer = execute_preprocessing_pipeline(self.dataset_train)
        self.vocab_size, self.padding_size = tokenizer.vocab_size, tokenizer.padding_size
        execute_preprocessing_pipeline(self.dataset_validation, tokenizer=tokenizer)
        tokenizer.store_vocab(vocab_checkpoint_path)
        print("Completed preprocessing pipeline.")


    def __train_with_config(self, config: dict, num_epochs: int = 50, patience: int = 5):
        batch_size = config['batch_size']
        embedding_size = config['embedding_size']
        hidden_size = config['hidden_size']
        lr = config['learning_rate']
        weight_decay = config['weight_decay']

        # Create DataLoader
        dataloader_train = DataLoader(self.dataset_train, batch_size=batch_size, shuffle=True, num_workers=1)
        dataloader_validation = DataLoader(self.dataset_validation, batch_size=256, shuffle=False, num_workers=1)

        # Set up a LSTMClassifier for training
        model = LSTMClassifier(vocab_size=self.vocab_size, padding_size=self.padding_size, embedding_size=embedding_size, hidden_size=hidden_size).to(self.device)
        trainer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss = nn.BCELoss()
        early_stopping = EarlyStopping(patience=patience)

        for epoch in range(num_epochs):
            train_loss_epoch, train_acc, n = 0.0, 0, 0
            model.train()
            for i_batch, sample_batched in enumerate(dataloader_train):
                y = sample_batched["sentiment"].type(torch.FloatTensor).to(self.device).reshape(-1, 1)
                y_hat = model(sample_batched["review"].to(self.device).reshape(-1, self.padding_size))
                
                l = loss(y_hat, y)
                train_loss_epoch += l.item()
                trainer.zero_grad()
                l.backward()
                trainer.step()
                train_acc += (y == (y_hat > .5).type(torch.FloatTensor).to(self.device)).sum().item()
                n += len(y)
            
            train_loss_epoch /= n
            train_acc /= n
            
            validation_loss, validation_acc, n = 0, 0, 0
            model.eval()
            with torch.no_grad():
                for i_batch, sample_batched in enumerate(dataloader_validation):
                    y = sample_batched["sentiment"].type(torch.FloatTensor).to(self.device).reshape(-1, 1)
                    y_hat = model(sample_batched["review"].to(self.device).reshape(-1, self.padding_size))
                    l = loss(y_hat, y)
                    validation_loss += l.item()
                    validation_acc += (y == (y_hat > .5).type(torch.FloatTensor).to(self.device)).sum().item()
                    n += len(y)
                
            validation_loss /= n
            validation_acc /= n
            print("Epoch: {}, Train Loss: {}, Train acc: {}, Validation Loss: {}, Validation acc: {}".format(epoch+1, train_loss_epoch, train_acc, validation_loss, validation_acc))
            
            perform_early_stop = early_stopping.track(epoch=epoch, model=model, validation_loss=validation_loss)
            if perform_early_stop:
                print("Stopping early as no improvement was reached for {} epochs".format(early_stopping.patience))
                early_stopping.get_best_version(model)
                validation_loss = early_stopping.best_validation_loss
                break
        
        return validation_loss, model


    def train(self, train_data_path: str = 'data/train.csv', model_checkpoint_path: str = 'model.pt', vocab_checkpoint_path: str = 'vocab.txt', num_epochs: int = 50, patience: int = 5):
        self.dataset_train, self.dataset_validation = MovieSentimentDatasetBuilder\
            .from_csv(csv_file=train_data_path)\
            .with_train_validation_split(splits=[.8, .2])\
            .build()

        # Restrict the number of reviews if running on the CPU
        if self.enable_sampling:
            self.dataset_train.movie_sentiments = self.dataset_train.movie_sentiments.sample(2000)
            self.dataset_validation.movie_sentiments = self.dataset_validation.movie_sentiments.sample(1000)

        self.__proprocess_datasets(vocab_checkpoint_path)

        if self.use_grid_search:
            grid_search = GridSearch(self.param_grid, self.__train_with_config)
            model = grid_search.run(num_epochs, patience)
        else:
            _, model = self.__train_with_config(best_param_config, num_epochs, patience)
        
        torch.save(model, model_checkpoint_path)
