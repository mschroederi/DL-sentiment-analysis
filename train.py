import torch

from app.trainers.lstm_classifier_trainer import LSTMClassifierTrainer

if __name__ == '__main__':
    trainer = LSTMClassifierTrainer(use_grid_search=False)
    trainer.train(train_data_path='data/train.csv', num_epochs=2, patience=5)
