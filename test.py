import torch

from app.testers.lstm_classifier_tester import LSTMClassifierTester

if __name__ == '__main__':
    tester = LSTMClassifierTester()
    tester.test(test_data_path='data/test.csv', model_checkpoint_path='best_model.pt', vocab_path='vocab.txt')
