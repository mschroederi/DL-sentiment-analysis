import argparse

from app.testers.lstm_classifier_tester import LSTMClassifierTester


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-data-path', type=str, default="data/test.csv", dest="test_data_path",
                        help='File location where the processed test data is stored.')
    parser.add_argument('--model-checkpoint', type=str, default="model.pt", dest="model_checkpoint",
                        help='File location where the model from training is stored.')
    parser.add_argument('--vocab-checkpoint', type=str, default="data/vocab.txt", dest="vocab_checkpoint",
                        help='File location where the vocabulary of the sequence tokenizer from training is stored.')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    tester = LSTMClassifierTester()
    tester.test(test_data_path=args.test_data_path,
                model_checkpoint_path=args.model_checkpoint,
                vocab_path=args.vocab_checkpoint)
