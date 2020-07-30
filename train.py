import argparse

from app.trainers.lstm_classifier_trainer import LSTMClassifierTrainer


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data-path', type=str, default="data/train.csv", dest="train_data_path",
                        help='File location where the processed training data is stored.')
    parser.add_argument('--model-checkpoint', type=str, default="model.pt", dest="model_checkpoint",
                        help='File location where the trained model will be stored.')
    parser.add_argument('--vocab-checkpoint', type=str, default="data/vocab.txt", dest="vocab_checkpoint",
                        help='File location where the vocabulary of the sequence tokenizer will be stored.')
    parser.add_argument('--grid-search', type=str2bool, default=False, dest="grid_search",
                        help='If `True` performs grid-search to find the best model configuration.')
    parser.add_argument('--num-epochs', type=int, default=50, dest="num_epochs",
                        help='The number of epochs for training.')
    parser.add_argument('--patience', type=int, default=5,
                        help='The number of epochs early-stopping is waiting for significant changes.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    trainer = LSTMClassifierTrainer(use_grid_search=args.grid_search)
    trainer.train(train_data_path=args.train_data_path,
                  model_checkpoint_path=args.model_checkpoint,
                  vocab_checkpoint_path=args.vocab_checkpoint,
                  num_epochs=args.num_epochs,
                  patience=args.patience)
