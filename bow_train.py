import argparse

from app.trainers.bow_classifier_trainer import BowClassifierTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data-path', type=str, default="data/train.csv", dest="train_data_path",
                        help='File location where the processed training data is stored.')
    parser.add_argument('--model-checkpoint', type=str, default="checkpoints/bow_model.pt", dest="model_checkpoint",
                        help='File location where the trained model will be stored.')
    parser.add_argument('--vocab-checkpoint', type=str, default="checkpoints/bow_vocab.txt", dest="vocab_checkpoint",
                        help='File location where the vocabulary of the tokenizer will be stored.')
    parser.add_argument('--num-epochs', type=int, default=3, dest="num_epochs",
                        help='The number of epochs for training.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    trainer = BowClassifierTrainer()
    trainer.train(train_data_path=args.train_data_path,
                  model_checkpoint_path=args.model_checkpoint,
                  vocab_checkpoint_path=args.vocab_checkpoint,
                  num_epochs=args.num_epochs)
