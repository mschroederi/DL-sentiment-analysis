import torch
from torch import nn
from torch.utils.data import DataLoader

from app.embeddings.sequence_tokenizer import SequenceTokenizer
from architecture import AttentionRNNClassifier
from data_loading import MovieSentimentDatasetBuilder
from train import execute_preprocessing_pipeline

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
        enable_sampling = False
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        enable_sampling = True
        print("Running on the CPU")

    checkpoint_path = "best_model.pt"
    model: AttentionRNNClassifier = torch.load(checkpoint_path)
    padding_size = model.padding_size
    loss = nn.BCELoss()

    dataset_test = MovieSentimentDatasetBuilder \
        .from_csv(csv_file='data/test.csv') \
        .build()
    tokenizer = SequenceTokenizer.from_vocab_file("sequence_vocab.txt", padding_size)
    execute_preprocessing_pipeline(dataset_test, tokenizer=tokenizer)
    dataloader_test = DataLoader(dataset_test, batch_size=256, shuffle=False, num_workers=1)

    test_loss, test_acc, n = 0, 0, 0
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader_test):
            y = sample_batched["sentiment"].type(torch.FloatTensor).to(device).reshape(-1, 1)
            y_hat = model(sample_batched["review"].to(device).reshape(-1, padding_size))
            l = loss(y_hat, y)
            test_loss += l.item()
            test_acc += (y == (y_hat > .5).type(torch.FloatTensor).to(device)).sum().item()
            n += len(y)

    test_loss /= n
    test_acc /= n
    print("Test Loss: {}, Test acc: {}".format(test_loss, test_acc))