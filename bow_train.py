import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from data_loader import MovieSentimentDataset, MovieSentimentDatasetBuilder
from embeddings.bag_of_words import BagOfWords
from models.bow_classifier import BowClassifier


if __name__ == '__main__':
    dataset_train, dataset_validation = MovieSentimentDatasetBuilder\
        .from_csv(csv_file='data/train.csv')\
        .with_train_validation_split(splits=[.8, .2])

    dataset = dataset_train
    # Restrict the number of reviews for now due to long run time
    dataset.movie_sentiments = dataset.movie_sentiments.sample(100)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=1)

    # Preprocess reviews and create a bag of words embedding
    reviews = dataset.movie_sentiments["review"]
    reviews = BagOfWords.preprocess(reviews)
    dataset.movie_sentiments["review"] = reviews
    embedding = BagOfWords.from_pandas(reviews)
    vocab_size = len(embedding.vocab)
    print("Vocab Size: {}".format(vocab_size))

    # Set up a bag of words model and training
    model = BowClassifier(vocab_size)
    loss = nn.BCELoss()
    num_epochs = 3
    lr = 0.1
    trainer = optim.SGD(model.parameters(), lr)

    for epoch in range(num_epochs):
        train_loss_epoch, n = 0.0, 0
        model.train()
        for i_batch, sample_batched in enumerate(dataloader):
            y = sample_batched["sentiment"].type(torch.FloatTensor)
            y = torch.unsqueeze(y, 1)
            bow = embedding.embed(sample_batched["review"])
            y_hat = model(bow)
            
            l = loss(y_hat, y)
            train_loss_epoch += l.item()
            trainer.zero_grad()
            l.backward()
            trainer.step()
            n += len(y)

        train_loss_epoch /= n
        print("Epoch: {}, Train Loss: {}".format(epoch+1, train_loss_epoch))

    # Find the words that are responsible for sentiment
    param = next(model.parameters())
    transform = embedding.embedding.inverse_transform
    _, max_indices = torch.topk(param, 10, largest=True)
    _, min_indices = torch.topk(param, 10, largest=False)
    print("Words that speak for a good review:")
    print(transform(max_indices[0].tolist()))
    print("Words that speak for a bad review:")
    print(transform(min_indices[0].tolist()))
