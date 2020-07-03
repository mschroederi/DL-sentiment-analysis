import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from data_loader import MovieSentimentDataset
from embeddings.bag_of_words import BagOfWords
from models.lstm_classifier import LSTMClassifier
from sklearn.feature_extraction.text import CountVectorizer
from embeddings.one_hot_encoder import OneHotSequenceEncoder


def get_word_to_idx(reviews):
    # https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
    word_to_idx = {}
    for sent, tags in reviews:
        for word in sent:
            if word not in word_to_idx:
                word_to_ix[word] = len(word_to_idx)
    return word_to_idx

if __name__ == '__main__':
    dataset = MovieSentimentDataset(csv_file='data/train.csv')
    # Restrict the number of reviews for now due to long run time
    dataset.movie_sentiments = dataset.movie_sentiments.sample(100)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=1)

    # Preprocess reviews
    reviews = dataset.movie_sentiments["review"]
    reviews = BagOfWords.preprocess(reviews)

    encoder = OneHotSequenceEncoder(max_seq_len=500)
    dataset.movie_sentiments["review"] = encoder.fit_transform(reviews)
    
    dataset.movie_sentiments = dataset.movie_sentiments[dataset.movie_sentiments["review"] != "empty"]
    
    print(dataset.movie_sentiments["review"].head())

    print("padding_size: ", encoder.padding_size)

    # Set up a bag of words model and training
    embedding_size = 1024
    model = LSTMClassifier(vocab_size=encoder.vocab_size, padding_size=encoder.padding_size, embedding_size=embedding_size, hidden_size=128)
    loss = nn.BCELoss()
    num_epochs = 10
    lr = 0.5
    trainer = optim.SGD(model.parameters(), lr)

    for epoch in range(num_epochs):
        train_loss_epoch, n = 0.0, 0
        model.train()
        for i_batch, sample_batched in enumerate(dataloader):
            y = sample_batched["sentiment"].type(torch.FloatTensor)
            y = y.reshape(-1, 1)
            y_hat = model(sample_batched["review"].reshape(-1, encoder.padding_size))
            
            l = loss(y_hat, y)
            train_loss_epoch += l.item()
            trainer.zero_grad()
            l.backward()
            trainer.step()
            n += len(sample_batched)

        train_loss_epoch /= n
        print("Epoch: {}, Train Loss: {}".format(epoch+1, train_loss_epoch))

    # # Find the words that are responsible for sentiment
    # param = next(model.parameters())
    # transform = embedding.embedding.inverse_transform
    # _, max_indices = torch.topk(param, 10, largest=True)
    # _, min_indices = torch.topk(param, 10, largest=False)
    # print("Words that speak for a good review:")
    # print(transform(max_indices[0].tolist()))
    # print("Words that speak for a bad review:")
    # print(transform(min_indices[0].tolist()))
