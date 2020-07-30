import torch

from app.trainers.lstm_classifier_trainer import LSTMClassifierTrainer

if __name__ == '__main__':
    trainer = LSTMClassifierTrainer(use_grid_search=False)
    trainer.train(train_data_path='data/train.csv', test_data_path='data/test.csv', num_epochs=50, patience=5)








# from torch import nn, optim
# from torch.utils.data import DataLoader

# from data_loading import MovieSentimentDataset, MovieSentimentDatasetBuilder
# from architecture import AttentionRNNClassifier
# from app.embeddings.sequence_tokenizer import SequenceTokenizer
# from app.preprocessing.preprocessor import Preprocessor
# from app.regularizer.early_stopping import EarlyStopping


# # Preprocess reviews
# def execute_preprocessing_pipeline(dataset: MovieSentimentDataset, tokenizer=None):
#     reviews = dataset.movie_sentiments["review"]
#     dataset.movie_sentiments["review"] = Preprocessor.remove_symbols(dataset.movie_sentiments["review"])
#     dataset.movie_sentiments = Preprocessor.remove_long_sequences(dataset.movie_sentiments, max_len=1000)

#     if tokenizer is None:
#         tokenizer = SequenceTokenizer()
#         tokenizer.fit(dataset.movie_sentiments["review"])
#     dataset.movie_sentiments["review"] = tokenizer.transform(dataset.movie_sentiments["review"])
#     return tokenizer


# if __name__ == '__main__':
#     if torch.cuda.is_available():
#         device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
#         enable_sampling = False
#         print("Running on the GPU")
#     else:
#         device = torch.device("cpu")
#         enable_sampling = True
#         print("Running on the CPU")

#     dataset_train, dataset_validation = MovieSentimentDatasetBuilder\
#         .from_csv(csv_file='data/train.csv')\
#         .with_train_validation_split(splits=[.8, .2])\
#         .build()
    
#     # Restrict the number of reviews if running on the CPU
#     if enable_sampling:
#         dataset_train.movie_sentiments = dataset_train.movie_sentiments.sample(100)
#         dataset_validation.movie_sentiments = dataset_validation.movie_sentiments.sample(20)
    
#     tokenizer = execute_preprocessing_pipeline(dataset_train)
#     vocab_size, padding_size = tokenizer.vocab_size, tokenizer.padding_size
#     execute_preprocessing_pipeline(dataset_validation, tokenizer=tokenizer)

#     # Create DataLoader
#     dataloader_train = DataLoader(dataset_train, batch_size=128, shuffle=True, num_workers=1)
#     dataloader_validation = DataLoader(dataset_validation, batch_size=256, shuffle=False, num_workers=1)

#     # Set up a bag of words model and training
#     #model = LSTMClassifier(vocab_size=vocab_size, padding_size=padding_size, embedding_size=200, hidden_size=32).to(device)
#     model = AttentionRNNClassifier(vocab_size=vocab_size, padding_size=padding_size, embedding_size=50, hidden_size=20, attn_encoder_hidden_size=20).to(device)
#     loss = nn.BCELoss()
#     num_epochs = 50
#     lr = 1e-2
#     trainer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
#     early_stopping = EarlyStopping(patience=5)

#     for epoch in range(num_epochs):
#         train_loss_epoch, train_acc, n = 0.0, 0, 0
#         model.train()
#         for i_batch, sample_batched in enumerate(dataloader_train):
#             y = sample_batched["sentiment"].type(torch.FloatTensor).to(device).reshape(-1, 1)
#             y_hat = model(sample_batched["review"].to(device).reshape(-1, padding_size))
            
#             l = loss(y_hat, y)
#             train_loss_epoch += l.item()
#             trainer.zero_grad()
#             l.backward()
#             trainer.step()
#             train_acc += (y == (y_hat > .5).type(torch.FloatTensor).to(device)).sum().item()
#             n += len(y)
        
#         train_loss_epoch /= n
#         train_acc /= n
        
#         validation_loss, validation_acc, n = 0, 0, 0
#         with torch.no_grad():
#             for i_batch, sample_batched in enumerate(dataloader_validation):
#                 y = sample_batched["sentiment"].type(torch.FloatTensor).to(device).reshape(-1, 1)
#                 y_hat = model(sample_batched["review"].to(device).reshape(-1, padding_size))
#                 l = loss(y_hat, y)
#                 validation_loss += l.item()
#                 validation_acc += (y == (y_hat > .5).type(torch.FloatTensor).to(device)).sum().item()
#                 n += len(y)
            
#         validation_loss /= n
#         validation_acc /= n
#         print("Epoch: {}, Train Loss: {}, Train acc: {}, Validation Loss: {}, Validation acc: {}".format(epoch+1, train_loss_epoch, train_acc, validation_loss, validation_acc))
        
#         perform_early_stop = early_stopping.track(epoch=epoch, model=model, validation_loss=validation_loss)
#         if perform_early_stop:
#             print("Stopping early as no improvement was reached for {} epochs".format(early_stopping.patience))
#             model = early_stopping.get_best_version(model)
#             break
#     # Store the vocabulary used for training
#     tokenizer.store_vocab("sequence_vocab.txt")
