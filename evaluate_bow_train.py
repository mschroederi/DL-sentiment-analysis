import copy
import torch
import numpy as np
import pandas as pd

from torch import nn
from torch.utils.data import DataLoader
from typing import Dict, List

from bow_train import BowMovieSentimentDataset
from app.embeddings.bag_of_words import BagOfWords

def evaluation(model: nn.Module, embedding) -> None:
    # Find the words that are responsible for sentiment
    param = next(model.parameters())
    transform = embedding.embedding.inverse_transform
    
    def words_to_prob(k: int, largest: bool, absolute: bool = False) -> List[str]:
        my_param = param.abs() if absolute else param
        values, indices = torch.topk(my_param, k, largest=largest)
        values = values.squeeze(dim=0).tolist()
        words = transform(indices[0].tolist())
        word_to_prob = ["{} ({})".format(word, np.round(prob, 8)) for (word, prob) in zip(words, values)]
        return word_to_prob
    
    k = 10
    pos_sentiment = words_to_prob(k, True)
    neg_sentiment = words_to_prob(k, False)
    no_sentiment = words_to_prob(k, False, True)
    print("Words that speak for a good review:")
    print(", ".join(pos_sentiment))
    print("Words that speak for a bad review:")
    print(", ".join(neg_sentiment))
    print("Words that have no sentiment:")
    print(", ".join(no_sentiment))

def further_evaluation(model: nn.Module, embedding) -> None:
    param = next(model.parameters())
    transform = embedding.embedding.inverse_transform
    # print(param)
    num_no_contribution = torch.sum(param.abs() <= 0.5)
    print("No contribution:", num_no_contribution)


def test(model: nn.Module, dataloader: DataLoader, embedding, loss_function) -> Dict[str, float]:
    test_loss, matches, n = 0.0, 0, 0
    model.eval()
    with torch.no_grad():
        for _, sample_batched in enumerate(dataloader):
            y = sample_batched["sentiment"].type(torch.FloatTensor)
            y = torch.unsqueeze(y, 1)
            bow = sample_batched["review"].type(torch.FloatTensor)
            y_hat = model(bow)
            
            prediction = y_hat >= 0
            matches += (prediction == y).sum().item()
            l = loss_function(y_hat, y)
            test_loss += l.item()
            n += len(sample_batched["sentiment"])
            
    accuracy = matches / n
    test_loss /= n
    
    return {
        "loss": test_loss,
        "accuracy": accuracy
    }

def shrink_model(model: nn.Module, embedding, k) -> nn.Module:
    param = next(model.parameters())

    max_values, max_indices = torch.topk(param, k, largest=True)
    min_values, min_indices = torch.topk(param, k, largest=False)

    shrinked_model = copy.deepcopy(model)
    weights = shrinked_model.linear.weight.data
    new_weights = torch.zeros_like(weights)
    new_weights[0][max_indices] = max_values
    new_weights[0][min_indices] = min_values
    shrinked_model.linear.weight.data = new_weights
    return shrinked_model

def shrink_model2(model: nn.Module, embedding, k) -> nn.Module:
    shrinked_model = copy.deepcopy(model)
    param = next(shrinked_model.parameters())
    _, indices = torch.topk(param.abs(), k, largest=True)
    values = param[0][indices]

    weights = shrinked_model.linear.weight.data
    new_weights = torch.zeros_like(weights)
    new_weights[0][indices] = values
    shrinked_model.linear.weight.data = new_weights
    return shrinked_model


if __name__ == "__main__":
    embedding = BagOfWords.from_vocab_file("data/bow_vocab2.txt")
    vocab_size = len(embedding.vocab)
    print(vocab_size)

    test_df = pd.read_csv("data/bow_test2.csv")
    test_dataset = BowMovieSentimentDataset(test_df, embedding=embedding, binary_vectorizer=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=True, num_workers=4)
    print("Created Test Batches")

    loss = nn.BCEWithLogitsLoss()
    checkpoint_loc = "checkpoints/bow_model"
    model = torch.load(checkpoint_loc)
    model.eval()
    # further_evaluation(model, embedding)

    results = test(model, test_loader, embedding, loss)
    print(results)
    i = 1
    accuracies = [(0, 0.5)]

    while i * 2 < vocab_size:
        shrinked_model = shrink_model(model, embedding, i)
        shrinked_results = test(shrinked_model, test_loader, embedding, loss)
        print(i * 2, shrinked_results)
        accuracies.append((i * 2, shrinked_results["accuracy"]))
        i *= 2

    # while i < vocab_size:
    #     shrinked_model = shrink_model2(model, embedding, i)
    #     shrinked_results = test(shrinked_model, test_loader, embedding, loss)
    #     print(i, shrinked_results)
    #     accuracies.append((i, shrinked_results["accuracy"]))
    #     i *= 2

    accuracies.append((vocab_size, results["accuracy"]))
    print(accuracies)
