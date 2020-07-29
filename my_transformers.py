import pandas as pd
from transformers import pipeline

# x = 'We are very happy to include pipeline into the transformers repository.'
# y = nlp([x, x])
# print(y)

test = pd.read_csv("data/test.csv").sample(1000)

def shorten(words):
    if len(words) > 256:
        words = words[:256]
    return " ".join(words)
test["review"] = test["review"].str.split().apply(shorten)

reviews = list(test["review"])

nlp = pipeline("sentiment-analysis")

# sentiments = nlp(reviews)
print(len(reviews))
sentiments = []
failed_reviews = []
for i, review in enumerate(reviews):
    if i % 100 == 0:
        print("Review:", i)
    try:
        sentiment = nlp(review)[0]
        sentiments.append(sentiment)
    except Exception as e:
        print(len(review.split()))
        sentiments.append({"label": "FAILED"})
        failed_reviews.append(review)
    
print(len(failed_reviews))
# t = [len(f.split()) for f in reviews]
# t2 = [len(f.split()) for f in failed_reviews]
# print(t)
# print(len(t2))
def translate_sentiment(sentiment):
    if sentiment["label"] == "POSITIVE":
        return 1
    elif sentiment["label"] == "NEGATIVE":
        return 0
    return -1

# print(sentiments)
y_hat = list(map(translate_sentiment, sentiments))

test["prediction"] = y_hat

test["hit"] = test["sentiment"] == test["prediction"]
print(test.head())

print(sum(test["hit"]) / len(test))
