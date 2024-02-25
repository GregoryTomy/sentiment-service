import pickle
import sys
from nltk import download
from train_model import extract_features, bag_of_words


download("punkt")
download("stopwords")

print(f"Testing extract features: {
    extract_features("Hellow world, corpuses calling")
}")

with open("sa_classifier.pickle", "rb") as model_file:
    model = pickle.load(model_file)

def get_sentiment(review):
    words = extract_features(review)
    words = bag_of_words(words)
    return model.classify(words)


if __name__ == "__main__":
    positive_review = "This movie is the best, with witty dialog and beautiful shots."
    negative_review = "I hated everything about this unimaginative mess. Two thumbs down!"

    print(f"Positive review prediction: {get_sentiment(positive_review)}")
    print(f"Negative review prediction: {get_sentiment(negative_review)}")