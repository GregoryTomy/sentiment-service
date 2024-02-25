import json
import pickle
import re
from string import punctuation

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.util import everygrams

# download datasets for the model
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

MODEL_PATH = "sa_classifier.pickle"
with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

stopwords_english = stopwords.words("english")
lemmatizer = WordNetLemmatizer()

def is_useful_word(word):
    """Determine if a word is useful for analysis by checking if it is not a stopword 
    or punctuation.

    Args:
        word (str): The word to evaluate

    Returns:
        bool: True if the words is useful, False otherwise.
    """
    return (word not in stopwords_english) and (word not in punctuation)

def extract_features(document):
    """Extracts and preprocesses features from a given document. The process includes 
    tokenization, lemmatization, removing punctuation and stopwords, and creating 
    n-grams up to 3 words.

    Args:
        document (str): The text document from which to extract features.

    Returns:
        list: A list of features (n-grams ans strings) extracted from the document. 
    """
    words = word_tokenize(document)
    lemmas = [str(lemmatizer.lemmatize(w)) for w in words if is_useful_word(w)]
    document_2 = " ".join(lemmas)
    document_2 = document_2.lower()
    document_2 = re.sub(r'[^a-zA-Z0-9\s]', " ", document_2)

    words = [w for w in document_2.split(" ") if w != "" and is_useful_word(w)]

    return [str('_'.join(ngram)) for ngram in list(everygrams(words, max_len=3))]

def bag_of_words(words):
    """Creates a bag of words model from a list of words by counting the occurrences
      of each word.

    Args:
        words (list): The list of words to transform into a bag of words

    Returns:
        bag (dict): A dictionary where keys are words and values are the counts
          of their occurences. 
    """
    bag = {}
    for word in words:
        bag[word] = bag.get(word, 0) + 1
    return bag


def get_sentiment(review):
    """Classify the sentiment of a review

    Args:
        review (str): Review string to be classified.

    Returns:
        str: 'pos' for positive, 'neg' for negative
    """
    words = extract_features(review)
    words = bag_of_words(words)
    return model.classify(words)
