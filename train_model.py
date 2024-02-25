import re
import pickle
import logging
from nltk import download
from nltk.corpus import stopwords, movie_reviews
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.util import everygrams
from nltk.classify.util import accuracy
from nltk.classify import NaiveBayesClassifier
from random import shuffle
from string import punctuation

download('movie_reviews')
download('stopwords')
download('punkt')
download('wordnet')

SPLIT_PCT = 0.80

logging.basicConfig(level=logging.INFO)

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

def prepare_dataset():
    """Prepares the dataset by extracting features from positive and negative movie reviews,
    and organizing them into positive and negative review sets with labels.

    Returns:
        tuple: Two lists containing positive and negative review data respectively.
        Tuple of futures_dict, label 
    """
    reviews_positive = []
    reviews_negative = []
    
    for fileid in movie_reviews.fileids("pos"):
        words = extract_features(movie_reviews.raw(fileid))
        reviews_positive.append((bag_of_words(words), "pos"))
    for fileid in movie_reviews.fileids("neg"):
        words = extract_features(movie_reviews.raw(fileid))
        reviews_negative.append((bag_of_words(words), "neg"))

    return reviews_positive, reviews_negative

def split_set(review_set, SPLIT_PCT):
    """Splits a dataset into training and testing sets based on the specified split 
    percentage.

    Args:
        review_set (list): The dataset to split
        SPLIT_PCT (float): The percentage of the dataset to include in the training set

    Returns:
        tuple: Two lists of training set and testing set.
    """
    split = int(len(review_set) * SPLIT_PCT)
    return (review_set[:split], review_set[split:])

def create_test_train_sets(reviews_positive, reviews_negative, SPLIT_PCT):
    """ Creates training and testing sets by combining positive and 
    negative reviews and then shuffling.

    Args:
        reviews_positive (list): The list of positive reviews.
        reviews_negative (list): The list of negative reviews.
        SPLIT_PCT (float): The percentage of the dataset to include in the training set.

    Returns:
        tuple: Two lists representing the training set and the testing set.
    """
    shuffle(reviews_positive)
    shuffle(reviews_negative)

    positive_train, positive_test = split_set(reviews_positive, SPLIT_PCT)
    negative_train, negative_test = split_set(reviews_negative, SPLIT_PCT)

    train_set = positive_train + negative_train
    test_set = positive_test + negative_test

    return train_set, test_set

def select_features(words, features):
    """Filters a list of words to include only those that are present in a specified list of features.
    
    Args:
        words (list): The list of words to filter.
        features (list): The list of features to filter by.

    Returns:
        list: A list of words that are in the specified features.
    """
    return [word for word in words if word in features]

def retrain_dataset(reviews_positive, reviews_negative, features):
    """Retrains the dataset by filtering the reviews to include only the 
    most informative features.

    Args:
        reviews_positive (list): The list of positive reviews.
        reviews_negative (list): The list of negative reviews.
        features (list): The list of most informative features.

    Returns:
        tuple: Two lists containing filtered positive and negative review data respectively.
    """
    reviews_positive_filtered = []
    reviews_negative_filtered = []
    for (bag, label) in reviews_positive:
        words = select_features(bag.keys(), features)
        reviews_positive_filtered.append((bag_of_words(words), "pos"))
    for (bag, label) in reviews_negative:
        words = select_features(bag.keys(), features)
        reviews_negative_filtered.append((bag_of_words(words), "neg"))

    return reviews_positive_filtered, reviews_negative_filtered

def save_model(model):
    model_file = open("sa_classifier.pickle", "wb")
    pickle.dump(model, model_file)
    model_file.close()
    logging.info(f"Model saved to sa_classifier.pickle")


if __name__ == '__main__':
    print(f"Testing extract features: \n {extract_features("Hello world, corpuses calling!")}")

    logging.info("Preparing dataset...")
    reviews_positive, reviews_negative = prepare_dataset()

    logging.info("Getting test and train sets...")
    train_set, test_set = create_test_train_sets(reviews_positive, reviews_negative, SPLIT_PCT)

    model = NaiveBayesClassifier.train(train_set)

    accuracy_score = accuracy(model, test_set)
    logging.info(f"The accuracy score is {100 * accuracy_score}%")
    # model.show_most_informative_features(20)

    # train again using the most informative features
    features = [w for (w, v) in model.most_informative_features(1000)]

    reviews_positive_filtered, reviews_negative_filtered = retrain_dataset(reviews_positive,
                                                                           reviews_negative,
                                                                           features)
    
    train_set_filtered, test_set_filtered = create_test_train_sets(reviews_positive_filtered,
                                                                   reviews_negative_filtered,
                                                                   SPLIT_PCT)
    
    model_filtered = NaiveBayesClassifier.train(train_set_filtered)
    accuracy_score_filtered = accuracy(model_filtered, test_set_filtered)
    logging.info(f"They accuracy score using the most informative features: {100 * accuracy_score_filtered}%")

    logging.info(f"Saving model")
    save_model(model_filtered)
