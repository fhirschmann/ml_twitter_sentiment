#!/usr/bin/env python
import csv
import sys

from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
import numpy as np

from TweetProcessor import TweetProcessor
from vect import vectorizer


TESTING = "--test" in sys.argv


if __name__ == "__main__":
    csv_table = [['size', 'training_size', 'classifier', 'pp', 'precision', 'recall', 'f1_score']]

    # 2^22 = 4194304 < 5292305 (# of rows in tweets.big.db)
    for exp in (xrange(15, 16) if TESTING else xrange(10, 22)):
        size = 2 ** exp

        # Minimal or full preprocessing
        for pp in [True, False]:

            # Train and evaluate with two classifiers
            for cls in [RidgeClassifier(), SGDClassifier()]:
                print("Now training a %s with %s instances (Train-test-split of 5 to 95) and %s pp" % (
                      cls.__class__.__name__, size, "full" if pp else "minimal"))

                processor = TweetProcessor("tweets.small.db" if TESTING else "tweets.big.db")
                corpus = processor.get_corpus(pp, size)
                tweets, outcomes = zip(*corpus)

                tweets = [" ".join(sen) for sen in tweets]
                y = np.array(outcomes)

                X = vectorizer.fit_transform(tweets)

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.95, random_state=0)
                cls.fit(X_train, y_train)
                y_predicted = cls.predict(X_test)
                target_names = ['class 0', 'class 1', 'class 2']

                result = classification_report(y_test, y_predicted, target_names=target_names).split()
                # Take only the averaged scores over the three classes
                precision, recall, f1_score = result[-4], result[-3], result[-2]
                print f1_score
                csv_table += [[size, len(y_train), cls.__class__.__name__, pp, precision, recall, f1_score]]

    with open('split_result.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(csv_table)
