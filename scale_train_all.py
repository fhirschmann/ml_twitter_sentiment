#!/usr/bin/env python
from __future__ import print_function, division
import csv
import sys
from itertools import islice, chain, repeat, tee

from sklearn.metrics import classification_report
import numpy as np

from vect import vectorizer
from cls import cls1, cls2
from pp import PreProcessor, MAPPING


HOLDOUT = 0.05

if "--test" in sys.argv:
    TESTING = True
    SIZE = 10000
else:
    TESTING = False
    SIZE = 100000  # how many rows are there in the all db?


def take(n, iterable):
    """
    Return first n items of the iterable as a list
    """
    return list(islice(iterable, n))


def evaluate(cls, X, y):
    """
    Returns the precision, recall, and f-score produced when predicting
    the given test samples.
    """
    y_predicted = cls.predict(X)
    target_names = ['class 0', 'class 1', 'class 2']

    result = classification_report(y, y_predicted, target_names=target_names).split()
    return (result[-4], result[-3], result[-2])


def transform(batch):
    """
    Transforms a batch in order to be fed to sklearn.
    """
    tweets, outcomes = zip(*batch)
    y = np.array(outcomes)
    X = vectorizer.fit_transform(tweets)

    return (X, y)


if __name__ == "__main__":
    with open('split_result.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(['size', 'classifier', 'pp', 'precision', 'recall', 'f1_score'])

        for e in xrange(11, 23):
            size = 2 ** e
            if size > SIZE:
                break

            # Minimal or full preprocessing
            for full_pp in [True, False]:

                # Train and evaluate with two classifiers
                for cls in [cls1, cls2]:

                    print("Now training a %s with ~%s training instances and %s pp" % (
                        cls.__class__.__name__, int(SIZE * (1 - HOLDOUT)), "full" if full_pp else "minimal"))

                    pp = PreProcessor("tweets.small.db" if TESTING else "tweets.big.db", full_pp)
                    tweets = pp.tweets()

                    # Create the test set. Do note that we did shuffle the database
                    # before and retweets are removed by the PreProcessor.
                    test_tweets = take(int(HOLDOUT * size), tweets)
                    X_test, y_test = transform(test_tweets)

                    # Limit the train set
                    limited_tweets = islice(tweets, int(size * (1 - HOLDOUT)))

                    X, y = transform(limited_tweets)

                    # Partially fit the instances
                    cls.fit(X, y)
                    results = evaluate(cls, X_test, y_test)

                    # Write intermediate results to file
                    writer.writerow([int(size * (1 - HOLDOUT)), cls.__class__.__name__, full_pp, results[0], results[1], results[2]])
                    f.flush()
