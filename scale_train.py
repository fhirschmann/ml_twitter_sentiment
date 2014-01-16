#!/usr/bin/env python
from __future__ import print_function
import csv
import sys
from itertools import islice

from sklearn.metrics import classification_report
import numpy as np

from vect import vectorizer
from cls import cls1, cls2
from pp import PreProcessor, MAPPING


TESTING = "--test" in sys.argv
HOLDOUT = 0.05
BATCH_SIZE = 5000


def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))


if __name__ == "__main__":
    with open('split_result.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(['size', 'classifier', 'pp', 'precision', 'recall', 'f1_score'])

        # 2^22 = 4194304 < 5292305 (# of rows in tweets.big.db)
        for exp in (xrange(13, 14) if TESTING else xrange(16, 23)):
            size = 2 ** exp

            # Minimal or full preprocessing
            for full_pp in [True, False]:

                # Train and evaluate with two classifiers
                for cls in [cls1, cls2]:
                    print("Now training a %s with at most %s instances and %s pp" % (
                        cls.__class__.__name__, size, "full" if full_pp else "minimal"))

                    pp = PreProcessor("tweets.small.db" if TESTING else "tweets.big.db", full_pp)

                    # Take at most `size` elements
                    it = islice(pp.tweets(), size)

                    # Hold out HOLDOUT * size instances that will form the test
                    # set. This will not shuffle the data and may lead to
                    # a bad test set if the first HOLDOUT * size instances belong
                    # to the same class. However, due to the nature of our
                    # architecture (generators/lazy evaluation) this is the
                    # only way. For this reason, we just shuffled the sqlite
                    # database beforehand.
                    test_tweets, test_outcomes = zip(*take(int(HOLDOUT * size), it))
                    y_test = np.array(test_outcomes)
                    X_test = vectorizer.fit_transform(test_tweets)
                    actual_size = 0

                    while True:
                        batch = take(BATCH_SIZE, it)
                        if not batch:
                            # no more items
                            break

                        actual_size += len(batch)
                        print("\rConsumed: {0} of {1}".format(actual_size, int(size * (1 - HOLDOUT)) + 1), end="")
                        tweets, outcomes = zip(*batch)

                        y = np.array(outcomes)
                        X = vectorizer.fit_transform(tweets)

                        cls.partial_fit(X, y, classes=MAPPING.values())
                    print()

                    y_predicted = cls.predict(X_test)
                    target_names = ['class 0', 'class 1', 'class 2']

                    result = classification_report(y_test, y_predicted, target_names=target_names).split()
                    # Take only the averaged scores over the three classes
                    precision, recall, f1_score = result[-4], result[-3], result[-2]
                    print("F1: ", f1_score)
                    writer.writerow([actual_size, cls.__class__.__name__, full_pp, precision, recall, f1_score])
