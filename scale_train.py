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
SIZE = [s for s in sys.argv if s.isdigit()]
if SIZE:
    SIZE = int(SIZE[0])
else:
    SIZE = -1
HOLDOUT = 0.05
BATCH_SIZE = 100000


def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))


def evaluate(cls, test_set):
    tweets, outcomes = zip(*test_set)
    y = np.array(outcomes)
    X = vectorizer.fit_transform(tweets)

    y_predicted = cls.predict(X)
    target_names = ['class 0', 'class 1', 'class 2']

    result = classification_report(y, y_predicted, target_names=target_names).split()
    return (result[-4], result[-3], result[-2])


if __name__ == "__main__":
    with open('split_result.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(['size', 'tsize', 'classifier', 'pp', 'precision', 'recall', 'f1_score'])

        # Minimal or full preprocessing
        for full_pp in [True, False]:

            # Train and evaluate with two classifiers
            for cls in [cls1, cls2]:
                print("Now training a %s with at most %s instances and %s pp" % (
                    cls.__class__.__name__, SIZE, "full" if full_pp else "minimal"))

                pp = PreProcessor("tweets.small.db" if TESTING else "tweets.big.db", full_pp)

                # Take at most `size` elements
                if SIZE > 0:
                    it = islice(pp.tweets(), SIZE)
                else:
                    it = pp.tweets()

                # Hold out HOLDOUT * SIZE instances that will form the test
                # set. This will not shuffle the data and may lead to
                # a bad test set if the first HOLDOUT * SIZE instances belong
                # to the same class. However, due to the nature of our
                # architecture (generators/lazy evaluation) this is the
                # only way. For this reason, we just shuffled the sqlite
                # database beforehand.
                holdouts = []
                actual_size = 0

                while True:
                    # Take the next batch
                    batch = take(int((1 - HOLDOUT) * BATCH_SIZE), it)

                    # Append HOLDOUT instances to the training set
                    holdout = take(int(BATCH_SIZE * HOLDOUT), it)

                    if not batch or not holdout:
                        # no more items
                        break

                    holdouts.extend(holdout)
                    actual_size += len(batch)

                    print("\rConsumed: {0} instances".format(actual_size), end="")
                    tweets, outcomes = zip(*batch)

                    y = np.array(outcomes)
                    X = vectorizer.fit_transform(tweets)

                    # Partially fit the instances
                    cls.partial_fit(X, y, classes=MAPPING.values())

                    precision, recall, f1 = evaluate(cls, holdouts)

                    # Write intermediate results to file
                    writer.writerow([actual_size, len(holdouts), cls.__class__.__name__, full_pp, precision, recall, f1])
                    f.flush()
                print()
                print("F1:", f1)
