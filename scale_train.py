#!/usr/bin/env python
from __future__ import print_function, division
import csv
import sys
from itertools import islice

from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
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
BATCH_SIZE = 5000


def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))


def evaluate(cls, X, y):
    y_predicted = cls.predict(X)
    target_names = ['class 0', 'class 1', 'class 2']

    result = classification_report(y, y_predicted, target_names=target_names).split()
    return np.array((result[-4], result[-3], result[-2]))


if __name__ == "__main__":
    with open('split_result.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(['size', 'classifier', 'pp', 'precision', 'recall', 'f1_score'])

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

                n_instances = 0
                prev_results = None

                while True:
                    # Take the next batch
                    batch = take(BATCH_SIZE, it)

                    if not batch:
                        # no more items
                        break

                    tweets, outcomes = zip(*batch)
                    y = np.array(outcomes)
                    X = vectorizer.fit_transform(tweets)

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=HOLDOUT, random_state=0)

                    print("\rConsumed: {0} instances".format(n_instances), end="")

                    # Partially fit the instances
                    cls.partial_fit(X, y, classes=MAPPING.values())

                    results = evaluate(cls, X_test, y_test)
                    if prev_results:
                        # This multiplies the previous results with the current
                        # number of instances, adds it to the current results
                        # multiplied by the additional number of instances and
                        # normalizes it by the total instances processed so
                        # far.
                        results = (prev_results * n_instances + results * len(batch)) / (n_instances + len(batch))

                    n_instances += len(batch)

                    # Write intermediate results to file
                    writer.writerow([n_instances, cls.__class__.__name__, full_pp, results[0], results[1], results[2]])
                    f.flush()
                print()
