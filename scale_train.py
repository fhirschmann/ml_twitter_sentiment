#!/usr/bin/env python
from __future__ import print_function, division
import csv
import sys
import cPickle as pickle
from itertools import islice, chain, repeat, tee

from sklearn.metrics import classification_report
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import HashingVectorizer
from nltk.corpus import stopwords
import numpy as np

from vect import LemmaTokenizer
from cls import cls1, cls2
from pp import PreProcessor, MAPPING


HOLDOUT = 0.05

if "--test" in sys.argv:
    TESTING = True
    BATCH_SIZE = 2000
    SIZE = 10000
else:
    TESTING = False
    BATCH_SIZE = 300000
    #SIZE = 25690000
    SIZE = 10000000


# Dynamic batch size: We want to have smaller chunks for the first
# few thousand instances (this helps when plotting stuff).
def dyn_batch_gen():
    for i in xrange(4, 22):
        yield 2 ** i
    while True:
        yield BATCH_SIZE


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


def transform(vectorizer, batch, fit=True):
    """
    Transforms a batch in order to be fed to sklearn.
    """
    tweets, outcomes = zip(*batch)
    y = np.array(outcomes)
    if fit:
        X = vectorizer.fit_transform(tweets)
    else:
        X = vectorizer.transform(tweets)

    return (X, y)


if __name__ == "__main__":
    with open('split_result.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(['size', 'classifier', 'pp', 'precision', 'recall', 'f1_score'])

        # Which of the 4 independent dynamic batch size iterators to use
        dyn_batch_num = -1

        # Minimal or full preprocessing
        for full_pp in [True, False]:

            # Train and evaluate with two classifiers
            for cls in [SGDClassifier(n_iter=1), MultinomialNB()]:
                batch_sizes = BATCH_SIZE if TESTING else dyn_batch_gen()

                print("Now training a %s with ~%s training instances and %s pp" % (
                    cls.__class__.__name__, int(SIZE * (1 - HOLDOUT)), "full" if full_pp else "minimal"))

                pp = PreProcessor("tweets.small.db" if TESTING else "tweets.all_shuffled.db", full_pp)
                tweets = pp.tweets()

                vectorizer = HashingVectorizer(stop_words=stopwords.words("english"),
                                               non_negative=True if cls.__class__.__name__ == "MultinomialNB" else False)

                # Create the test set. Do note that we did shuffle the database
                # before and retweets are removed by the PreProcessor.
                test_tweets = take(int(HOLDOUT * SIZE), tweets)

                # Not quite sure if we should fit or not
                X_test, y_test = transform(vectorizer, test_tweets, False)

                # Limit the train set
                limited_tweets = islice(tweets, int(SIZE * (1 - HOLDOUT)))

                n_instances = 0

                while True:
                    batch = take(BATCH_SIZE if TESTING else batch_sizes.next(), limited_tweets)
                    if not batch:
                        # no more instances
                        break

                    n_instances += len(batch)
                    X, y = transform(vectorizer, batch)

                    # Partially fit the instances
                    cls.partial_fit(X, y, classes=MAPPING.values())
                    results = evaluate(cls, X_test, y_test)

                    # Write intermediate results to file
                    writer.writerow([n_instances, cls.__class__.__name__, full_pp, results[0], results[1], results[2]])
                    f.flush()

                    print("\rConsumed: {0} instances".format(n_instances), end="")
                    sys.stdout.flush()
                print()

                pickle.dump(cls, cls.__class.__name__ + ".p")
