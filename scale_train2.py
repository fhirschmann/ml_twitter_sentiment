#!/usr/bin/env python
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.cross_validation import cross_val_score
import numpy as np

from TweetProcessor import TweetProcessor

vectorizer = HashingVectorizer()


if __name__ == "__main__":
    # Total corpus sizes (11, 20) may be a good range?
    for exp in range(11, 13):
        size = 2 ** exp

        # Minimal or full preprocessing
        for pp in [True, False]:

            # Train and evaluate with two classifiers
            for cls in [RidgeClassifier(), SGDClassifier()]:
                print("Now training a %s classifier with %s instances and %s pp" % (
                    cls.__class__.__name__, size, "full" if pp else "minimal"))

                processor = TweetProcessor("tweets.small.db")
                corpus = processor.get_corpus(pp, size)
                tweets, outcomes = zip(*corpus)

                # Why is this required? Dominik?
                tweets = [" ".join(sen) for sen in tweets]
                y = np.array(outcomes)

                X = vectorizer.fit_transform(tweets)

                scores = cross_val_score(cls, X, y, cv=5, scoring="f1")
                print np.mean(scores)
