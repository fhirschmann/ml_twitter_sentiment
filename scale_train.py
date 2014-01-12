#!/usr/bin/env python
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
import numpy as np

from TweetProcessor import TweetProcessor

vectorizer = HashingVectorizer()


if __name__ == "__main__":
    # Total corpus sizes (11, 20) may be a good range?
    for exp in range(14, 16):
        size = 2 ** exp

        # Minimal or full preprocessing
        for pp in [True, False]:

            # Train and evaluate with two classifiers
            for cls in [RidgeClassifier(), SGDClassifier()]:
                print("Now training a %s classifier with %s instances (Train-test-split of 5 to 95) and %s pp" % (
                    cls.__class__.__name__, size, "full" if pp else "minimal"))

                processor = TweetProcessor("tweets.small.db")
                corpus = processor.get_corpus(pp, size)
                tweets, outcomes = zip(*corpus)

                tweets = [" ".join(sen) for sen in tweets]
                y = np.array(outcomes)

                X = vectorizer.fit_transform(tweets)
                
                X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.95, random_state=0)
                cls.fit(X_train,y_train)
                y_predicted = cls.predict(X_test)
                target_names = ['class 0', 'class 1', 'class 2']
                print(classification_report(y_test, y_predicted, target_names=target_names))