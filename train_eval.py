#!/usr/bin/env python
"""
This module uses 10-fold cross-validation in order to find
the best algorithm and parameter using grid search.
"""
import numpy as np

from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

from data import retrieve_tweets_db


estimators = [("svm", svm.SVC())]

params = {
    "svm__C": [0.1, 10, 100, 1000, 10000]
}


if __name__ == "__main__":

    tweets, outcomes = retrieve_tweets_db("tweets.small.db", 1000)

    y = np.array(outcomes)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(tweets)

    pipeline = Pipeline(estimators)

    grid_search = GridSearchCV(pipeline, params, score_func=f1_score)

    grid_search.fit(X, y)

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(params.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
