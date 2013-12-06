#!/usr/bin/env python
"""
This module uses 10-fold cross-validation in order to find
the best algorithm and parameter using grid search.
"""
from __future__ import print_function

import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import Perceptron, RidgeClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from pandas import DataFrame

from data import retrieve_tweets_db


estimators = {
    'svm': (svm.SVC(), {'svm__C': [0.1, 10, 100, 1000, 10000],
                        'svm__kernel': ['linear', 'poly', 'rbf']}),
    'nn': (Perceptron(), {'nn__alpha': [0.0001, 0.001]}),
    'ridge': (RidgeClassifier(), {'ridge__alpha': [1, 0.1, 0.001]}),
}


if __name__ == "__main__":
    tweets, outcomes = retrieve_tweets_db("tweets.small.db", 1000)

    y = np.array(outcomes)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(tweets)

    pipeline = [
        ('tfidf', TfidfTransformer()),
    ]

    results = []

    for name, estimator in estimators.items():
        grid_search = GridSearchCV(Pipeline(pipeline + [(name, estimator[0])]),
                                   estimator[1], score_func=f1_score)

        grid_search.fit(X, y)
        results.append((name, grid_search))

    results = sorted(results, reverse=True, key=lambda r: r[1].best_score_)

    for name, grid_search in results:
        print("==== " + name + " ====")
        print("Best score: %0.3f" % grid_search.best_score_)
        print("Best parameters set:")
        best_parameters = grid_search.best_estimator_.get_params()
        for param_name in sorted(estimators[name][1].keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))
        print("Complete grid:")
        df = DataFrame.from_dict([dict(x[0], **{'score': x[1]}) for x in grid_search.grid_scores_])
        with open("res_" + name + ".tex", "w") as f:
            f.write(df.to_latex())
        print(df)
        print()
