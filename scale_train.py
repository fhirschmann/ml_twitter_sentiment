#!/usr/bin/env python
"""
Evaluates different classifiers with increasing the training dataset
"""
from __future__ import print_function

import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, HashingVectorizer
from sklearn.linear_model import Perceptron, RidgeClassifier, RidgeClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.cross_validation import cross_val_score
from pandas import DataFrame

from data import retrieve_tweets_db


pipeline = [
    ('tfidf', TfidfTransformer()),
]

if __name__ == "__main__":
    allData, outcome = retrieve_tweets_db("tweets.small.db", 100)
    breakpoint = len(allData)
    if breakpoint > 1000:
	breakpoint = int(len(allData)*0.05)
    results = []
    # 2^5 = 32, with 10 folds we need at least more than one example for 
    # each fold, which sums up to 10*2 = 20 examples at least for training.
    iterator = 5
    while ((2**iterator)-1) < breakpoint:
	SomeData = random.sample(allData, ((2**iterator)-1))
	y = np.array(SomeData)
	vectorizer = CountVectorizer()
    	X = vectorizer.fit_transform(SomeData)
	scores = cross_val_score(Pipeline(pipeline+[('ridge', RidgeClassifier())]), X, y, scoring=f1_score, cv=10, n_jobs=1, verbose=0, fit_params=None, score_func=f1_score, pre_dispatch='2*n_jobs')
    	results.append((len(SomeData),scores))
	iterator = iterator + 1
    if breakpoint < 1000:
	y = np.array(allData)
	vectorizer = CountVectorizer()
    	X = vectorizer.fit_transform(allData)
	scores = cross_val_score(Pipeline(pipeline+[('ridge', RidgeClassifier())]), X, y, scoring=f1_score, cv=10, n_jobs=1, verbose=0, fit_params=None, score_func=f1_score, pre_dispatch='2*n_jobs')
    	results.append((len(allData),scores))

    amount = [x[0] for x in results]
    score = [0.1*sum(x[1]) for x in results]
    print(amount, score)
    df = DataFrame.from_dict({x[0]:[0.1*sum(x[1])] for x in results})
    with open("res_" + "Ridge_Scale" + ".tex", "w") as f:
	f.write(df.to_latex())
    #with open("res_" + "Ridge_Scale" + ".csv", "w") as f:
    #	f.write(df.to_csv())
    print(df)
    print()
