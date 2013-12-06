# Twitter sentiment analysis project
#
# Group name:
# Group members: Dominik Schreiber, Ji-Ung Lee, Fabian Hirschmann
#

import sqlite3
import csv
import pickle
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import f1_score
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from nltk.classify.scikitlearn import SklearnClassifier
from nltk import word_tokenize

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

MAPPING = {
    ':)': 0,
    ':(': 1,
    '<3': 2
}

#This class is used to test your classifier on unseen data in Feburary.
#It should contain everything that is needed to predict a text, e.g. your classifier and vectorizer,
#so that you can serialize it properly. That also means: don't use global variables inside '__init__',
#'predict' and 'predict_all'

class TwitterSentiment:
    def __init__(self):
        #set all needed variables and parameters here
        #self.param = 'something'
        #self.vectorizer = ...
        #self.classifier = ...
        pass

    def predict(self, text):
        '''predict an emoticon for any string given by text by using a trained classifier'''
        return  #0 for :), 1 for :(, 2 for <3

    def predict_all(self, seq):
        '''predict all emoticons for a list of strings by using a trained classifier'''
        return  #list of predictions

#If you have enough memory (min 8GB), you can load the sqlite version of the corpus completely into memory
#If not, you may want to modify this function (or use the CSV data if you don't like SQL) to load the data in chunks that fit in your memory
#You can also retrieve a smaller portion of the data with the limit parameter

def retrieve_tweets_db(db_name, limit=-1) :
    conn = sqlite3.connect(db_name)
    crs = conn.cursor()

    if(limit == -1):
        crs.execute('SELECT text, search_term FROM tweets')
    else:
        crs.execute('SELECT text, search_term FROM tweets LIMIT 0,' + str(limit))

    corpus = crs.fetchall()
    return corpus


if __name__ == '__main__':
    ts = TwitterSentiment()

    corpus = retrieve_tweets_db("tweets.small.db", 1000)

    dt = [(t, MAPPING[s]) for (t, s) in corpus]
    tweets, outcomes = zip(*dt)
    y = np.array(outcomes)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(tweets)

    estimators = [("svm", svm.SVC())]
    params = {
        "svm__C": [0.1, 10, 100, 1000, 10000]
    }

    pipeline = Pipeline(estimators)

    grid_search = GridSearchCV(pipeline, params, score_func=f1_score)

    grid_search.fit(X, y)

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(params.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    #you can vectorize your TwitterSentiment class by using pickle:
    #pickle.dump(ts, 'TwitterSentiment.pickle')
