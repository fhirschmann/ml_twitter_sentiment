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


if __name__ == '__main__':
    ts = TwitterSentiment()

    #you can vectorize your TwitterSentiment class by using pickle:
    #pickle.dump(ts, 'TwitterSentiment.pickle')
