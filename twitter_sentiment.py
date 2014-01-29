# Twitter sentiment analysis project
#
# Group name:
# Group members: Dominik Schreiber, Ji-Ung Lee, Fabian Hirschmann
#

import pickle
import numpy as np

from sklearn.naive_bayes import MultinomialNB
from vect import vectorizer
from pp import PreProcessor, MAPPING

class TwitterSentiment:
    def __init__(self):
        self.vectorizer = vectorizer
        self.classifier = MultinomialNB()
        self.preprocessor = PreProcessor("tweets.small.db", True)
        X, y = self.transform(self.preprocessor.tweets())
        self.classifier.fit(X, y)

    def transform(self, batch):
        tweets, outcomes = zip(*batch)
        y = np.array(outcomes)
        X = self.vectorizer.fit_transform(tweets)
        return X, y

    def predict(self, text):
        '''predict an emoticon for any string given by text by using a trained classifier'''
        clean = []
        words = text.replace('"', '').replace('\n', ' ').split(' ')
        for word in words: clean += self.preprocessor.process_word(word)
        return self.classifier.predict(self.vectorizer.fit_transform([' '.join(clean)]))[0]

    def predict_all(self, seq):
        '''predict all emoticons for a list of strings by using a trained classifier'''
        return [self.predict(str) for str in seq]


if __name__ == '__main__':
    ts = TwitterSentiment()
    text = raw_input('How may I serve you, humble master?\n')
    while text is not 'q':
        prediction = ts.predict(text)
        for smiley, m in MAPPING.iteritems():
            if prediction == m:
                print smiley
        text = raw_input('How may I serve you, humble master? [q to quit]\n')
    print 'Master has given Dobby a sock. Dobby is free.'
    #you can vectorize your TwitterSentiment class by using pickle:
    #pickle.dump(ts, 'TwitterSentiment.pickle')
