# Twitter sentiment analysis project
#
# Group name:
# Group members: Dominik Schreiber, Ji-Ung Lee, Fabian Hirschmann
#
import sys
import cPickle as pickle

from vect import vectorizer
from pp import PreProcessor, INV_MAPPING


class TwitterSentiment:
    def __init__(self):
        self.vec = vectorizer
        self.pp = PreProcessor(full_pp=True)
        self.cls = None

    def predict(self, text):
        '''predict an emoticon for any string given by text by using a trained classifier'''
        return self.predict_all([text])[0]

    def predict_all(self, seq):
        '''predict all emoticons for a list of strings by using a trained classifier'''
        return self.cls.predict(self.vec.transform(map(self.pp.process_tweet, seq)))


if __name__ == '__main__':
    ts = TwitterSentiment()
    ts.cls = pickle.load(open("MultinomialNB_pp.p", "rb"))

    if "--dump" in sys.argv:
        with open('TwitterSentiment.pickle', 'wb') as f:
            pickle.dump(ts, f)

    text = raw_input('How may I serve you, humble master?\n')
    while text is not 'q':
        print(INV_MAPPING[ts.predict(text)])
        text = raw_input('How may I serve you, humble master? [q to quit]\n')
    print('Master has given Dobby a sock. Dobby is free.')
