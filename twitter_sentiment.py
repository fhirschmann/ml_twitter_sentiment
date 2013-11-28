# Twitter sentiment analysis project
#
# Group name:
# Group members: Dominik Schreiber, Ji-Ung Lee, Fabian Hirschmann
#

import sqlite3
import csv
import pickle

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
	#call your functions here to load and process the corpus
	#then vectorize your data and train a classifier

	#you can vectorize your TwitterSentiment class by using pickle:
	pickle.dump(ts, 'TwitterSentiment.pickle')
