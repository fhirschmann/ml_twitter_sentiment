import sqlite3
import re


class TweetProcessor:
	connection = False
	corpus = False

	def __init__(self, database_url):
		self.connection = sqlite3.connect(database_url)

	def get_raw(self, amount = False):
		if amount:
			return self.connection.execute('SELECT search_term, text FROM tweets LIMIT 0, %d' % amount).fetchall()
		else:
			return self.connection.execute('SELECT search_term, text FROM tweets').fetchall()

	def process_tweets(self, tweets):
		return map(self.process_tweet, filter(self.allow_tweet, tweets))

	def process_tweet(self, tweet):
		words = tweet[1].replace('"', '').replace('\n', ' ').split(' ')
		clean = []
		for word in words:
			clean += self.process_word(word)
		return (tweet[0], clean)

	def process_word(self, word):
		if len(word) > 0:
			if word[0] == '@':
				return ['<user>']
			if word[0] == '#':
				return re.sub(r'([a-z])([A-Z])', r'\g<1> \g<2>', word[1:]).split(' ')
			if word[:4] == 'http':
				return ['<url>']
			for mood in [u'\u2665', r'&lt;3+', r'&amp;lt;3+', r'<3+', r':\)+', r':-\)+', r':\(+', r':-\(+']:
				word = re.sub(mood, '', word)
			if len(word) > 0:
				return [word]
			else:
				return []
		else:
			return []

	def allow_tweet(self, tweet):
		return tweet[1].find('RT @') == -1

	def build_corpus(self, amount = False):
		self.corpus = self.process_tweets(self.get_raw(amount))

	def get_corpus(self, amount = False):
		if not self.corpus or (amount and len(self.corpus) != amount):
			self.build_corpus(amount)
		return self.corpus


if __name__ == '__main__':
	tp = TweetProcessor('tweets.small.db')
	for search, tweet in tp.get_corpus():
		print search, '->', ' '.join(tweet)
