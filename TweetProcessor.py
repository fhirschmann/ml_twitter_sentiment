import sqlite3
import re
import pprint

from data import MAPPING


class TweetProcessor:
    def __init__(self, database_url):
        self.connection = sqlite3.connect(database_url)
        self.corpus = None
        self.raw = None

    def process_tweets(self, tweets, additional_preprocessing = True):
        return map(lambda x: self.process_tweet(x, additional_preprocessing), filter(self.allow_tweet, tweets))

    def process_tweet(self, tweet, additional_preprocessing = True):
        words = tweet[1].replace('"', '').replace('\n', ' ').split(' ')
        clean = []
        for word in words:
            clean += self.process_word(word, additional_preprocessing)
        return (clean, MAPPING[tweet[0]])

    def process_word(self, word, additional_preprocessing = True):
        if len(word) > 0:
            if additional_preprocessing:
                if word[0] == '@':
                    return [u'<user>']
                if word[0] == '#':
                    return [w.lower() for w in re.sub(r'([a-z])([A-Z])', r'\g<1> \g<2>', word[1:]).split(' ')]
                if word[:4] == 'http':
                    return [u'<url>']
            # unicode-heart, html-heart, encoded html-heart, text-heart,
            # everything ending with ) and not starting with (, everything ending with )
            # everything starting with :
            for mood in [u'\u2665', r'&lt;3+', r'&amp;lt;3+', r'<3+', r'[^\(]*\)+', r'.*\(+', r':.*']:
                word = re.sub(mood, '', word)
            if len(word) > 0:
                return [word.lower()]
            else:
                return []
        else:
            return []

    def allow_tweet(self, tweet):
        return tweet[1].find('RT @') == -1

    def build_raw(self, amount = False):
        if amount:
            self.raw = self.connection.execute('SELECT search_term, text FROM tweets LIMIT 0, %d' % amount).fetchall()
        else:
            self.raw = self.connection.execute('SELECT search_term, text FROM tweets').fetchall()

    def get_raw(self, amount = False):
        if not self.raw or (amount and len(self.raw) != amount) or not amount:
            self.build_raw(amount)
        return self.raw

    def build_corpus(self, additional_preprocessing = True, amount = False):
        self.corpus = self.process_tweets(self.get_raw(amount), additional_preprocessing)

    def get_corpus(self, additional_preprocessing = True, amount = False):
        if not self.corpus or (amount and len(self.corpus) != amount) or not additional_preprocessing:
            self.build_corpus(additional_preprocessing, amount)
        return self.corpus


if __name__ == '__main__':
    tp = TweetProcessor('tweets.small.db')
    pprint.pprint(tp.get_corpus())
