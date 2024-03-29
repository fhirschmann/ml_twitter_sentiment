import sqlite3
import re

MAPPING = {
    ':)': 0,
    ':(': 1,
    '<3': 2
}

INV_MAPPING = {v: k for k, v in MAPPING.items()}


class PreProcessor(object):
    """
    Python generator that fetches and preprocesses tweets
    in a lazy fashion.
    """
    def __init__(self, db=None, full_pp=True):
        self.full_pp = full_pp

        if db:
            connection = sqlite3.connect(db)
            self.query = connection.execute('SELECT search_term, text FROM tweets')

    def tweets(self):
        while True:
            res = self.query.fetchone()
            if not res:
                # no more tweets from query
                break

            tweet = res[1]
            cls = MAPPING[res[0]]

            if "RT @" in tweet:
                # retweet -> don't yield it
                continue

            else:
                ptweet = self.process_tweet(tweet)
                if ptweet:
                    yield ptweet, cls
                else:
                    continue

    def process_tweet(self, tweet):
        words = tweet.replace('"', '').replace('\n', ' ').split(' ')

        if len(words) == 0:
            # tweet is empty -> don't yield it
            return None
        clean = []
        for word in words:
            clean += self.process_word(word)

        return ' '.join(clean)

    def process_word(self, word):
        if len(word) > 0:
            if self.full_pp:
                if word[0] == '@':
                    return [u'<user>']
                if word[0] == '#':
                    return re.sub(r'([a-z])([A-Z])', r'\g<1> \g<2>', word[1:]).split(' ')
                if word[:4] == 'http':
                    return [u'<url>']
            for mood in [u'\u2665', r'&lt;3+', r'&amp;lt;3+', r'<3+', r'[^\(]*\)+', r'.*\(+', r':.*']:
                word = re.sub(mood, '', word)
            if len(word) > 0:
                return [word]
            else:
                return []
        else:
            return []

if __name__ == "__main__":
    pp = PreProcessor("tweets.small.db", 100)
    for tweet, outcome in pp.tweets():
        print outcome, '->', tweet
