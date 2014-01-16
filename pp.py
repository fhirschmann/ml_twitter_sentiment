import sqlite3


MAPPING = {
    ':)': 0,
    ':(': 1,
    '<3': 2
}

# Migh be faster to use re.compile


class PreProcessor(object):
    """
    Python generator that fetches and preprocesses tweets
    in a lazy fashion.
    """
    def __init__(self, db, amount=None, full_pp=True):
        connection = sqlite3.connect(db)

        if amount:
            self.query = connection.execute('SELECT search_term, text FROM tweets LIMIT 0, %d' % amount)
        else:
            self.query = connection.execute('SELECT search_term, text FROM tweets')

    def tweets(self):
        while True:
            res = self.query.fetchone()
            if not res:
                break

            tweet = res[1]
            cls = MAPPING[res[0]]

            if "RT @" in tweet:
                # we just don't yield retweets
                continue

            else:
                words = tweet.replace('"', '').replace('\n', ' ').split(' ')
                if len(words) == 0:
                    continue

                # blah blah
                yield (tweet, cls)


if __name__ == "__main__":
    pp = PreProcessor("tweets.small.db", 100)
    for tweet in pp.tweets():
        print(tweet)
