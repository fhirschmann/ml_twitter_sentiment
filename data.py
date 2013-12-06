"""
This module contains utilities dealing with data import.
"""
import sqlite3

MAPPING = {
    ':)': 0,
    ':(': 1,
    '<3': 2
}


def retrieve_tweets_db(db_name, limit=-1):
    conn = sqlite3.connect(db_name)
    crs = conn.cursor()

    if(limit == -1):
        crs.execute('SELECT text, search_term FROM tweets')
    else:
        crs.execute('SELECT text, search_term FROM tweets LIMIT 0,' + str(limit))

    corpus = crs.fetchall()
    return zip(*[(t, MAPPING[s]) for (t, s) in corpus])
