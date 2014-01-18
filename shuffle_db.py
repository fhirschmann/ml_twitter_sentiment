import sqlite3
import os

DB = "tweets.small.db"

if __name__ == "__main__":
    db = sqlite3.connect(DB)
    try:
        os.remove(DB[:-3] + "_shuffled.db")
    except OSError:
        pass
    db2 = sqlite3.connect(DB[:-3] + "_shuffled.db")
    c = db.cursor()
    db2.cursor().execute("CREATE TABLE tweets('search_term' varchar(255), 'created_at' varchar(255), 'from_user' varchar(255), 'from_user_id' varchar(255), 'from_user_name' varchar(255), 'geo' varchar(255),'id' varchar(255), 'in_reply_to_status_id' varchar(255), 'iso_language_code' varchar(255), 'source' varchar(255), 'text' varchar(255), 'to_user' varchar(255), 'to_user_id' varchar(255), 'to_user_name' varchar(255))")
    c2 = db2.cursor()

    query = c.execute("SELECT * from tweets ORDER BY RANDOM()")

    i = 0
    while True:
        row = query.fetchone()
        if not row:
            break
        i += 1

        c2.execute("INSERT INTO tweets VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?) ", row)
        if (i % 10000) == 0:
            db2.commit()
            print("Wrote %s instances" % i)
