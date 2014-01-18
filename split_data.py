import sqlite3
import numpy

def split(inputDB, trainDB, testDB, length):
    input = sqlite3.connect(inputDB)
    train = sqlite3.connect(trainDB)
    test = sqlite3.connect(testDB)
    inputCursor = input.cursor()
    trainCursor = train.cursor()
    testCursor = test.cursor()
    #Check the parameterstyle for sqlite3
    #print(sqlite3.paramstyle)
    #Ensure that the test and train databases are empty
    if bool(trainCursor.execute("SELECT * FROM tweets")):
        trainCursor.execute("DROP TABLE tweets")
    if bool(testCursor.execute("SELECT * FROM tweets")):
        testCursor.execute("DROP TABLE tweets")
    
    trainCursor.execute("CREATE TABLE tweets('search_term' varchar(255), 'created_at' varchar(255), 'from_user' varchar(255), 'from_user_id' varchar(255), 'from_user_name' varchar(255), 'geo' varchar(255),'id' varchar(255), 'in_reply_to_status_id' varchar(255), 'iso_language_code' varchar(255), 'source' varchar(255), 'text' varchar(255), 'to_user' varchar(255), 'to_user_id' varchar(255), 'to_user_name' varchar(255))")
    testCursor.execute("CREATE TABLE tweets('search_term' varchar(255), 'created_at' varchar(255), 'from_user' varchar(255), 'from_user_id' varchar(255), 'from_user_name' varchar(255), 'geo' varchar(255),'id' varchar(255), 'in_reply_to_status_id' varchar(255), 'iso_language_code' varchar(255), 'source' varchar(255), 'text' varchar(255), 'to_user' varchar(255), 'to_user_id' varchar(255), 'to_user_name' varchar(255))")
    inputCursor.execute("SELECT * FROM tweets")
    tweets = [tweet for tweet in inputCursor.fetchall()]
    numpy.random.shuffle(tweets)

    #Setting Percentage of the split
    split = len(tweets)/2
    if (length > 0) & (length < 1):
        split = length*len(tweets)
        
    #for tweet in tweets[:split]:
    #        trainCursor.execute("INSERT INTO tweets '%s' " % tweet)
    #        print("Inserting some training data")
    for tweet in tweets[split:]:
            testCursor.execute("INSERT INTO tweets (?,?,?,?,?,?,?,?,?,?,?,?,?,?) ", (tweet))
            print("Inserting some testing data")
    train.commit(), test.commit()
    train.close(), test.close(), input.close()

            
if __name__ == "__main__":
    split("tweets.small.db", "test.db", "train.db", -1)
