from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import HashingVectorizer


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


vectorizer = HashingVectorizer(norm="l1", stop_words=stopwords.words("english"),
                               tokenizer=LemmaTokenizer())
