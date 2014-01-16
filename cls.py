from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import MultinomialNB

cls1 = SGDClassifier(loss="log", n_jobs=-1)
cls2 = RidgeClassifier()
cls2 = MultinomialNB()
