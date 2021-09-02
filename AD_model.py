# -*- coding: utf-8 -*-


import pickle
import pandas as pd

from sklearn import tree
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

data_df = pd.read_csv('data/final.csv')
# remove the "Neutral" class
data_df = data_df[data_df['sentiment'] != "neutral"]

# change values to numeric
data_df['sentiment'] = data_df['sentiment'].map({'positive': 1, 'negative': 0})

# idneitfy the data and the labels
data = data_df['text']
target = data_df['sentiment']

data_df = data_df.dropna()

# Use TfidfVectorizer for feature extraction (TFIDF to convert textual data to numeric form):
tf_vec = TfidfVectorizer()
X = tf_vec.fit_transform(data)

# Training Phase

X_train, X_test, y_train, y_test=train_test_split(X,target,
                                               test_size=0.5,random_state=random.seed())

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
"""# Decision tree Classifier"""
# create the classifer and fit the training data and lables

classifier_DTC = tree.DecisionTreeClassifier()
classifier_DTC.fit(X_train, y_train) 

print(classifier_DTC.predict(X_test))
print(y_test)
print("score estimé du classifieur appris", classifier_DTC.score(X_test,y_test))
print("l'erreur comise est :",1-classifier_DTC.score(X_test,y_test))



with open("models/arabic_sentiment_Decision-Tree_tokenizer.pickle", "wb") as f:
    pickle.dump(tf_vec, f)
with open("models/arabic_sentiment_Decision-Tree.pickle", "wb") as f:
    pickle.dump(classifier_DTC, f)






