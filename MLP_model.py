# -*- coding: utf-8 -*-

import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier


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
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.5, random_state=0)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

"""# CNN Classifier"""
# create the classifer and fit the training data and lables
classifier_cnn= MLPClassifier(hidden_layer_sizes=(13,13,13,3),max_iter=550).fit(X_train, y_train)

predictions = classifier_cnn.predict(X_test)
print(" y = ",y_test)
print("predict = ", predictions)
print("accuracy" ,classifier_cnn.score(X_test, y_test))
print("\nClassification_report of SVM classifier:")
print(classification_report(y_test,predictions))
print("----------------------------------------------------------------------------")


with open("models/arabic_sentiment_cnn_tokenizer.pickle", "wb") as f:
    pickle.dump(tf_vec, f)
with open("models/arabic_sentiment_cnn.pickle", "wb") as f:
    pickle.dump(classifier_cnn, f)

