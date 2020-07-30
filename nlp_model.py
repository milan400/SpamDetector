# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


# Read dataset
df = pd.read_csv('spam.csv', encoding="latin-1")

df.columns

df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

df['label'] = df['class'].map({'ham':0, 'spam':1})

X = df['message']
y= df['label']

#pipeline--->countvectorizer ,Tfidftransformer and MultinominalNB
pipeline = Pipeline([
     ('vect', CountVectorizer()),
    ('clf', MultinomialNB()),
    ])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

pipeline.fit(X_train, y_train)
pipeline.score(X_test, y_test)

#Save model
filename = 'nlp_model.pkl'
pickle.dump(pipeline, open(filename, 'wb'))




































