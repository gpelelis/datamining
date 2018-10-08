#!/usr/bin/env python
import pandas as pd
from pandas import read_csv
import numpy as np
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer,ENGLISH_STOP_WORDS
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold
from sklearn import svm
from itertools import cycle

print 'Predicting categories for test file '

TRAIN_SET_FILE = '../datasets/train_set.csv'
TEST_SET_FILE = '../datasets/test_set.csv'

features = 250
classifier = svm.LinearSVC()

print 'Preparing file to train model'
file_contents = read_csv(TRAIN_SET_FILE, sep='\t')
words = file_contents.Title + file_contents.Content

print 'Vectorizing data & feature selection'
vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words=ENGLISH_STOP_WORDS) # prepare vectorizer
X_train = vectorizer.fit_transform(words) # get tf-idf 
Y_train = np.array(file_contents["Category"])

print 'Dimensionality reduction'
svd = TruncatedSVD(n_components=features) # http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
X_train = svd.fit_transform(X_train, Y_train)

folds = 10
fold = KFold(n_splits=folds)

print 'Training the model'
for train_rows, test_rows in fold.split(X_train):
    classifier.fit(X_train[train_rows], Y_train[train_rows])

print 'Reading and preparing test file for prediction'
test_contents = read_csv(TRAIN_SET_FILE, sep='\t')
words_to_test = test_contents.Title + test_contents.Content
X_test = vectorizer.fit_transform(words_to_test)
X_test = svd.fit_transform(X_test)

print 'Started the prediction'
pred = classifier.predict(X_test)

print 'Writing file with predictions'
file = open('testSet_categories.csv', 'w')
file.write('ID'+'\t'+'Predicted_Category'+'\n')

for id,predicted in zip(test_contents.Id,pred):
    file.write(str(id)+'\t'+str(predicted)+'\n')

print "Done!"
file.close()
