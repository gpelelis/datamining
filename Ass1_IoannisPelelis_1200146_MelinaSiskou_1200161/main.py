#!/usr/bin/env python
from sys import argv
from wordcloud import WordCloud
import random
import time
import pandas as pd
from pandas import read_csv
import numpy as np
from sklearn import preprocessing
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer,ENGLISH_STOP_WORDS
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold
from sklearn import svm
from itertools import cycle
import matplotlib.pyplot as plt


from Categories import Categories

# CONSTANTS
TRAIN_SET_FILE = '../datasets/train_set.csv'
TEST_SET_FILE = '../datasets/test_set.csv'
KAGGLE_TRAIN = '../kaggle/train_set.csv'
KAGGLE_TEST = '../kaggle/test_set.csv'
IMAGE_FOLDER = '../output/images/'
EVAL_FOLDER = '../output/'


# FUNCTIONS
def generate_word_cloud():
  # get the file from train set and prepare an object that keeps all the information for the files
  categories = Categories(TRAIN_SET_FILE)

  for category_name in categories.categories:
    text = categories.get_body_as_string(category_name)
    wordcloud = WordCloud(stopwords=ENGLISH_STOP_WORDS).generate(text)

    # display the image
    image = wordcloud.to_file(IMAGE_FOLDER + category_name + '.png')
    print('generated the image for ' + category_name + '. Check it at ' + IMAGE_FOLDER + category_name + '.png')

def predict_for_test_file(kaggle=False):
    print 'Predicting categories for test file '

    features = 250
    classifier = svm.LinearSVC()

    print 'Preparing file to train model'
    if (kaggle):
        file_contents = read_csv(KAGGLE_TRAIN, sep='\t')
    else:
        file_contents = read_csv(TRAIN_SET_FILE, sep='\t')
    words = file_contents.Title*10 + file_contents.Content

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
        classifier.fit(X_train, Y_train)

    print 'Reading and preparing test file for prediction'
    if (kaggle):
        test_contents = read_csv(KAGGLE_TEST, sep='\t')
    else:
        test_contents = read_csv(TEST_SET_FILE, sep='\t')
    words_to_test = test_contents.Title*10 + test_contents.Content
    X_test = vectorizer.fit_transform(words_to_test)
    X_test = svd.fit_transform(X_test)

    print 'Started the prediction'
    pred = classifier.predict(X_test)

    print 'Writing file with predictions'
    
    if (kaggle):
        file_for_predictions = open('../kaggle/solution.csv', 'w')
        file_for_predictions.write('Id,Category'+'\n')
        delim = ','
    else:
        file_for_predictions = open('../output/testSet_categories.csv', 'w')
        file_for_predictions.write('ID'+'\t'+'Predicted_Category'+'\n')
        delim = '\t'
    
    for id,predicted in zip(test_contents.Id,pred):
        file_for_predictions.write(str(id) + delim + str(predicted)+'\n')

    print "Done!"
    file_for_predictions.close()



file = open(EVAL_FOLDER + 'EvaluationMetric_10fold.csv', 'w')
output = ['Statistic Measure'+'\t'+'Naive Bayes'+'\t'+'Random Forest'+'\t'+'SVM'+'\t'+'my_method','Accuracy','Precision','Recall','F-Measure']
def generate_test_metrics(classifier_name, test=False):
    print 'Starting to generate metrics for ' + classifier_name

    file_contents = read_csv(TRAIN_SET_FILE, sep='\t')
    words = file_contents.Title*10 + file_contents.Content

    vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words=ENGLISH_STOP_WORDS) # prepare vectorizer

    X_train_init = vectorizer.fit_transform(words) # get tf-idf 
    Y_train = np.array(file_contents["Category"])

    # 1. from tests on accuracy we've found an optimal value to use on each algorithm.
    if classifier_name == 'random_forests':
        classifier = RandomForestClassifier()
        test_feature_vals = [100] 
    elif classifier_name == 'bayes':
        classifier = GaussianNB()
        test_feature_vals = [100] 
    elif classifier_name == 'svm_clf':
        classifier = svm.LinearSVC()
        test_feature_vals = [100] 
    elif classifier_name == 'my_method':
        test_feature_vals = [250] # 1
        classifier = svm.LinearSVC()

    features_acc = []
    if (test):
        test_feature_vals = [20, 100, 200, 300, 400, 500, 600]
        # test_feature_vals = [10, 20, 30, 50, 70, 80, 100, 120, 150, 170, 200, 220, 250, 270, 300, 340, 360, 400]

    # if we are on test, test_feature_vals will have more than one value
    for features in test_feature_vals:
        print "Starting preprocess using " + str(features) + " features for " + classifier_name 
        X_train = X_train_init
        svd = TruncatedSVD(n_components=features) # http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
        X_train = svd.fit_transform(X_train)

        print "Starting k-fold validation for " + classifier_name 

        folds = 10
        fold = KFold(n_splits=folds)
        predicted = np.empty(shape=len(X_train),dtype=basestring)

        for train_rows, test_rows in fold.split(X_train):
            classifier.fit(X_train[train_rows], Y_train[train_rows])
            pred = classifier.predict(X_train[test_rows])

            predicted[test_rows] = np.array(pred)

        # uncomment the following section to test without kfold
        # sev_perc = int(len(X_train)*0.9)
        # test_rows = np.asarray(range(sev_perc, len(X_train)))
        # train_rows = np.asarray(range(0, sev_perc))
        
        # predicted = np.empty(shape=len(X_train),dtype=basestring)
        # classifier.fit(X_train[train_rows], Y_train[train_rows])
        # pred = classifier.predict(X_train[test_rows])

        # predicted[test_rows] = np.array(pred)

        accuracy = round(metrics.accuracy_score(Y_train[test_rows], predicted[test_rows]), 5)
        precision = round(metrics.precision_score(Y_train[test_rows], predicted[test_rows], average='macro'), 5)
        recall =  round(metrics.recall_score(Y_train[test_rows], predicted[test_rows], average='macro'), 5)
        fmeasure =  round(metrics.f1_score(Y_train[test_rows], predicted[test_rows], average='macro'), 5)
        features_acc.append(accuracy)

        if (not test):
            output[1] += ('\t'+str(accuracy))
            output[2] += ('\t'+str(precision))
            output[3] += ('\t'+str(recall))
            output[4] += ('\t'+str(fmeasure))
        
        print('For features: ' + str(features) + ' the accuracy: ' + str(accuracy) + ', precision: ' + str(precision) + ', recall: ' + str(recall) + ', fmeasure: ' + str(fmeasure) + '\n')

    # if we are on test mode create the 
    if (test):
        plt.figure()
        plt.plot(test_feature_vals, features_acc)

        plt.xlabel('Number of components')
        plt.ylabel('Accuracy')
        plt.legend(loc="lower right")
        plt.title('Accuracy plot for classifier: ' + classifier_name)
        plt.savefig(IMAGE_FOLDER +'accuracy_'+classifier_name+'.png')

        print "Done with clasification"


# MAIN
if len(argv) < 2:
    file.close()
    exit()
for i in range(1, len(argv)):
    c = argv[i]
    if c == 'w':
        generate_word_cloud()
    elif c == 'b':
        generate_test_metrics('bayes', test=True)
    elif c == 'rf':
        generate_test_metrics('random_forests', test=True)
    elif c == 'svm':
        generate_test_metrics('svm_clf', test=True)
    elif c == 'knn':
        generate_test_metrics('knn_alg', 2)
    elif c == 'my':
        generate_test_metrics('my_method', test=True)
    elif c == 'all':
        generate_test_metrics('bayes')
        generate_test_metrics('random_forests')
        generate_test_metrics('svm_clf')
        generate_test_metrics('my_method')
        print('Writing the metrics to EvaluationMetric_10fold.csv')
        for i in range(5):
            output[i] += '\n'
            file.write(output[i])
        print('Bye')
    elif c == 'test_kaggle':
        predict_for_test_file(kaggle=True)
    elif c == 'test_categories':
        predict_for_test_file()
file.close()