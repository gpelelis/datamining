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
IMAGE_FOLDER = '../output/images/'


# FUNCTIONS
def generate_word_cloud():
  # get the file from train set and prepare an object that keeps all the information for the files
  categories = Categories(TRAIN_SET_FILE)

  for category_name in categories.categories:
    text = categories.get_body_as_string(category_name)
    wordcloud = WordCloud().generate(text)

    # display the image
    image = wordcloud.to_file(IMAGE_FOLDER + category_name + '.png')
    print('generated the image for ' + category_name + '. Check it at output/' + category_name + '.png')


def generate_test_metrics(classifier_name):
    print 'Reading and vectorizing data...'

    file_contents = read_csv(TRAIN_SET_FILE, sep='\t')
    Categories = file_contents["Category"].tolist()
    words = file_contents.Title*15 + file_contents.Content

    vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words=ENGLISH_STOP_WORDS)
    X_train = vectorizer.fit_transform(words)

    features = 150
    svd = TruncatedSVD(n_components=features)
    X_train = svd.fit_transform(X_train)

    Y_train = np.array(file_contents["Category"])
    classes = np.unique(Y_train)
    n_classes = len(classes)
    Y_train_bin = preprocessing.label_binarize(Y_train, classes=classes)

    # clf = [GaussianNB(),
    #        RandomForestClassifier(),
    #        svm.LinearSVC(),
    #        'KNN']
    if classifier_name == 'random_forests':
        classifier = RandomForestClassifier()
    elif classifier_name == 'bayes':
        classifier = GaussianNB()
    elif classifier_name == 'svm_clf':
        classifier = svm.LinearSVC()
    elif classifier_name == 'knn':
        classifier = svm.LinearSVC()

    print "Done."
    print "Begin calculating metrics & roc plot for classifiers (using 10-fold cross validation) ... "

    folds = 10
    kf = KFold(n_splits=folds)

    output = ['Statistic Measure'+'\t'+'Naive Bayes'+'\t'+'Random Forest'+'\t'+'SVM'+'\t'+'KNN','Accuracy','Precision','Recall','F-Measure','AUC']
    file = open('EvaluationMetric_10fold.csv', 'w')

    predicted = np.empty(shape=len(X_train),dtype=basestring)

    for train_index, test_index in kf.split(X_train):
        if (classifier != 'KNN'):
            classifier.fit(X_train[train_index], Y_train[train_index])
            pred = classifier.predict(X_train[test_index])
        else:
            pred = KNN_Classifier(3, X_train[train_index], Y_train[train_index], X_train[test_index])

        predicted[test_index] = np.array(pred)

    predicted_bin = preprocessing.label_binarize(predicted, classes=classes)

    accuracy = round(metrics.accuracy_score(Y_train, predicted), 5)
    precision = round(metrics.precision_score(Y_train, predicted, average='macro'), 5)
    recall =  round(metrics.recall_score(Y_train, predicted, average='macro'), 5)
    fmeasure =  round(metrics.f1_score(Y_train, predicted, average='macro'), 5)
    AUC = round(metrics.roc_auc_score(Y_train_bin, predicted_bin, average='macro'), 5)

    output[1] += ('\t'+str(accuracy))
    output[2] += ('\t'+str(precision))
    output[3] += ('\t'+str(recall))
    output[4] += ('\t'+str(fmeasure))
    output[5] += ('\t'+str(AUC))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(Y_train_bin[:, i], predicted_bin[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])


    plt.figure()
    lw = 2
    colors = cycle(['aqua', 'orange', 'red', 'green', 'purple'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of '+str(classes[i])+' (area = {1:0.3f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    if(classifier != 'KNN'):
        plt.title('ROC plot for classifier: '+str(classifier.__class__.__name__))
        plt.savefig(IMAGE_FOLDER +'roc_10fold_'+str(classifier.__class__.__name__)+'.png')
    else:
        plt.title('ROC plot for classifier: KNN')
        plt.savefig(IMAGE_FOLDER +'roc_10fold_KNN.png')


    print "Done. Writing results in EvaluationMetric_10fold.csv & saving Roc plot pictures."

    for i in range(6):
        output[i] += '\n'
        file.write(output[i])
    file.close()



# MAIN
if len(argv) < 2:
   print("Need at least 1 argument. Type --help for help")
   exit()
for i in range(1, len(argv)):
    c = argv[i]
    if c == 'w':
        generate_word_cloud()
    elif c == 'rf':
        generate_test_metrics('random_forests')
    elif c == 'b':
        generate_test_metrics('bayes')
    elif c == 'svm':
        generate_test_metrics('svm_clf')
    elif c == 'knn':
        generate_test_metrics('knn_alg', 2)
    elif c == 'all':
        generate_test_metrics('random_forests')
        generate_test_metrics('bayes')
        generate_test_metrics('svm_clf')
    elif c == '--help':
        print("Help for ", argv[0])
        print("Options:")
        print("w: word cloud image generation")
        print("rf: Random Forest")
        print("b: Naive Bayes")
        print("svm: SVM")
        print("knn: K-NN")
        # print("prod: Predict the class of the documents in data_sets/test_set.csv\n\tusing the Random Forest algorithm")
    else:
        print("Wrong argument. Type --help for help")
