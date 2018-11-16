import read_data as rd
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB

def run():
  # run the read_data.py file to read in the dataset from CSV to a numpy array
  spamdata = rd.run()
  
  # split the data into two arrays
  # X is the matrix with all data points and attributes except for the spam (1) or not spam (0) attribute
  # y is the matrix with the spam attribute
  X = spamdata[:,:57]
  y = spamdata[:,57]
  #print '\nX.shape\n', X.shape
  #print '\ny.shape\n', y.shape
  #print '\nX\n', X
  #print '\ny\n', y

  # divide the data into training and test sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
  #print '\nX_train\n', X_test
  #print '\nX_test\n', X_test
  #print '\ny_train\n', y_train
  #print '\ny_test\n', y_test

  # training the algorithm on the training data, using SVM and Naive Bayes classifier
  svm_model = SVC(kernel='linear')
  nb_model = MultinomialNB()
  svm_model.fit(X_train, y_train)
  nb_model.fit(X_train, y_train)

  # make predictions
  y_pred_svm = svm_model.predict(X_test)
  y_pred_nb = nb_model.predict(X_test)

  # can update results based on what we want
  # confusion matrix: 0,0 - true negatives, 0,1 - false positives, 1,0 - false negatives, 1,1true positives
  print 'SVM\n'
  print(confusion_matrix(y_test,y_pred_svm))
  print(classification_report(y_test,y_pred_svm))
  print 'Naive Bayes\n'
  print(confusion_matrix(y_test,y_pred_nb))
  print(classification_report(y_test,y_pred_nb))

run()
