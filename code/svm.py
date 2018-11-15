import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

def run():
  # I downloaded the data file into the same folder as this file
  # had to add line at the beginning of data file that says all the attribute names
  spamdata = pd.read_csv("/homes/chang352/cs373/spambase/spambase.data");
  #print 'spamdata.shape\n', spamdata.shape
  #print '\nspamdata.head(1)\n', spamdata.head(1)

  # X is the matrix with all data points and attributes except for spam (1) or not spam (0) attribute
  # y is the matrix with spam attribute
  X = spamdata.drop('spam', axis=1)
  y = spamdata['spam']
  #print '\nX\n', X
  #print '\ny\n', y

  # divide the data into training and test sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
  #print '\nX_train\n', X_test
  #print '\nX_test\n', X_test
  #print '\ny_train\n', y_train
  #print '\ny_test\n', y_test

  # training the algorithm on the training data
  svclassifier = SVC(kernel='linear')
  svclassifier.fit(X_train, y_train)

  # make predictions
  y_pred = svclassifier.predict(X_test)

  # can update results based on what we want
  print(confusion_matrix(y_test,y_pred))
  print(classification_report(y_test,y_pred))

run()
