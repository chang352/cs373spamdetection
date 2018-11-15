import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB

def run():
  # preprocessing: add atribute variable line at the beginning of the csv
  attribute_line = "word_freq_make,word_freq_address,word_freq_all,word_freq_3d,word_freq_our,word_freq_over,word_freq_remove,word_freq_internet,word_freq_order,word_freq_mail,word_freq_receive,word_freq_will,word_freq_people,word_freq_report,word_freq_addresses,word_freq_free,word_freq_business,word_freq_email,word_freq_you,word_freq_credit,word_freq_your,word_freq_font,word_freq_000,word_freq_money,word_freq_hp,word_freq_hpl,word_freq_george,word_freq_650,word_freq_lab,word_freq_labs,word_freq_telnet,word_freq_857,word_freq_data,word_freq_415,word_freq_85,word_freq_technology,word_freq_1999,word_freq_parts,word_freq_pm,word_freq_direct,word_freq_cs,word_freq_meeting,word_freq_original,word_freq_project,word_freq_re,word_freq_edu,word_freq_table,word_freq_conference,char_freq_;,char_freq_(,char_freq_[,char_freq_!,char_freq_$,char_freq_#,capital_run_length_average,capital_run_length_longest,capital_run_length_total,spam\n"
  spamdata = open("/homes/chang352/cs373/spambase/spambase.data","r")
  first_line = spamdata.readlines()
  # if the line isn't currently present at the beginning of the file, then add it
  if first_line[0] != attribute_line:
    # take the first line of the file, and prepend the attribute_line
    first_line.insert(0,attribute_line)
    spamdata.close()
    spamdata = open("/homes/chang352/cs373/spambase/spambase.data","w")
    # write the updated line back into file
    spamdata.writelines(first_line)
    spamdata.close()

  # I downloaded the data file into the same folder as this file
  spamdata = pd.read_csv("/homes/chang352/cs373/spambase/spambase.data")
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
