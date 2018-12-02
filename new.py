import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from random import randrange
import read_data as rd

def run():
  # run the read_data.py file to read in the dataset from CSV to a numpy array
    spamdata = rd.read_data()


    rowList = []
    
    for b in range(0,200):
        k = randrange(0,1812)
        rowList.append(k)
    
    for b in range(0,200):
        k = randrange(1813,4601)
        rowList.append(k)
    
    
    print (rowList)
  
  # split the data into two arrays
  # X is the matrix with all data points and attributes except for the spam (1) or not spam (0) attribute
  # y is the matrix with the spam attribute
    X = spamdata[rowList,:57]
    y = spamdata[rowList,57]
  
  #print '\nX.shape\n', X.shape
  #print '\ny.shape\n', y.shape
  #print '\nX\n', X
  #print '\ny\n', y
    
    
    
  # divide the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
    X_validation, X_finalTest, y_validation, y_finalTest = train_test_split(X_test, y_test, test_size = 0.50)
    #print (X_train)
  #print '\nX_train\n', X_test
  #print '\nX_test\n', X_test
  #print '\ny_train\n', y_train
  #print '\ny_test\n', y_test

  #Splitting the data into five folds to train
    split_dataX = []
    split_dataY = []

    copy1 = X_train.tolist()
    
    copy2 = y_train.tolist()
    
    foldsize = int(len(X_train)/5)
    
    for i in range (0,5):
        fold1 = []
        fold2 = []
    
        indexList = []
        
        while len(fold1) < foldsize:
            
            index = randrange(0 , len(copy1))
            #print (index)
            if index not in indexList:
                index = randrange(0 , len(copy1))
                
                fold1.append(copy1[index])
                fold2.append(copy2[index])
            
            indexList.append(index)
        
        split_dataX.append(fold1)
        split_dataY.append(fold2)
        
#training on the solit data to get the best value for hyperparameter 

    hyperparameterList = [0.1,1,10]
    scores = []
    validationScores = []

    #print (len(split_dataX))
    
    for k in range (0,3):
        parameterValue = hyperparameterList[k]
        
        dataSetScores = []
        
        for i in split_dataX:
            
            #dataSetScores = []
            indexOfi = split_dataX.index(i)
    
            completeListX = split_dataX.copy()
            testDataX = i 
            completeListX.remove(i)
            trainingDataX = completeListX
    
            completeListY = split_dataY.copy()
            testDataY = split_dataY[indexOfi]
            completeListY.remove(testDataY)
            trainingDataY = completeListY
        
            flat_list = [] 
            for sublist in trainingDataX:
                for item in sublist:
                    flat_list.append(item)
        
            flat_list2 = []
            for sublist in trainingDataY:
                for item in sublist:
                    flat_list2.append(item)
        
            svm_model = SVC(kernel='linear', C = parameterValue)
            svm_model.fit( flat_list, flat_list2 )
            score = svm_model.score(testDataX,testDataY)
            dataSetScores.append(score)
        
        a = 0 
        for m in dataSetScores:
            a = a+m
        average = a/len(dataSetScores)
        
        scores.append(average)
        
        score = svm_model.score(X_validation,y_validation)
        validationScores.append(score)
        
        #print ("Finished")
        
    #print (scores)
    #print (validationScores)

    best1 = scores[0]
    best2 = validationScores[0]
    
    for i in range(1,3):
        
        if scores[i]>best1:
            if validationScores[i]>best2:
                best1 = scores[i]
                best2 = validationScores[i]
        
    #print (best1)
    #print (best2)
    
    indexOfHyperparameter = scores.index(best1)
    hyperparameterValue = hyperparameterList[indexOfHyperparameter]

    print ("Ideal Hyperparameter:" , hyperparameterValue)
    
    
    svm_model = SVC(kernel='linear', C = hyperparameterValue)
    svm_model.fit( X_train, y_train ) 
    finalScoreSVM = svm_model.score(X_finalTest,y_finalTest)
    print ("SVM accracy :", finalScoreSVM)
    
    #Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    finalScoreNB = nb_model.score(X_finalTest, y_finalTest)
    print ("Naive Bayes accuracy: " , finalScoreNB)    
    
    
    
run()
print ("Done")

