import csv
import numpy as np

def read_data():
    X = np.genfromtxt('spambase.data', delimiter=',')
    #print (X.shape)
    #print (X[0])
    return X
