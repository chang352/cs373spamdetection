import csv
import numpy as np

def run():
    X = np.genfromtxt('spambase.data', delimiter=',')
    print X.shape
    print X[0]
    return X
