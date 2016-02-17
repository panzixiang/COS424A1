import scipy.io as spio
import sys
import csv
import pickle
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import zero_one_loss
import random

def main():
    
    filenameLB = 'mfcc_lb.csv'
    featureDict = pickle.load(open('mfcc_fv.p', 'rb'))
    
    with open('mfcc_lb.csv') as f:
        reader = csv.reader(f)
        for row in reader:
            labels = row

    # select training and test sets
    TEidx = np.array(random.sample(range(0,1000), 100))
    
    training = []
    test = []
    
    trainingLB = []
    testLB = []

    # make numpy arrays
    for i in range(1000):
        if i in TEidx:
            test.append(featureDict[i])
            testLB.append(labels[i])
        else:
            training.append(featureDict[i])
            trainingLB.append(labels[i])
        
    # fit with classifier and predict
    X = np.array(training)
    Y = np.array(trainingLB)


    cla = GaussianNB()
    cla.fit(X,Y)
    predictions = cla.predict(np.array(test))
    errorRF = zero_one_loss(predictions, testLB)
    print errorRF
    

if __name__ == "__main__":
    main()
