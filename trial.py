import scipy.io as spio
import sys
import csv
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import svm
from sklearn.metrics import zero_one_loss
import random

global mfcc, chroma, energy, brightness, hcdf;
def main(arg):
    
    # first arg will be classifier
    if arg[0] == 'knn3':
        cla = KNeighborsClassifier(n_neighbors=3)
    elif arg[0] == 'knn5':
        cla = KNeighborsClassifier(n_neighbors=5)
    elif arg[0] == 'lda':
        cla = LinearDiscriminantAnalysis()
    elif arg[0] == 'qda':
        cla = QuadraticDiscriminantAnalysis()
    elif arg[0] == 'svmLin':
        cla = svm.SVC(kernel='linear')
    elif arg[0] == 'svmRbf':
        cla = svm.SVC(kernel='rbf')
    elif arg[0] == 'svmSig':
        cla = svm.SVC(kernel='sigmoid')
    else:
        exit()

    # open all pickles
    mfcc = pickle.load(open('mfcc_fv.p', 'rb'))
    chroma = pickle.load(open('chroma_fv.p', 'rb'))
    energy = pickle.load(open('energy_fv.p', 'rb'))
    brightness = pickle.load(open('brightness_fv.p', 'rb'))
    hcdf = pickle.load(open('hcdf_fv.p', 'rb'))

    # get labels
    with open('mfcc_lb.csv') as f:
        reader = csv.reader(f)
        for row in reader:
            labels = row


    # for loop each feature, selecting the two highest
    l = [mfcc]
    training, test, trainingLB, testLB = getFeatures(labels, l)
    
    # fit with classifier and predict
    X = np.array(training)
    Y = np.array(trainingLB)

    cla.fit(X,Y)
    predictions = cla.predict(np.array(test))
    errorRF = zero_one_loss(predictions, testLB)
    print arg[0] + ": " +  str(errorRF)
    
def getFeatures(labels, features):
    # select training and test sets
    TEidx = np.array(random.sample(range(0,1000), 100))
    
    training = []
    test = []
    
    trainingLB = []
    testLB = []



    # make total featuresDict
    featureDict = features[0];
    if len(features) > 1:
        for index in range(1, len(features)):
            for i in range(1000):
                featureDict[i].append(features[index][i])

    for i in range(1000):
        if i in TEidx:
            test.append(featureDict[i])
            testLB.append(labels[i])
        else:
            training.append(featureDict[i])
            trainingLB.append(labels[i])

    return training, test, trainingLB, testLB


if __name__ == "__main__":
    main(sys.argv[1:])
