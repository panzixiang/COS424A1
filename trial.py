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
import itertools
import operator



def main(arg):
    global mfcc, chroma, energy, brightness, hcdf;
    
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
    featureCombos = []
    for i in range(5):
        featureCombos += list(itertools.combinations([mfcc, chroma, energy, brightness, hcdf], i+1));
    outcomes = {}

    for l in featureCombos:
        featStr, error = runTrial(cla, arg[0], l, labels)
        outcomes[featStr] = error

    outcomesSorted = sorted(outcomes.items(), key=operator.itemgetter(1))
    
    print "-------------"
    print "top five:"
    index = 0
    for o in outcomesSorted:
        if index == 5:
            break
        formatted = "%s" % (o,)
        print formatted # + ", error=" + str(outcomesSorted(o))
        index += 1


   
def runTrial(cla, claName, featList, labels): 
    training, test, trainingLB, testLB = getFeatures(featList, labels)

     # fit with classifier and predict
    X = np.array(training)
    Y = np.array(trainingLB)

    cla.fit(X,Y)
    predictions = cla.predict(np.array(test))
    errorRF = zero_one_loss(predictions, testLB)
    print claName + ", feats: " + printFeatures(featList) + ": " 
    print "error= " + str(errorRF)
    return printFeatures(featList), errorRF


def printFeatures(featList):
    featStr = ''
    for f in featList:
        if f is mfcc:
            featStr += 'mfcc'
        elif f is chroma:
            featStr += 'chroma'
        elif f is brightness:
            featStr += 'brightness'
        elif f is energy:
            featStr += 'energy'
        elif f is hcdf:
            featStr += 'hcdf'
        featStr += ' '
    return featStr

def getFeatures(features, labels):
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
                featureDict[i] += features[index][i]

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
