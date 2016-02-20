import scipy.io as spio
import sys
import csv
import pickle
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics import zero_one_loss
from sklearn.metrics import confusion_matrix
import random
import matplotlib.pyplot as plt

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
            testLB.append(int(labels[i]))
        else:
            training.append(featureDict[i])
            trainingLB.append(int(labels[i]))
        
    # fit with classifier and predict
    X = np.array(training)
    Y = np.array(trainingLB)


    cla = svm.SVC(kernel='linear')
    cla.fit(X,Y)
    predictions = cla.predict(np.array(test))
    errorRF = zero_one_loss(predictions, testLB)
    print errorRF

    print testLB
    print predictions
    # Compute confusion matrix
    cm = confusion_matrix(testLB, predictions)
    np.set_printoptions(precision=2)
    print('Confusion matrix, without normalization')
    print(cm)
    plt.figure()
    plot_confusion_matrix(cm)
    

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, [1,2,3,4,5,6,7,8,9,10], rotation=45)
    plt.yticks(tick_marks, [1,2,3,4,5,6,7,8,9,10])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



if __name__ == "__main__":
    main()
