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
from sklearn.cross_validation import KFold
from random import shuffle

# confusion matrix plotting code from: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#example-model-selection-plot-confusion-matrix-py
def main():
    
    filenameLB = 'mfcc_lb.csv'
    featureDict = pickle.load(open('mfcc_fv.p', 'rb'))
    
    with open('mfcc_lb.csv') as f:
        reader = csv.reader(f)
        for row in reader:
            labels = row

    # select training and test sets
    '''
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

    '''

    feats_shuf = []
    labels_shuf = []
    index_shuf = range(len(labels))
    shuffle(index_shuf)
    for i in index_shuf:
        feats_shuf.append(featureDict[i])
        labels_shuf.append(labels[i])


    X = np.array(feats_shuf)
    Y = np.array(labels_shuf)

    kf = KFold(1000, n_folds=10)
    cla = svm.SVC(kernel='linear')

    cm_all = np.zeros((10,10))
    for train, test in kf:
        X_train, X_test, y_train, y_test = X[train], X[test], Y[train], Y[test]
        cla.fit(X_train, y_train)
        predictions = cla.predict(X_test)
        

        # Compute confusion matrix
        cm = confusion_matrix(y_test, predictions)
        np.set_printoptions(precision=2)
        print(cm) 
        np.add(cm_all, cm)
    

    plt.figure()
    plot_confusion_matrix(cm_all)

    plt.show()
    

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
