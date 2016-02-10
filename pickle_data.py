import scipy.io as spio
import sys
import csv
import pickle

def main():
    filenameFV = 'mfcc_fv.csv'
    filenameLB = 'mfcc_lb.csv'
    
    dict = {}
    for x in range(1000):
        dict[x] = []

    with open('mfcc_fv.csv') as f:
        reader = csv.reader(f)  
        for row in reader:
            print len(row)
            i = 0
            for x in row:
                dict[i].append(float(x))
                i += 1
        
        pickle.dump(dict, open('mfcc_fv.p', 'wb'))

           
    
if __name__ == "__main__":
    main()
