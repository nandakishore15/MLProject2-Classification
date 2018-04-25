"""
Date: 04/21/2018
Author: Chuji Luo
Email: cjluo@ufl.edu
"""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def MyConfusionMatrix(Y, ClassNames, ClassLabels):
    """
    Function 'MyConfusionMatrix' computes confusion matrix and average accuracy
    INPUTS:
           Y: a numpy array, shape = [N, Nc+1], representing the estimated class labels
           ClassNames: a numpy array [C_1, ..., C_Nc, Non-Class], shape=[Nc+1]
           ClassLabels: a numpy array, shape = [N, Nc], representing the true class labels
    RETURNS:
           Confusion matrix: a numpy array, shape = [Nc+1, Nc+1]
           Average accuracy: a number
    """
    
    # the number of known classes
    Nc = np.shape(ClassLabels)[1]
    
    # extract true labels
    y_true = []
    Labels = ClassLabels.tolist()
    for lis in Labels:
        if 1 in lis:
            y_true.append(lis.index(1))
        else:
            y_true.append(Nc)
    # compute the number of samples in each class
    n = []
    for i in range(0, Nc+1):
        n.append(sum(j == i for j in y_true))
    n = np.asarray(n)
    y_true = ClassNames[y_true]
    
    # extract estimated labels (If the sample doesn't belong to any class, Y[i,Nc+1]=1. Otherwise, Y[i,Nc+1]=-1.)
    y_pred = np.argmax(Y, axis = 1)
    y_pred = ClassNames[y_pred]
    
    # compute confusion matrix
    Cf = confusion_matrix(y_true, y_pred, labels = ClassNames) ##unnormalized
    row_sums = Cf.sum(axis = 1)
    A = 1.0*Cf / row_sums[:, np.newaxis]
        
    # compute average accuracy
    a = A.diagonal()
    indices = ~np.isnan(a)
    accuracy = np.average(a[indices], weights = n[indices])
        
    # print the confusion matrix
    CF = pd.DataFrame(A, columns = ClassNames, index = ClassNames)
    print "Confusion Matrix:" 
    print CF
    print "'NaN' represents no samples in this row of class"
        
    # return
    return (A, accuracy)

