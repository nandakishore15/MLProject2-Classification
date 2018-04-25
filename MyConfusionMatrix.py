## Function 'MyConfusionMatrix' computes confusion matrix and average accuracy
## INPUTS:
##        Y: a numpy array, shape = [N, Nc+1], representing the estimated class labels
##        ClassNames: a numpy array [C_1, ..., C_Nc], shape=[Nc]
##        ClassLabels: a numpy array, shape = [N, Nc], representing the true class labels
## RETURNS:
##         Confusion matrix: a numpy array, shape = [Nc, Nc]
##         Average accuracy: a number
###############################################################################
## Date: 04/21/2018
## Author: Chuji Luo
## Email: cjluo@ufl.edu
def MyConfusionMatrix(Y, ClassNames, ClassLabels):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import confusion_matrix
    
    # the number of classes
    Nc = np.shape(ClassLabels)[1]
    
    # extract true labels
    y_true = []
    for lis in ClassLabels:
        for i, j in enumerate(lis):
            if j == 1:
                y_true.append(i)
                break
    y_true = ClassNames[y_true]
    
    # extract estimated labels
    Y = Y[:, :Nc]
    y_pred = np.argmax(Y, axis = 1)
    y_pred = ClassNames[y_pred]
    
    # compute confusion matrix
    Cf = confusion_matrix(y_true, y_pred, labels = ClassNames) ##unnormalized
    row_sums = Cf.sum(axis = 1)
    A = 1.0*Cf / row_sums[:, np.newaxis]
    
    # compute average accuracy
    accuracy = np.mean(A.diagonal())
    
    # print the confusion matrix
    CF = pd.DataFrame(A, columns = ClassNames, index = ClassNames)
    print "Confusion Matrix:" 
    print A
    
    # return
    return (A, accuracy)