"""
Date: 04/21/2018
Author: Chuji Luo
Email: cjluo@ufl.edu
"""
import numpy as np
from compiler.ast import flatten
from MyConfusionMatrix import MyConfusionMatrix
from TrainMyClassifier import TrainMyClassifier

def MyCrossValidate(XTrain, ClassLabels, Nf, Parameters):

    """
    Function 'MyCrossValidate' performs the cross validation on training dataset
    INPUTS:
          XTrain: a numpy array, shape = [N, D]
          ClassLabels: a numpy array, shape = [N, Nc], representing the true class labels
          Parameters: used as a INPUT of 'TrainMyClassifier' function
          Nf: the number of folds
    RETURNS:
          Ytrain: a numpy array, representing the estimated class labels for each validation sample
          EstParameters: a numpy array of estimated parameters for each Vn (Validation Set)
          EstConfMatrices: a numpy array of confusion matrix for each Vn (Validation Set)
          ConfMatrix: a numpy array, shape = [Nc+1, Nc+1], the overall confusion matrix
    """
    
    # the number of samples in 'XTrain'
    N = np.shape(ClassLabels)[0]
    Nc = np.shape(ClassLabels)[1]
    
    
    # step 1: randomly partition 'XTrain' into Nf pairs of En and Vn
    new_X = np.zeros_like(XTrain)
    new_Y = np.zeros_like(ClassLabels)
    idx = np.random.permutation(len(XTrain))
    for i, j in enumerate(idx):
        new_X[i], new_Y[i] = XTrain[j], ClassLabels[j]
        
    k = int(N / Nf)
    V_X = [] ## validation set in XTrain
    V_Y = [] ## validation set in ClassLabels
    E_X = [] ## estimation set in XTrain
    E_Y = [] ## estimation set in ClassLabels
    for i in range(0, Nf):
        V_X.append(new_X[range(i*k, (i+1)*k)])
        E_X.append(np.delete(new_X, range(i*k, (i+1)*k), axis = 0))
        V_Y.append(new_Y[range(i*k, (i+1)*k)])
        E_Y.append(np.delete(new_Y, range(i*k, (i+1)*k), axis = 0))
    V_X, V_Y, E_X, E_Y = np.asarray(V_X), np.asarray(V_Y), np.asarray(E_X), np.asarray(E_Y)

    
    # step 2: estimate parameters and hyper-parameters using En and Vn
    Ytrain = []
    EstParameters = []
    for i in range(0, Nf):
        Xe, Xv, Ce, Cv = E_X[i], V_X[i], E_Y[i], V_Y[i]
        ## Call function 'TrainMyClassifier'. Here I assume all the inputs except 'Parameters' are numpy arrays.
        [y, par] = TrainMyClassifier(XEstimate = Xe, XValidate = Xv, ClassLabelsEstimate = Ce, ClassLabelsValidate = Cv, Parameters = Parameters)
        Ytrain.append(y)
        EstParameters.append(par)
    Ytrain, EstParameters = np.asarray(Ytrain), np.asarray(EstParameters)
    
    
    # step 3: produce a confusion matrix Cn, for each Vn
    EstConfMatrices = []
    ClassNames = range(1, Nc+1)
    ClassNames = ClassNames.append("NonClass")
    ClassNames = np.asarray(ClassNames)
    for i in range(0, Nf):
        cur_y = Ytrain[i]
        cur_label = V_Y[i]
        [A, a] = MyConfusionMatrix(Y = cur_y, ClassNames = ClassNames, ClassLabels = cur_label)
        EstConfMatrices.append(A)
    EstConfMatrices = np.asarray(EstConfMatrices)
    
    
    # step 4: produce a confusion matrix for all of XTrain using all the class labels
    ConfMatrix = []
    y_all = flatten(np.array(Ytrain).tolist())
    [ConfMatrix, a] = MyConfusionMatrix(Y = y_all, ClassNames = ClassNames, ClassLabels = ClassLabels)

    
    # return
    return (Ytrain, EstParameters, EstConfMatrices, ConfMatrix);