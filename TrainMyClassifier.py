"""
Authors: Vineeth Chennapalli, Bhavesh Poddar, Nanda Kishore
"""

from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.decomposition import PCA

from sklearn.svm import SVC
from skrvm import RVC
from sklearn.gaussian_process.kernels import RBF

from time import time
import numpy as np

def TrainMyClassifier(XEstimate, YEstimate, XValidate, YValidate, Parameters=[]):
    """
    INPUTS:
          XEstimate - Feature vectors on which the model has to be trained
          YEstimate - Target estimates for the XEstimates
          XValidate - Feature vectors which are used to tune the hyperparameters
          Parameters - Hyerparameters of the respective algorithm

    OUTPUTS:
          classLabels - The estimated label values for each XValidate entry
          EstParams - The estimated parameters corresponding to each algorithm
    """

    threshold = 0.5 # metric to classify non-class entry
    Algorithm = Parameters[0]

    # extract true labels estimate
    Y_E = []
    Labels = YEstimate.tolist()
    for lis in Labels:
        if 1 in lis:
            Y_E.append(lis.index(1))
        else:
            Y_E.append(5)

    # extract true labels validate
    Y_V = []
    Labels = YValidate.tolist()
    for lis in Labels:
        if 1 in lis:
            Y_V.append(lis.index(1))
        else:
            Y_V.append(5)

    Y_V = np.array(Y_V)

    if Algorithm == "SVM":
        model = SVC(decision_function_shape='ovo', probability=True)
        clf = GridSearchCV(model, Parameters[1], cv=2)
        clf.fit(XEstimate, Y_E)
        proba = clf.predict_proba(XValidate)
        accuracy = clf.score(XValidate, Y_V)

        estParams = {
            'hyper': clf.best_params_,
            'model': clf.best_estimator_,
            'dual_coef': clf.best_estimator_.dual_coef_,
            'intercept': clf.best_estimator_.intercept_,
            'support_vectors': clf.best_estimator_.n_support_
        }

    elif Algorithm == "RVM":
        # perform PCA on data to reduce time
        Y_E = np.array(Y_E)
        #pca = PCA(n_components=8)
        #XEstimate = pca.fit_transform(XEstimate)
        #XValidate = pca.fit_transform(XValidate)

        threshold = 0.3

        posn, XEstimate_Fraction, Y_E_Fraction = {}, {}, {}

        Nc = 5

        for i in range(Nc):
            posn[i] = np.where(Y_E == i)

        for i in range(Nc):
            XEstimate_Fraction[i] = XEstimate[posn[i]]
            Y_E_Fraction[i] = Y_E[posn[i]]

        size = np.shape(XValidate)[0]
        predict_proba = np.zeros((size, 5))
        num_rvectors = []

        classifierObjs = []

        for i in range(Nc):
        	for j in range(i+1, Nc):
        		classifierObjs.append(RVC(n_iter=1, kernel = 'linear'))
        		classifierObjs[-1].fit(np.concatenate((XEstimate_Fraction[i], XEstimate_Fraction[j]), axis = 0), np.concatenate((Y_E_Fraction[i], Y_E_Fraction[j]), axis = 0))
        		sc_proba = classifierObjs[-1].predict_proba(XValidate)

        		predict_proba[:, i] += sc_proba[:, 0]
        		predict_proba[:, j] += sc_proba[:, 1]

        		num_rvectors.append(classifierObjs[-1].relevance_.shape[0])

        proba = predict_proba / 10

        count1 = count2 = 0

        #print(proba)
        
        for i in range(len(predict_proba)):
    		pos = predict_proba[i].argmax(axis=0)
    		if pos == Y_V[i]:
        		if predict_proba[i][pos] > 0.3:
        			count1 += 1
        		count2 += 1

        accuracy = float(count1)/len(predict_proba)
        
        #print("Inside accuracy < 0.3 is" + str(accuracy))

        accuracy = float(count2)/len(predict_proba)
        
        #print("Inside accuracy is" + str(accuracy))

        avg_rvectors = np.average(num_rvectors)

        print("Average number of relevance vectors: " + str(avg_rvectors))
        
        estParams = {
         'model': classifierObjs,
         'avg_rel_vectors': avg_rvectors
        }

    elif Algorithm == "GPR":
        # perform PCA on data to reduce time
        # pca = PCA(n_components=8)
        # XEstimate = pca.fit_transform(XEstimate[:1000,:])
        # XValidate = pca.fit_transform(XValidate)
        #print XEstimate.shape
        #print len(Y_E)

        kernal_rbf = 1*RBF(length_scale=1.0, length_scale_bounds=(1e-05, 100000.0))
        #clf = OneVsRestClassifier(GaussianProcessClassifier(kernel = kernal_rbf))
        clf = GaussianProcessClassifier(kernel = kernal_rbf, multi_class = 'one_vs_rest')
        print 'fitting'
        clf.fit(XEstimate, Y_E)
        print 'predicting'
        proba = clf.predict_proba(XValidate)
        print 'scoring'
        accuracy = clf.score(XValidate, Y_V)
        print 'accuracy'
        print accuracy
        estParams = {
            'model': clf
        }

    classLabels = np.full((len(YValidate), 6), -1, dtype=np.int)

    for i, p in enumerate(proba):
        idx = np.argmax(p)
        if p[idx] < threshold:
           classLabels[i][-1] = 1
        else:
            # print p
            classLabels[i][idx] = 1

    estParams['classLabels'] = classLabels
    estParams['accuracy'] = accuracy
    print("Accuracy is: " + str(accuracy))

    return classLabels, estParams

