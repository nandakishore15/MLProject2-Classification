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

    #TODO Check this with others

    #Y_E = np.array(Y_E)
    #validEntries = np.where(Y_E != -1)

    #XEstimate = XEstimate[validEntries]
    #Y_E = Y_E[validEntries]

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
        pca = PCA(n_components=8)
        XEstimate = pca.fit_transform(XEstimate)
        XValidate = pca.fit_transform(XValidate)

        model = RVC(n_iter=1, kernel='linear', verbose=True)
        #clf = OneVsRestClassifier(model)
        model.fit(XEstimate, Y_E)
        proba = model.predict(XValidate)
        #proba = clf.predict()
        accuracy = model.score(XValidate, Y_V)

        classLabels = np.full((len(YValidate), 6), -1, dtype=np.int)

        for i, p in enumerate(proba):
         classLabels[i][p] = 1


        estParams = {
         'model': model,
        }

        estParams['classLabels'] = classLabels
        estParams['accuracy'] = accuracy
        print("Accuracy is: " + str(accuracy))
        return classLabels, estParams

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

