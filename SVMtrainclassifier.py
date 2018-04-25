from sklearn import svm
from sklearn.model_selection import GridSearchCV


def TrainMyClassifier(XEstimate, YEstimate, XValidate, YValidate, Parameters=[]):
    # extract true labels estimate
    threshold = 0.5
    Y_E = []
    Labels = YEstimate.tolist()
    for lis in Labels:
        if 1 in lis:
            Y_E.append(lis.index(1))
        else:
            Y_E.append(Nc)

    # extract true labels validate
    Y_V = []
    Labels = YValidate.tolist()
    for lis in Labels:
        if 1 in lis:
            Y_V.append(lis.index(1))
        else:
            Y_V.append(Nc)

    model = svm.SVC(decision_function_shape='ovo', probability=True)
    clf = GridSearchCV(model, Parameters, cv=2)
    clf.fit(XEstimate, Y_E)
    proba = clf.predict_proba(XValidate)
    classlabels = np.full((len(YValidate), 6), -1, dtype=np.int)
    for i, p in enumerate(proba):
        idx = np.argmax(p)
        if p[idx] < threshold:
            classlabels[i][-1] = 1
        else:
            classlabels[i][idx] = 1
    est_params = {
        'score': clf.score(XValidate, Y_V),
        'hyper': clf.best_params_,
        'classlabels': classlabels,
        'model': clf.best_estimator_,
        'dual_coef': clf.best_estimator_.dual_coef_,
        'intercept': clf.best_estimator_.intercept_,
        'support_vectors': clf.best_estimator_.n_support_
    }
    return classlabels, est_params
