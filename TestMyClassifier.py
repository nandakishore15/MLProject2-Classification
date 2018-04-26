"""
Author: Bhavesh Poddar, Nanda Kishore, Vineeth Chennapalli
"""

def TestMyClassifier(XTest, Parameters, EstParameters, YPred):
    """
    INPUT:
        XTest - Data on which the pre-trained model is tested
        Parameters - The parameter values corresponding to the algorithm
        EstParameters - Contains the parameters that were estimated after training.
            Includes the pre-trained model object as well.
    OUTPUT:
        YTest - Class labels for each sample in XTest
    """
    import numpy as np
    from sklearn.decomposition import PCA
    Algorithm = Parameters[0]
    threshold = 0.5
    clf = EstParameters[0]['model']


    if Algorithm == 'RVM':
        pca = PCA(n_components=8)
        XTest = pca.fit_transform(XTest)
        proba = clf.predict(XTest)

        num_samples = np.shape(XTest)[0]

        YTest = np.full((num_samples, 6), -1, dtype=np.int)

        for i, p in enumerate(proba):
            YTest[i][p] = 1

        return YTest

    #Predict probabilities
    proba = clf.predict_proba(XTest)

    num_samples = np.shape(XTest)[0]

    YTest = np.full((num_samples, 6), -1, dtype=np.int)

    for i, p in enumerate(proba):
        idx = np.argmax(p)
        if p[idx] < threshold:
            YTest[i][-1] = 1
        else:
            YTest[i][idx] = 1

    return YTest
