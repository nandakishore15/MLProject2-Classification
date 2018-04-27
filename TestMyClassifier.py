"""
Author: Bhavesh Poddar, Nanda Kishore, Vineeth Chennapalli
"""

def TestMyClassifier(XTest, Parameters, EstParameters):
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
    print(Algorithm)
    threshold = 0.5
    clf = EstParameters[0]['model']

    num_samples = np.shape(XTest)[0]

    if Algorithm == "RVM":
        threshold = 0.3
        predict_proba = np.zeros((num_samples, 5))
        objNum = 0
        for i in range(5):
            for j in range(i+1, 5):
                proba = clf[objNum].predict_proba(XTest)
                predict_proba[:, i] += proba[:, 0]
                predict_proba[:, j] += proba[:, 1]
                objNum += 1

        proba = predict_proba / 10

    else:
    #Predict probabilities
        proba = clf.predict_proba(XTest)

    YTest = np.full((num_samples, 6), -1, dtype=np.int)

    for i, p in enumerate(proba):
        idx = np.argmax(p)
        if p[idx] < threshold:
            YTest[i][-1] = 1
        else:
            YTest[i][idx] = 1

    return YTest
