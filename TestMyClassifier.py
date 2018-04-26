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
    Algorithm = Parameters[0]
    threshold = 0.5
    clf = EstParameters[0]['model']

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
