def get_parameters(Algorithm='SVM'):
    if Algorithm == 'SVM':
        return [
            {
                'kernel': ['rbf'],
                'gamma': [1e-3, 1e-4],
                'C': [1, 10, 100, 1000]
            },
            {
                'kernel': ['linear'],
                'gamma': ['auto'],
                'C': [1, 10, 100, 1000]
            }
        ]

    elif Algorithm == 'RVM':
        return {}

    elif Algorithm == 'GPR':
        return {}

    else:
        print "Wrong Choice. Choose between SVM, RVM or GPR"
