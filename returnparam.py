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
        return [{ 'length_scale': 1.0, 'length_scale_bounds': (1e-05, 100000.0)}]

    else:
        print "Wrong Choice. Choose between SVM, RVM or GPR"
