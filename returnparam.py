def get_parameters(Algorithm='SVM'):
    if Algorithm == 'SVM':
        params = ['SVM', [
                    {
                       'kernel': ['linear'],
                       'gamma': ['auto'],
                       'C': [1, 10, 100, 1000]
                    }
            ]]
        print(repr(params))
        return params

    elif Algorithm == 'RVM':
        params = ['RVM', [
            {
                'kernel':'linear',
                'tolerance': '1e-3',
                'Initial alpha': '1e-6',
                'threshold_alpha': '1e9',
                'Initial beta': '1.e-6',
                'beta_fixed': 'false',
                'bias_used': 'true'
            }
        ]]
        print(repr(params))
        return params

    elif Algorithm == 'GPR':
        params = ['GPR', [{ 'length_scale': 1.0, 'length_scale_bounds': (1e-05, 100000.0)}]]
        print(repr(params))
        return params
    else:
        print "Wrong Choice. Choose between SVM, RVM or GPR"
