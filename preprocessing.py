import numpy as np

import itertools
from kernel import *

def preprocessing(X, Y, percent=0.8):
    """
    Preprocessing the data.(expect as input numpy arrays)
        - Shuffle
        - Divide data and labels
        - centering
        - add dimension to the data
        - cut into a training dataset and a test dataset
    """

    # Shuffle phase
    np.random.RandomState(1)
    rand_ix = np.arange(X.shape[0])
    np.random.shuffle(rand_ix)
    n_training = int(percent  * Y.shape[0])
    train_ix = rand_ix[:n_training]
    test_ix = rand_ix[n_training:]

    ## only for numerical data (does not work with sequences)
    # centering data
    # X = X-np.mean(X, axis=0)

    # Scaling data
    # X = scale( X, axis=0, with_mean=True, with_std=True, copy=True )

    # Add one dimension to your data points in order to account for
    # the offset if your data is not centered.
    # X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)

    # Compute the training and the test set
    X_train = X[train_ix]
    Y_train = (Y[train_ix] - 0.5) * 2
    X_test = X[test_ix]
    Y_test = (Y[test_ix] - 0.5) * 2

    return X_train, Y_train, X_test, Y_test

# params is a dictionnary containing the parameters on which doing the grid_search and the desired values:
# eg: grid = {'lamb':[0.1,0.01,0.005], 'k':[1,2,3]}
def grid_search(X, Y, grid, kernel = 'k_gram'):
    X_train, Y_train, X_test, Y_test = preprocessing(X, Y)
    # baseline parameters
    keys, values = zip(*grid.items()) 
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    best_param = dict(zip(grid.keys(), [None]*len(grid.keys())))
    best_acc_test = 0
    best_acc_train = 0
    param = {'X_train':X_train, 'X_test':X_test, 'Y_train':Y_train, 'Y_test':Y_test, 'kernel':kernel}
    
    for e in experiments:
        for p in e.keys():    
            param[p] = e[p]
        print(e)
        acc_test, acc_train = solve_svm_kernel(**param)
        print('accuracy for train : {}'.format(acc_train))
        print('accuracy for test : {}'.format(acc_test))
        if acc_test > best_acc_test:
            best_acc_test = acc_test
            best_acc_train = acc_train
            for p in grid.keys():
                best_param[p] = e[p]
    print('best parameters:', best_param, 'for a test accuracy of ', best_acc_test)
    return best_acc_test, best_acc_train, best_param