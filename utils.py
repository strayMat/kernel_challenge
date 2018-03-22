import numpy as np
from cvxopt import matrix, solvers
from kernel_functions import *
import itertools


# Core functions for solving svm and preprocessing
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
    n_training = int(percent * Y.shape[0])
    train_ix = rand_ix[:n_training]
    test_ix = rand_ix[n_training:]

    # only for numerical data (does not work with sequences)
    # centering data
    # X = X-np.mean(X, axis=0)
    # Scaling data
    # X = scale( X, axis=0, with_mean=True, with_std=True, copy=True )

    # Compute the training and the test set
    # Compute the training and the test set
    X_train = X[train_ix]
    Y_train = Y[train_ix] + 0.0
    X_test = X[test_ix]
    Y_test = Y[test_ix] + 0.0

    return X_train, Y_train, X_test, Y_test


def kernelize(X_train, X_test, kernel='linear', Kernels=None):
    """Kernalization from a feature vector
    """
    if kernel == 'linear':
        K = linear_kernel(X_train, X_train)
        K_test = linear_kernel(X_test, X_train)
    if kernel == 'gaussian':
        K = gaussian_kernel(X_train, X_train, gamma)
        K_test = gaussian_kernel(X_test, X_train, gamma)
    if kernel == 'custom':
        K = Kernels[0]
        K_test = Kernels[1]
    return K, K_test


def fit(K, y, lamb=0.1, verbose=False):
    """Solving the dual with cvxopt library
    """
    # map to [-1,1]
    y = (y - 0.5) * 2
    # We solve the Dual
    NUM = K.shape[0]
    P = matrix(2 * K * y.reshape((-1, 1)).dot(y.reshape((1, -1))))
    q = matrix(-np.ones((NUM, 1)))
    G = matrix(np.concatenate((-np.eye(NUM), np.eye(NUM)), axis=0))
    h = matrix(np.concatenate((np.zeros(NUM), lamb * np.ones(NUM)), axis=0))
    A = matrix(y.reshape(1, -1))
    b = matrix(np.zeros(1))
    solvers.options['show_progress'] = verbose
    sol = solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x']) * y[:, None]
    bias = np.mean(y - np.dot(K, alphas))
    return alphas, bias


def predict(alphas, bias, K_test):
    mat = np.dot(K_test, alphas)
    mat = mat + bias > 0.
    return mat.reshape(-1)


# Testing functions

def testing_lambda(X_train,
                   Y_train,
                   X_test,
                   Y_test,
                   lamb=0.1,
                   kernel='linear',
                   gamma=1000,
                   Kernels=None):
    K, K_test = kernelize(X_train, X_test, kernel='linear', Kernels=Kernels)
    alphas, bias = fit(K, Y_train, lamb=lamb)

    Y_pred = predict(alphas, bias, K_test)
    acc_test = np.sum(Y_pred == Y_test) / Y_test.shape[0]

    Y_pred_train = predict(alphas, bias, K)
    acc_train = np.sum(Y_pred_train == Y_train) / X_train.shape[0]

    if np.alltrue(Y_pred == 1):
        print("test Toute les valeurs sont TRUE")

    if np.alltrue(Y_pred == -1):
        print("Toute les valeurs sont FALSE")

    return acc_train, acc_test


def test_kernel(X_train,
                Y_train,
                X_test,
                Y_test,
                kernel='linear',
                gamma=1000,
                kernel_feat=count_kuplet_k,
                lamb=0.1,
                **kwargs):
    X_feat_train = []
    X_feat_test = []
    for xx in tqdm(X_train):
        X_feat_train.append(kernel_feat(xx, kwargs))
    for xx in tqdm(X_test):
        X_feat_test.append(kernel_feat(xx, kwargs))

    X_feat_train = np.array(X_feat_train)
    X_feat_test = np.array(X_feat_test)

    K, K_test = kernelize(X_feat_train, X_feat_test,
                          kernel=kernel)
    alphas, bias = fit(K, Y_train, lamb=lamb)

    Y_pred = predict(alphas, bias, K_test)
    acc_test = np.sum(Y_pred == Y_test) / Y_test.shape[0]
    Y_pred_train = predict(alphas, bias, K)
    acc_train = np.sum(Y_pred_train == Y_train) / X_train.shape[0]

    if np.alltrue(Y_pred == 1):
        print("test Toute les valeurs sont TRUE")
    if np.alltrue(Y_pred == -1):
        print("Toute les valeurs sont FALSE")

    return acc_train, acc_test


# Params is a dictionnary containing the parameters on which doing the grid_search and the desired values:
# eg: grid = {'lamb':[0.1,0.01,0.005], 'k':[1,2,3]}
def grid_search(X, Y, grid, kernel_feat=count_kuplet_k):
    X_train, Y_train, X_test, Y_test = preprocessing(X, Y)
    # baseline parameters
    keys, values = zip(*grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    best_param = dict(zip(grid.keys(), [None] * len(grid.keys())))
    best_acc_test = 0
    best_acc_train = 0

    for e in experiments:
        acc_test, acc_train = test_kernel(**param)
        print('accuracy for train : {}'.format(acc_train))
        print('accuracy for test : {}'.format(acc_test))
        if acc_test > best_acc_test:
            best_acc_test = acc_test
            best_acc_train = acc_train
            for p in grid.keys():
                best_param[p] = e[p]
    print('best parameters:', best_param,
          'for a test accuracy of ', best_acc_test)
    return best_acc_test, best_acc_train, best_param
