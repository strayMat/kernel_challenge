import numpy as np

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