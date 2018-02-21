import numpy as np
from cvxopt import matrix, solvers


def transform_svm_primal(K,y,lamb):
    """
    Transform the primal Support Vector Machine (SVM)
    problem into a quadratic problem
    Args:
        - K : Kernel of x
        - Y : Target labels
        - lamb : regularization parameter
    Ouput:
        - Q : semi-definite matrix (quadratic parameter)
        |K   0 |
        |0     0 |

        - P : vector parameter in the minimization part
        |0,...,0 , 1, ..., 1|^t / lamb

        - A : matrix constraint
        |-yK   -I |
        |0     -I |

        - b : vector constraint
        |-1,...,-1 , 0, ..., 0|^t

    """
    K = np.matrix(K, dtype=np.float)
    y = np.array(y, dtype=np.float)

    # number of data
    n = y.size

    # Verify the shape of the data
    try:
        assert K.shape[0] == n
    except AssertionError:
        print("K and Y must have the same length")
        exit(1)

    A1 = np.concatenate((-np.transpose(np.multiply(K, y)), -np.eye(n)), axis=1)
    A2 = np.concatenate((np.zeros((n,n)), -np.eye(n)), axis=1)
    A = np.concatenate((A1, A2), axis=0)

    Q1 = np.concatenate((K, np.zeros((n,n))), axis=1)
    Q2 = np.zeros((n, 2*n))
    Q = lamb * np.concatenate((Q1, Q2), axis=0)

    p = np.concatenate((np.zeros(n), np.ones(n)), axis=0) / n

    b = np.concatenate((-np.ones(n), np.zeros(n)), axis=0)


    return Q, p, A, b


def solve_svm(K, Y_train, lamb=0.1, kktreg=1e-9):
    Q, p, A, b = transform_svm_primal(K, Y_train, lamb)
    n = K.shape[0]

    Q = matrix(2*Q)
    p = matrix(p)
    G = matrix(A)
    h = matrix(b)
    sol=solvers.qp(Q, p, G, h, kktsolver='ldl', options={'kktreg':1e-9}) # A, b)
    return sol['x']

def l2distance(X,Z=None):
    # function D=l2distance(X,Z)
    #
    # Computes the Euclidean distance matrix.
    # Syntax:
    # D=l2distance(X,Z)
    # Input:
    # X: nxd data matrix with n vectors (columns) of dimensionality d
    # Z: mxd data matrix with m vectors (columns) of dimensionality d
    #
    # Output:
    # Matrix D of size nxm
    # D(i,j) is the Euclidean distance of X(:,i) and Z(:,j)
    #
    # call with only one input:
    # l2distance(X)=l2distance(X,X)
    #
    # Remember that the Euclidean distance can be expressed as: x^2-2x*z+z^2

    if Z is None:
        Z = X
    n,d = X.shape
    m = Z.shape[0]
    S = np.dot(np.diag(np.dot(X,X.T)).reshape(n,1),np.ones((1, m),dtype=int))
    R = np.dot(np.ones((n, 1),dtype=int),np.diag(np.dot(Z,Z.T)).reshape(1,m))
    G = np.dot(X,Z.T)
    D2 = S-2*G+R
    D = np.sqrt(D2)
    return D


def computeK(kerneltype, X, Z, kpar=0):
    """
    function K = computeK(kernel_type, X, Z)
    computes a matrix K such that Kij=g(x,z);
    for three different function linear, rbf or polynomial.

    Input:
    kerneltype: either 'linear','polynomial','rbf'
    X: n input vectors of dimension d (dxn);
    Z: m input vectors of dimension d (dxn);
    kpar: kernel parameter (inverse kernel width gamma in case of RBF, degree in case of polynomial)

    OUTPUT:
    K : nxm kernel matrix
    """

    assert kerneltype in ["linear","polynomial","poly","rbf"], "Kernel type %s not known." % kerneltype
    assert X.shape[1] == Z.shape[1], "Input dimensions do not match"

    # TODO 2
    X = np.array(X)
    Z = np.array(Z)
    if kerneltype == "linear":
        K = np.dot(X,Z.T)
    elif kerneltype == "polynomial" or kerneltype == "poly":
        K = np.power(np.dot(X,Z.T)+1,kpar)
    else:
         K = np.exp(-kpar*np.square(l2distance(X,Z)))

    return K
