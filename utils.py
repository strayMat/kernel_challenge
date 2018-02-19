###############################################
# Gradient Descent Methods And SVM optimization
# Erwan Bourceret
###############################################

import numpy as np
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import linalg as LA
from matplotlib import rc
from sklearn.preprocessing import scale
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)


# Interior Points Method
# phi(x,t,Q,p,A,b), grad(x,t,Q,p,A,b) and hess(x,t,Q,p,A,b)

def phi(x,t,Q,p,A,b):
    """
    Compute the fonction value
    """
    x = np.array(x, dtype=np.float)
    Q = np.matrix(Q, dtype=np.float)
    p = np.array(p, dtype=np.float)
    A = np.matrix(A, dtype=np.float)
    b = np.array(b, dtype=np.float)

    phit = (0.5*x.dot(Q).dot(x) + p.dot(x))

    phit = t*phit

    tmp = b - A.dot(x)
    tmp = np.log(tmp)

    phit = np.array(phit - np.sum(tmp))[0][0]
    return phit

def grad(x,t,Q,p,A,b):
    """
    Compute the gradient of phit
    """
    x = np.array(x, dtype=np.float)
    Q = np.matrix(Q, dtype=np.float)
    p = np.array(p, dtype=np.float)
    A = np.matrix(A, dtype=np.float)
    b = np.array(b, dtype=np.float)

    gradt = Q.dot(x) + p
    gradt = t * gradt

    tmp = b - A.dot(x)
    tmp = 1./tmp
    gradt = np.array(gradt + tmp.dot(A))[0]
    return gradt


def hess(x,t,Q,p,A,b):
    """
    Compute the Hessian of phit
    """
    x = np.array(x, dtype=np.float)
    Q = np.matrix(Q, dtype=np.float)
    p = np.array(p, dtype=np.float)
    A = np.matrix(A, dtype=np.float)
    b = np.array(b, dtype=np.float)

    hesst = t*Q
    tmp = b - A.dot(x)
    tmp = 1./np.square(tmp)
    tmp = np.diag(np.array(tmp)[0])
    tmp = (A.T).dot(tmp).dot(A)
    hesst = hesst + tmp

    return hesst

############################
###### 2. Newton Step ######
############################

def dampedNewtonStep(x,f,g,h):
    """
    Compute the damped Newton step at point x
    Args:
        - x : point where the Newton step will be computed
        - f : the value function
        - g : the gradient function
        - h : the hessian function
    Ouput:
        - x_new : the damped Newton step at point x
        - lamdat2/2 : the estimated gap before the minimum
    """
    phit = f(x)
    gradt = np.array(g(x), dtype=np.float)
    hesst = np.matrix(h(x), dtype=np.float)
    hesst_inv = np.linalg.inv(hesst)

    lambdat2 = np.array(gradt.dot(hesst_inv).dot(gradt))[0][0]
    coef = (1.+np.sqrt(lambdat2))
    coef = 1./coef
    x_new = np.array(x - coef * hesst_inv.dot(gradt))[0]

    return x_new, lambdat2/2


def dampedNewton(x0,f,g,h,tol,Tmax=5):
    """
    Implement the damped Newton algorithm
    Args:
        - x0 : initial point
        - f : the value function to minimize
        - g : the gradient function
        - h : the hessian function
        - tol : the threshold (smaller than 0.3819660112501051)
        - Tmax : maximum number of iteration
    Ouput:
        - xstar : the point tol-minimizing f
        - xhist : the history of damped step
    """
    try:
        assert tol < 0.3819660112501051
    except AssertionError:
        print("The threshold in dampedNewton must be smaller ")
        exit(1)
    xstar, gap = dampedNewtonStep(x0,f,g,h)
    xhist = [x0, xstar]
    phi_w_hist = [f(x0), f(xstar)]
    it = 1
    while(gap>tol and it<Tmax):
        it +=1
        xstar, gap = dampedNewtonStep(xstar,f,g,h)
        phi_w_hist.append(f(xstar))
        xhist.append(xstar)
    xhist = np.array(xhist)

    return xstar, xhist, phi_w_hist


def newtonStep(x,f,g,h):
    """
    Compute the Newton step at point x
    Args:
        - x : point where the Newton step will be computed
        - f : the value function
        - g : the gradient function
        - h : the hessian function
    Ouput:
        - x_new : the Newton step at point x
        - lamdat/2 : the estimated gap before the minimum
    """
    phit = f(x)
    gradt = np.array(g(x), dtype=np.float)
    hesst = np.matrix(h(x), dtype=np.float)
    try:
        hesst_inv = np.linalg.inv(hesst)
    except LinAlgError("Singular matrix"):
        exit(1)


    lambdat = np.array(gradt.dot(hesst).dot(gradt))[0][0]
    x_new = np.array(x - hesst_inv.dot(gradt))[0]

    return x_new, lambdat/2.



def newtonLS(x0,f,g,h,tol, alpha=0.2, beta=0.9, Tmax = 5):
    """
    Implement the Newton algorithm with backtracking line-search
    Args:
        - x0 : initial point
        - f : the value function to minimize
        - g : the gradient function
        - h : the hessian function
        - tol : the threshold
        - alpha, beta : parameter for line-search
        - Tmax : maximum number of iteration
    Ouput:
        - xstar : the point tol-minimizing f
        - xhist : the history of damped step
    """
    xstar, gap = newtonStep(x0,f,g,h)
    xhist = [x0, xstar]
    it = 0
    while(gap>tol and it<Tmax):
        it+=1
        xtmp, gap = newtonStep(xstar,f,g,h)
        deltax = xtmp - xstar
        gradt = np.array(g(xstar), dtype=np.float)
        t=1
        # Backtracking line search
        while(f(xstar + t*deltax) > f(xstar) + alpha * t * gradt.dot(deltax)):
            t= beta*t
        xstar = xstar + t*deltax
        xhist.append(xstar)

    return xstar, xhist

###############################################
###### 3. Support Vector Machine Problem ######
###############################################

def transform_svm_primal(tau,X,y):
    """
    Transform the primal Support Vector Machine (SVM)
    problem into a quadratic problem
    Args:
        - tau : regularization parameter
        - X : data set
        - Y : Target labels
    Ouput:
        - Q : semi-definite matrix (quadratic parameter)
        - P : vector parameter in the minimization part
        - A : matrix constraint
        - b : vector constraint
    """
    X = np.matrix(X, dtype=np.float)
    y = np.array(y, dtype=np.float)

    # number of data
    n = y.size

    # Verify the shape of the data
    try:
        assert X.shape[0] == n
    except AssertionError:
        try:
            assert X.shape[1] == n
            X = np.transpose(X)
        except AssertionError:
            print("X and Y must have the same length")
            exit(1)
    d = X.shape[1]

    Q = np.append(np.repeat(1.,d), np.repeat(0.,n))
    P = (1. - Q)/tau/n
    Q = np.diag(Q)

    # condition : yx.w + z > 1
    A = np.concatenate((-np.transpose(np.multiply(np.transpose(X), y)), -np.eye(n)), axis=1)

    # condition : z>0 condition
    A_tmp = np.concatenate((np.zeros((n,d)), -np.eye(n)), axis=1)

    A = np.concatenate((A,A_tmp), axis=0)
    b = np.append(np.repeat(-1., n), np.repeat(0., n))

    return Q, P, A, b


def transform_svm_dual(tau,X,y):
    """
    Transform the dual Support Vector Machine (SVM)
    problem into a quadratic problem
    Args:
        - tau : regularization parameter
        - X : data set
        - Y : Target labels
    Ouput:
        - Q : semi-definite matrix (quadratic parameter)
        - P : vector parameter in the minimization part
        - A : matrix constraint
        - b : vector constraint
    """
    X = np.matrix(X, dtype=np.float)
    y = np.array(y)

    # Data size
    n = y.size

    # Verify the shape of the data
    try:
        assert X.shape[0] == n
    except AssertionError:
        try:
            assert X.shape[1] == n
            X = np.transpose(X)
        except AssertionError:
            print("X and Y must have the same length")
            exit(1)

    # Compute the Kernel Matrix with labels ponderation
    Q = np.diag(y).dot(X).dot(np.transpose(X)).dot(np.diag(y))

    P = -np.repeat(1.,n)

    # condition 0 < lambda < 1/n/tau
    A = np.concatenate((np.eye(n), -np.eye(n)), axis=0)
    b = np.append(np.repeat(1., n)/tau/n, np.repeat(0, n))


    return Q, P, A, b



def barr_method(Q,p,A,b,x_0,mu,tol):
    """
    Solve the Quadratic problem using damped Newton method
    Args:
        - Q : semi-definite matrix (quadratic parameter)
        - P : vector parameter in the minimization part
        - A : matrix constraint
        - b : vector constraint
        - x_0 : inital state
        - mu :  increment of the barrier parameter
        - tol : the threshold
    Ouput:
        - x_sol : the argument minimizing the quadratic problem
        - x_hist : the history of step
    """
    t = mu
    x_sol = x_0
    x_hist = np.matrix(x_sol)
    phi_w_hist = []
    it=0
    while(mu/t > tol):
        it+=1
        # if(it%100 == 0):
        print("we want {} to be less than {}".format(mu/t, tol))
        f = lambda x: phi(x,t,Q,p,A,b) ;
        g = lambda x: grad(x,t,Q,p,A,b) ;
        h = lambda x: hess(x,t,Q,p,A,b) ;
        x_sol, xhist_tmp, ph_w_tmp = dampedNewton(x_sol,f,g,h,tol)
        x_hist = np.concatenate((x_hist, xhist_tmp), axis = 0)
        phi_w_hist.append(ph_w_tmp)
        t = mu*t

    #phi_w_hist.append(f(x_sol))
    return x_sol, x_hist, phi_w_hist


def preprocessing(X, Y, percent=0.8):
    """
    Preprocessing the data.
        - Shuffle
        - Divide data and labels
        - centering
        - add dimension to the data
        - cut into a training dataset and a test dataset
    """

    # Shuffle phase
    np.random.RandomState(1)
    tmp = np.concatenate((X, np.matrix(Y)), axis=1)
    np.random.shuffle(tmp)
    X = tmp[:, :X.shape[1]]

    # centering data
    # X = X-np.mean(X, axis=0)

    # Scaling data
    # X = scale( X, axis=0, with_mean=True, with_std=True, copy=True )

    # Add one dimension to your data points in order to account for
    # the offset if your data is not centered.
    # X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)

    # Compute the training and the test set
    n_training = int(percent  * Y.size)
    X_train = X[:n_training]
    Y_train = Y[:n_training]
    X_test = X[n_training:]
    Y_test = Y[n_training:]

    return X_train, Y_train, X_test, Y_test


def SVM_vector(X_train, Y_train, tau, mu, dual=False, tol=0.01):
    """
    Compute the SVM vector corresponding to the normal of the hyperplan
    which separates the data
    """
    w_0 = np.append(np.repeat(0., X_train.shape[1]), np.repeat(2., X_train.shape[0]))

    if(dual):
        w_0 = np.repeat(1., X_train.shape[0])/2./tau/float(X_train.shape[0])
        Q, P, A, b = transform_svm_dual(tau,X_train,Y_train)
    else:
        w_0 = np.append(np.repeat(0., X_train.shape[1]), np.repeat(2., X_train.shape[0]))
        Q, P, A, b = transform_svm_primal(tau,X_train,Y_train)



    w, w_hist, phi_w_hist = barr_method(Q,P,A,b,w_0,mu,tol)

    w = w[:X_train.shape[1]]

    return w, w_hist, phi_w_hist



def predict(w, X_test, Y_test):
    """
    Given the normal vector w, it predicts the label of X_test
    and compare them to Y_test.
    """
    w = np.array(w, dtype=np.float)
    Y_predicted = []
    Y_predicted = np.array(np.sign(X_test.dot(w)))[0]
    accuracy = float(np.sum(Y_test-Y_predicted == 0))/float(Y_test.size)*100.
    return Y_predicted, accuracy







