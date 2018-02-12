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
    Y = np.array(np.transpose(tmp[:, -1]))[0]

    # Select 2 species
    # X = X[Y>0]
    # Y = Y[Y>0]

    # different species : {-1, 1}
    Y = (Y - 0.5)*2.

    # centering data
    # X = X-np.mean(X, axis=0)

    # Scaling data
    # X = scale( X, axis=0, with_mean=True, with_std=True, copy=True )

    # Add one dimension to your data points in order to account for
    # the offset if your data is not centered.
    X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)

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


### FOR IRIS DATASET

##########################
###### Iris Dataset ######
##########################

# iris = datasets.load_iris()
# Y = np.array(iris['target'])
# X = np.array(iris['data'])
# w_value_hist = []
# ac_range2 = []
# taurange = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3]

# for tau in taurange:
#     acc_range=[]
#     it = 0
#     while(it < 20):
#         it += 1
#         X_train, Y_train, X_test, Y_test = preprocessing(X, Y)
#         mu = 1.1
#         w, w_hist, phi_w_hist = SVM_vector(X_train, Y_train, tau, mu)
#         w_value_hist.append(w)
#         Y_predicted, acc = predict(w, X_test, Y_test)
#         acc_range.append(acc)
#     ac_range2.append(np.mean(acc_range))

# fig = plt.figure()
# ax = fig.gca()
# line1, = ax.plot(taurange, ac_range2)
# ax.set_xscale('log')
# ax.set_ylabel(r'Accuracy', fontsize=16)
# ax.set_xlabel(r'$\tau$', fontsize=16)
# plt.show()


############################
###### Random Dataset ######
############################

# # number of sample
# N = 200

# # number of features
# n = 3

# # data is considered as normal with different value of mu.
# mu1 = np.repeat(-1, n)
# mu2 = np.repeat(1, n)

# X = np.concatenate((
#         np.random.normal(mu1, scale=2, size=(N, n)),
#         np.random.normal(mu2, scale=2, size=(N, n)))
#     , axis=0)

# Y = np.append(np.repeat(-1, N), np.repeat(1, N))

# X_train, Y_train, X_test, Y_test = preprocessing(X, Y)

# mu = 1.1
# tau = 0.1
# w, w_hist = SVM_vector(X_train, Y_train, tau, mu, dual=True)
# Y_predicted, acc = predict(w, X_test, Y_test)
# print(acc)


############################
###### duality gap #########
############################

# # We use here the Iris Data
# iris = datasets.load_iris()
# Y = np.array(iris['target'])
# X = np.array(iris['data'])

# # Here we use random generated values
# # number of sample
# N = 3

# # number of features
# n = 2

# # data is considered as normal with different value of mu.
# mu1 = np.repeat(-1, n)
# scale1 = 0.7
# mu2 = np.repeat(4, n)
# scale2 = 0.6


# X = np.concatenate((
#         np.random.normal(mu1, scale=scale1, size=(N, n)),
#         np.random.normal(mu2, scale=scale2, size=(N, n)))
#     , axis=0)
# # Use y=1 of y=2 to have the same labels as the iris data
# Y = np.append(np.repeat(1, N), np.repeat(2, N))


# w_value_hist = []
# ac_range2 = []
# tau = 0.1
# X_train, Y_train, X_test, Y_test = preprocessing(X, Y, percent=1.)

# murange = [2 , 15, 50, 100]

# acc_range=[]
# acc_norm_step = []
# for mu in murange:
#     # get optimal solution history for the primal
#     w_primal, w_hist_primal, phi_w_hist_primal = SVM_vector(X_train, Y_train, tau, mu)
#     w_primal = np.array(w_primal)
#     w_hist_primal = w_hist_primal[:, :X_train.shape[1]]
#     #acc_norm_step.append(LA.norm(w_hist_primal - w_primal, axis=1))
#     phi_w_hist_primal = np.array(phi_w_hist_primal)[0]

#     # get optimal solution history for the dual
#     w_dual, w_hist_dual, phi_w_hist_dual = SVM_vector(X_train, Y_train, tau, mu, dual=True)
#     w_dual = np.array(w_dual)
#     w_hist_dual = w_hist_dual[:, :X_train.shape[1]]
#     acc_norm_step.append(LA.norm(w_hist_dual - w_dual, axis=1))
#     phi_w_hist_dual = np.array(phi_w_hist_dual)[0]


#     rest = phi_w_hist_primal.shape[0] - phi_w_hist_dual.shape[0]

#     # Enlarge the history of phi in order to have the same length
#     if(rest>0):
#         phi_w_hist_dual = np.append(phi_w_hist_dual, np.repeat(phi_w_hist_dual[-1], rest), axis=0)
#     elif(rest<0):
#         phi_w_hist_primal = np.append(phi_w_hist_primal, np.repeat(phi_w_hist_primal[-1], -rest))

#     duality_gap = phi_w_hist_primal - phi_w_hist_dual

#     # I firstly suppose that the duality gap is the distance between
#     # the current argument and the global minimizer
#     # but as we talk about "duality" gap,
#     # I suppose that the duality gap must be the distance between
#     # the function and the dual function
#     # But it does not converge to 0...
#     # I must have made a mistake somewhere...

# fig = plt.figure()

# ax = fig.gca()

# line0, = ax.plot(acc_norm_step[0], label=r'$\mu = 2$')
# line1, = ax.plot(acc_norm_step[1], label=r'$\mu = 15$')
# line2, = ax.plot(acc_norm_step[2], label=r'$\mu = 50$')
# line3, = ax.plot(acc_norm_step[3], label=r'$\mu = 100$')

# ax.legend(fontsize=12)
# ax.set_yscale('log')
# ax.set_ylabel(r'$|x_{t} - x^*|$', fontsize=16)
# ax.set_xlabel('iterations number')
# plt.show()


##################################################
####### PLOT THE FIRST 2 FEATURES IN A PLAN ######
##################################################

# fig = plt.figure()

# ax = plt.gca()

# n = 2 * N
# for i in np.arange(n):
#     xs = X_train[i,1]
#     ys = X_train[i,0]
#     if(Y_train[i] < 0):
#         ax.scatter(xs, ys, c='r', marker='o')
#     else:
#         ax.scatter(xs, ys, c='b', marker='^')

# x_surf, y_surf = w_primal[1], w_primal[0]

# x0 = np.arange(-2, 2, 0.25)
# y0 = -x_surf * x0 / y_surf

# plt.plot(x0, y0)

# plt.show()

# fig = plt.figure()


######################################################
####### PLOT THE FIRST 3 FEATURES IN A 3D-SPACE ######
######################################################
# ax = fig.add_subplot(111, projection='3d')


# for i in np.arange(n):
#     xs = X[i,0]
#     ys = X[i,1]
#     zs = X[i,2]
#     if(Y[i] < 0):
#         ax.scatter(xs, ys, zs, c='r', marker='o')
#     else:
#         ax.scatter(xs, ys, zs, c='b', marker='^')

# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')


# x_surf, y_surf, z_surf = w[0], w[1], w[2]
# x0 = np.arange(-2.5, 2.5, 0.25)
# y0 = np.arange(-2.5, 2.5, 0.25)
# x0, y0 = np.meshgrid(x0, y0)
# z0 = (-x0*x_surf - y0*y_surf)/z_surf
# surf = ax.plot_surface(x0, y0, z0,
#                        linewidth=0, antialiased=True)

# plt.show()










