import pandas as pd

from kernel_functions import *
from utils import *

# load all data as the numpy array type
# test inputs
X_raw0 = pd.read_csv('data/Xtr0.csv', sep=' ',
                     header=None).values.reshape((-1))
X_raw1 = pd.read_csv('data/Xtr1.csv', sep=' ',
                     header=None).values.reshape((-1))
X_raw2 = pd.read_csv('data/Xtr2.csv', sep=' ',
                     header=None).values.reshape((-1))
# test inputs
X_test0 = pd.read_csv('data/Xte0.csv', sep=' ',
                      header=None).values.reshape((-1))
X_test1 = pd.read_csv('data/Xte1.csv', sep=' ',
                      header=None).values.reshape((-1))
X_test2 = pd.read_csv('data/Xte1.csv', sep=' ',
                      header=None).values.reshape((-1))
# train outputs
Y0 = pd.read_csv('data/Ytr0.csv', sep=',', header=0)['Bound'].values
Y1 = pd.read_csv('data/Ytr1.csv', sep=',', header=0)['Bound'].values
Y2 = pd.read_csv('data/Ytr2.csv', sep=',', header=0)['Bound'].values

# Kernel computation, default is k_fold (spectrum kernel)
FOLD = 32
X_0 = count_kuplet_k(X_raw0, k=6)
X_test0 = count_kuplet_k(X_test0, k=6)

X_1 = to_k_fold(X_raw1, fold=FOLD)
X_test1 = to_k_fold(X_test1, fold=FOLD)

diDist = dynDiMismatchDist(6)
X_2 = diMismatchFeatures(X_raw2, k=6, m=3, diMismatchDist=diDist)
X_test2 = diMismatchFeatures(X_test2, k=6, m=3, diMismatchDist=diDist)

# Train and predict for the three TFs
K0, Ktest0 = kernelize(X_0, X_test0)
w0, bias0 = fit(K0, Y0, lamb=1.2e-4, verbose=False)
Y_pred0 = predict(w0, bias0, Ktest0)

K1, Ktest1 = kernelize(X_1, X_test1)
w1, bias1 = fit(K1, Y1, lamb=3e-5, verbose=False)
Y_pred1 = predict(w1, bias1, Ktest1)

K2, Ktest2 = kernelize(X_2, X_test2)
w2, bias2 = fit(K2, Y2, lamb=7e-4, verbose=False)
Y_pred2 = predict(w2, bias2, Ktest2)

test0 = np.transpose(Y_pred0)[:][0]
test1 = np.transpose(Y_pred1)[:][0]
test2 = np.transpose(Y_pred2)[:][0]

bound = np.concatenate((test0, test1, test2), axis=0).reshape((-1)).astype(int)
final = pd.DataFrame(np.arange(3000), columns=['Id'])
final['Bound'] = bound
final.to_csv('result_best.csv', index=None)
