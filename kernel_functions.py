import numpy as np
import pandas as pd
from tqdm import tqdm


from itertools import product
from string import ascii_lowercase

'''
Compute the gram matrix of a reproducing kernel K(x, y) and data X even if the embedding phi is unknown
input: 	X as array
		rk a python function computing K(x, y) (function from X^2 -> R)
output: [K(x_i, y_j)] = <phi(x_i), phi(x_j)> for i,j in 1..n
'''
def gram_rk(X, rk):
	n = X.shape[0]
	K = np.zeros((n, n))
	for i in tqdm(range(n)):
		for j in range(n):
			K[i, j] = rk(X[i], X[j])
	return K

'''
Compute the gram matrix of a kernel given the embedding phi of this kernel
input: 	X as array
		phi a python function computing phi(x) (from X to R^d where d is the dimension of the embedding)
output: [K(x_i, y_j)] = <phi(x_i), phi(x_j)> for i,j in 1..n
'''
def gram_phi(X, phi):
	n = X.shape[0]
	X_feat = np.array([phi(x) for x in X])
	K = X_feat.dot(X_feat.T)
	return K

# k_spectrum kernel with k = 3
## Creation of all possible combinations
base_azote = ['A', 'C', 'T', 'G']
dic_aa = [''.join(i) for i in product(base_azote, repeat = 3)]
nb_feat = len(dic_aa)

def count_kuplet_3(seq):
    k_grams_count = np.zeros(nb_feat)
    for i,e in enumerate(dic_aa):
        k_grams_count[i] = seq.count(e)
    return k_grams_count

def count_kuplet_k(seq, k=3):
    base_azote = ['A', 'C', 'T', 'G']
    dic_k = [''.join(i) for i in product(base_azote, repeat = k)]
    nb_feat = len(dic_k)

    k_grams_count = np.zeros(nb_feat)
    for i,e in enumerate(dic_k):
        k_grams_count[i] = seq.count(e)
    return k_grams_count

def compute_gap_kernel(X1, X2, k, lamb=0.5):
    """
    Compute the 'sub string kernel'
    complexity in O(k*X1*X2)
    """
    N_X1 = len(X1)
    N_X2 = len(X2)
    B = np.zeros((k+1, N_X1+1, N_X2+1))
    K = np.zeros((N_X1+1, N_X2+1))
    ker = np.zeros(k+1)
    for i in range(1, N_X1+1):
        for j in range(1, N_X2+1):
            if(X1[i-1] == X2[j-1]):
                B[1, i, j] = lamb**2
                ker[1] += lamb**2
    # ker = ker/(lamb**2) # renormalize
    for l in range(2, k+1):
        for i in range(1, N_X1+1):
            for j in range(1, N_X2+1):
                K[i, j] = B[l-1, i, j] + lamb*K[i-1, j] + lamb*K[i, j-1] - (lamb**2)*K[i-1, j-1]
                if X1[i-1]==X2[j-1]:
                    B[l, i, j] = lamb**2 * K[i-1, j-1]
                    ker[l] = ker[l] + B[l, i, j]
    return ker[k]

def compute_gap_kernel_param(param):
    return compute_gap_kernel(param[0], param[1], param[2])


def S(a, b):
    if a!=b:
        return 0.
    else:
        return 1.

def g(n, d=1, e=1):
    if n >0:
        return d + e*(n-1)
    elif n==0:
        return 0.
    else:
        print("Negative Gap")

def LA_kernel(u, v, d=1., e=1., beta = 0.00001):

    N_u = len(u)
    N_v = len(v)

    M = np.zeros((N_u+1, N_v+1))
    X = np.zeros((N_u+1, N_v+1))
    Y = np.zeros((N_u+1, N_v+1))
    X2 = np.zeros((N_u+1, N_v+1))
    Y2 = np.zeros((N_u+1, N_v+1))
    for i in range(1, N_u+1):
        for j in range(1, N_v+1):
            M[i, j] = np.exp(beta * S(u[i-1],v[j-1]))*(1.+X[i-1,j-1]+Y[i-1,j-1]+M[i-1,j-1])
            X[i, j] = np.exp(-beta*d) * M[i-1, j] + np.exp(-beta*e)*X[i-1, j]
            Y[i, j] = np.exp(-beta*d) *(M[i, j-1]+X[i, j-1]) + np.exp(-beta*e)*Y[i, j-1]
            X2[i, j] = M[i-1, j] + X2[i-1, j]
            Y2[i, j] = M[i, j-1] + X2[i, j-1] + Y2[i, j-1]
    return np.log(1. + X2[N_u, N_v] + Y2[N_u, N_v] + M[N_u, N_v])


