# Kernel functions

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

