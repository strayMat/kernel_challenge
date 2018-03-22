import numpy as np
from tqdm import tqdm
from itertools import product
import sys
from collections import defaultdict


'''
Compute the gram matrix of a reproducing kernel K(x, y) and data X even if the embedding phi is unknown
input:  X as array
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
input:  X as array
        phi a python function computing phi(x) (from X to R^d where d is the dimension of the embedding)
output: [K(x_i, y_j)] = <phi(x_i), phi(x_j)> for i,j in 1..n
'''


def gram_phi(X, phi):
    X_feat = np.array([phi(x) for x in X])
    K = X_feat.dot(X_feat.T)
    return K


# k_spectrum kernel with k = 3
# Creation of all possible combinations
base_azote = ['A', 'C', 'T', 'G']
dic_aa = [''.join(i) for i in product(base_azote, repeat=3)]
nb_feat = len(dic_aa)


def count_kuplet_3(seq):
    k_grams_count = np.zeros(nb_feat)
    for i, e in enumerate(dic_aa):
        k_grams_count[i] = seq.count(e)
    return k_grams_count


def count_kuplet_k(seq, k=3):
    base_azote = ['A', 'C', 'T', 'G']
    dic_k = [''.join(i) for i in product(base_azote, repeat=k)]
    nb_feat = len(dic_k)

    k_grams_count = np.zeros(nb_feat)
    for i, e in enumerate(dic_k):
        k_grams_count[i] = seq.count(e)
    return k_grams_count


def compute_gap_kernel(X1, X2, k, lamb=0.5):
    """
    Compute the 'sub string kernel'
    complexity in O(k*X1*X2)
    """
    N_X1 = len(X1)
    N_X2 = len(X2)
    B = np.zeros((k + 1, N_X1 + 1, N_X2 + 1))
    K = np.zeros((N_X1 + 1, N_X2 + 1))
    ker = np.zeros(k + 1)
    for i in range(1, N_X1 + 1):
        for j in range(1, N_X2 + 1):
            if(X1[i - 1] == X2[j - 1]):
                B[1, i, j] = lamb**2
                ker[1] += lamb**2
    # ker = ker/(lamb**2) # renormalize
    for l in range(2, k + 1):
        for i in range(1, N_X1 + 1):
            for j in range(1, N_X2 + 1):
                K[i, j] = B[l - 1, i, j] + lamb * K[i - 1, j] + \
                    lamb * K[i, j - 1] - (lamb**2) * K[i - 1, j - 1]
                if X1[i - 1] == X2[j - 1]:
                    B[l, i, j] = lamb**2 * K[i - 1, j - 1]
                    ker[l] = ker[l] + B[l, i, j]
    return ker[k]


def compute_gap_kernel_param(param):
    return compute_gap_kernel(param[0], param[1], param[2])


def S(a, b):
    if a != b:
        return 0.
    else:
        return 1.


def g(n, d=1, e=1):
    if n > 0:
        return d + e * (n - 1)
    elif n == 0:
        return 0.
    else:
        print("Negative Gap")


def LA_kernel(u, v, d=1., e=1., beta=0.00001):

    N_u = len(u)
    N_v = len(v)

    M = np.zeros((N_u + 1, N_v + 1))
    X = np.zeros((N_u + 1, N_v + 1))
    Y = np.zeros((N_u + 1, N_v + 1))
    X2 = np.zeros((N_u + 1, N_v + 1))
    Y2 = np.zeros((N_u + 1, N_v + 1))
    for i in range(1, N_u + 1):
        for j in range(1, N_v + 1):
            M[i, j] = np.exp(beta * S(u[i - 1], v[j - 1])) * (1. +
                                                              X[i - 1, j - 1] + Y[i - 1, j - 1] + M[i - 1, j - 1])
            X[i, j] = np.exp(-beta * d) * M[i - 1, j] + \
                np.exp(-beta * e) * X[i - 1, j]
            Y[i, j] = np.exp(-beta * d) * (M[i, j - 1] +
                                           X[i, j - 1]) + np.exp(-beta * e) * Y[i, j - 1]
            X2[i, j] = M[i - 1, j] + X2[i - 1, j]
            Y2[i, j] = M[i, j - 1] + X2[i, j - 1] + Y2[i, j - 1]
    return np.log(1. + X2[N_u, N_v] + Y2[N_u, N_v] + M[N_u, N_v])





# Mismatch and diMismatch Kernel from https://academic.oup.com/bioinformatics/article/33/19/3003/3852080
def substitutionDistance(s1, s2):
    assert len(s1) == len(
        s2), "Error ! {} and {} are not of the same length !".format(s1, s2)

    dist = 0
    for pos in np.arange(len(s1)):
        if s1[pos] != s2[pos]:
            dist += 1
    return dist


# Compute the di-mismatch distance (substitution distance fo di_gram)
def diDistance(s1, s2):
    assert len(s1) == len(
        s2), "Error ! {} and {} are not of the same length !".format(s1, s2)

    dist = 0
    prePos = -1
    for pos in np.arange(len(s1)):
        if s1[pos] != s2[pos]:
            if prePos + 1 == pos:
                dist += 1
            else:
                dist += 2
            if pos == len(s1) - 1:
                dist -= 1
            prePos = pos
    return dist


# create the list and dictionnary of all possible k_grams in the alphabet
def k_gramGen(k, alphabet='ATGC'):
    gramList = [''.join(i) for i in product(alphabet, repeat=k)]
    gramDict = dict(zip(gramList, np.arange(len(gramList))))
    return gramList, gramDict
# print(len(k_gramGen(3)[0]))

# compute the Mismatch distance between all possible kgram in a dynamic way
def dynMismatchDist(k, alphabet='ATGC',
                    gramList=None, gramDict=None):
    if (gramList is None) | (gramDict is None):
        gramList, gramDict = k_gramGen(k, alphabet=alphabet)

    nb_grams = len(gramList)
    mismatchDist = np.zeros((nb_grams, nb_grams))
    if k == 2:
        for i in np.arange(nb_grams):
            i_gram = gramList[i]
            for j in np.arange(nb_grams):
                j_gram = gramList[j]
                mismatchDist[i, j] = substitutionDistance(i_gram, j_gram)

    else:  # k >2, we use prefixes
        pMismatchDist = dynMismatchDist(k - 1, alphabet=alphabet)
        pgramList, pgramDict = k_gramGen(k - 1, alphabet=alphabet)
        for i in tqdm(np.arange(nb_grams)):
            i_gram = gramList[i]
            i_pgram = i_gram[:-1]
            i_pgram_ix = pgramDict[i_pgram]
            for j in np.arange(nb_grams):
                j_gram = gramList[j]
                j_pgram = j_gram[:-1]
                j_pgram_ix = pgramDict[j_pgram]
                pDist = pMismatchDist[i_pgram_ix, j_pgram_ix]
                if i_gram[k - 1] == j_gram[k - 1]:
                    mismatchDist[i, j] = pDist
                else:
                    mismatchDist[i, j] = pDist + 1

    return mismatchDist


# compute the diMismatch distance between all possible kgram in a dynamic way
def dynDiMismatchDist(k, alphabet='ATGC',
                      gramList=None, gramDict=None):
    if (gramList is None) | (gramDict is None):
        gramList, gramDict = k_gramGen(k, alphabet=alphabet)

    nb_grams = len(gramList)
    diMismatchDist = np.zeros((nb_grams, nb_grams))
    if k == 2:
        for i in np.arange(nb_grams):
            i_gram = gramList[i]
            for j in np.arange(nb_grams):
                j_gram = gramList[j]
                diMismatchDist[i, j] = diDistance(i_gram, j_gram)

    else:  # k >2, we use prefixes
        pDimismatchDist = dynDiMismatchDist(k - 1, alphabet=alphabet)
        pgramList, pgramDict = k_gramGen(k - 1, alphabet=alphabet)
        for i in tqdm(np.arange(nb_grams)):
            i_gram = gramList[i]
            i_pgram = i_gram[:-1]
            i_pgram_ix = pgramDict[i_pgram]
            for j in np.arange(nb_grams):
                j_gram = gramList[j]
                j_pgram = j_gram[:-1]
                j_pgram_ix = pgramDict[j_pgram]
                pDist = pDimismatchDist[i_pgram_ix, j_pgram_ix]
                if i_gram[k - 1] == j_gram[k - 1] and i_gram[k - 2] == j_gram[k - 2]:
                    diMismatchDist[i, j] = pDist
                else:
                    diMismatchDist[i, j] = pDist + 1

    return diMismatchDist

# compute the dimismatch dictionnary with m maximum number of mismatch for all k-gram (m mismatch threshold)
def buildDiMismatchTable(k, m, alphabet='ATGC',
                         gramList=None, gramDict=None,
                         diMismatchDist=None):
    assert k > 0, "Error! k={} needs to be positive".format(k)
    assert m >= 0, "Error! m={} needs to be non negative".format(k)
    assert k > m, "Error! k={} needs to be greater than m={}".format(k, m)

    if (gramList is None) | (gramDict is None):
        gramList, gramDict = k_gramGen(k, alphabet=alphabet)

    nb_grams = len(gramList)

    # if diMismatchDist has been dynamically computed
    if diMismatchDist is not None:
        diMismatchTable = (diMismatchDist <= m) * \
            (1 - diMismatchDist / (k - 1.0))

    # if dimismatchdist has not been dynamically computed
    else:
        diMismatchTable = np.zeros((nb_grams, nb_grams))
        for i in tqdm(np.arange(nb_grams)):
            i_gram = gramList[i]
            for j in np.arange(nb_grams):
                j_gram = gramList[j]
                dist = diDistance(i_gram, j_gram)
                if dist <= m:
                    diMismatchTable[i, j] = 1 - dist / (k - 1.0)
    return diMismatchTable

# compute the mismatch dictionnary with m maximum number of mismatch for all k-gram (m mismatch threshold)
def buildMismatchTable(k, m, alphabet='ATGC',
                       gramList=None, gramDict=None,
                       mismatchDist=None):
    assert k > 0, "Error! k={} needs to be positive".format(k)
    assert m >= 0, "Error! m={} needs to be non negative".format(k)
    assert k > m, "Error! k={} needs to be greater than m={}".format(k, m)

    if (gramList is None) | (gramDict is None):
        gramList, gramDict = k_gramGen(k, alphabet=alphabet)

    nb_grams = len(gramList)

    # if mismatchDist has been dynamically computed
    if mismatchDist is not None:
        mismatchTable = (mismatchDist <= m) * (1 - mismatchDist / (k - 1.0))

    # if mismatchdist has not been dynamically computed
    else:
        mismatchTable = np.zeros((nb_grams, nb_grams))
        for i in tqdm(np.arange(nb_grams)):
            i_gram = gramList[i]
            for j in np.arange(nb_grams):
                j_gram = gramList[j]
                dist = substitutionDistance(i_gram, j_gram)
                if dist <= m:
                    mismatchTable[i, j] = 1 - dist / (k - 1.0)
    return mismatchTable


# compute the mismatch features PHI for a training sequence
# (kernel is then obtained computing K = PHI.dot(PHI.T))
# complexity : O(NL) where N the number of sequences and L the length of the sequences
def mismatchFeatures(X_train, k, m, alphabet='ATGC',
                     gramList=None, gramDict=None,
                     mismatchDist=None, mismatchTable=None):

    if (gramList is None) | (gramDict is None):
        gramList, gramDict = k_gramGen(k, alphabet=alphabet)

    if mismatchTable is None:
        print('Building Mismatch table for substrings...')
        mismatchTable = buildMismatchTable(k, m, alphabet=alphabet,
                                            gramList=gramList, gramDict=gramDict,
                                            mismatchDist=mismatchDist)

    Phi = np.zeros((len(X_train), len(gramList)))
    for s_ix in tqdm(np.arange(len(X_train))):
        for subStart in np.arange(len(X_train[0]) - k + 1):
            subSeq = X_train[s_ix][subStart:(subStart + k)]
            if subSeq not in gramList:
                sys.stderr.write('Error, {} is not a valid {}-gram for the alphabet {}'.format(subSeq, k, alphabet))
                sys.exit(1)
            subSeq_ix = gramDict[subSeq]
            Phi[s_ix, :] += mismatchTable[subSeq_ix, :]
    return Phi

# compute the mismatch features PHI for a training sequence (kernel is then obtained computing K = PHI.dot(PHI.T))


def diMismatchFeatures(X_train, k, m, alphabet='ATGC',
                       gramList=None, gramDict=None,
                       diMismatchDist=None, diMismatchTable=None):

    if (gramList == None) | (gramDict == None):
        gramList, gramDict = k_gramGen(k, alphabet=alphabet)

    if diMismatchTable == None:
        print('Building Mismatch table for substrings...')
        diMismatchTable = buildDiMismatchTable(k, m, alphabet=alphabet, gramList=gramList, gramDict=gramDict,
                                               diMismatchDist=diMismatchDist)

    Phi = np.zeros((len(X_train), len(gramList)))
    for s_ix in tqdm(np.arange(len(X_train))):
        for subStart in np.arange(len(X_train[0]) - k + 1):
            subSeq = X_train[s_ix][subStart:(subStart + k)]
            if subSeq not in gramList:
                sys.stderr.write('Error, {} is not a valid {}-gram for the alphabet {}'.format(subSeq, k, alphabet))
                sys.exit(1)
            subSeq_ix = gramDict[subSeq]
            Phi[s_ix, :] += diMismatchTable[subSeq_ix, :]
    return Phi


def count_kuplet_gap(seq, k_tmp, fold=5, k_grams_count=None):
    assert fold%2 == 1 or fold ==2, "fold must be odd"
    fold_size = len(bin(fold)[2:])
    fold = bin(fold)[2:]
    base_azote = ['A', 'C', 'T', 'G']

    if k_grams_count is None:
        k_grams_count = dict()

    tab = [''.join(i) for i in product(base_azote, repeat=np.sum([int(i) for i in fold]))]
    for code in tab:
        # print(''.join([code, '_', fold]))
        k_grams_count[''.join([code, '_', fold])] = 0.
        k_tmp[''.join([code, '_', fold])] = 0.

    for i in range(len(seq) - fold_size+1):
        l = seq[i:(i+fold_size)]
        # print(l)
        l = ''.join(l[i]*int(fold[i]) for i in range(len(fold)))
        k_grams_count[''.join([l, '_', fold])] += 1./(len(seq) - fold_size+1)
        k_tmp[''.join([l, '_', fold])] += 1.
    return k_grams_count, k_tmp

def count_k_fold(sent, k_tmp, fold1 = 9, fold2 = 64):
    k_grams_count = dict()
    k_tmp = dict()
    for fold in np.arange(fold1, fold2, 2):
        if(np.sum([int(i) for i in bin(fold)[2:]]) > 2):
            k_grams_count, k_tmp = count_kuplet_gap(sent, k_tmp, fold = fold, k_grams_count=k_grams_count)
    return np.array([k_grams_count[key] for key in sorted(k_grams_count)]), np.array([k_tmp[key] for key in sorted(k_tmp)])


def to_k_fold(X, fold1=9, fold2=64):
    '''
    From X (X_raw) compute a sequence X_process using count_k_fold
    '''

    X_process = []
    X_count = []
    k_tmp = defaultdict(int)
    for sent in tqdm(X):
        sent_process, k_tmp = count_k_fold(sent, k_tmp, fold1 = fold1, fold2 = fold2)
        X_process.append(sent_process)
        X_count.append(k_tmp)

    X_process = np.array(X_process) * len(X) * len(X[0])
    sum_tmp = np.sum(np.array(X_count), axis=0)

    X_process = np.divide(X_process, sum_tmp, out=np.zeros_like(X_process), where=sum_tmp!=0)

    return X_process
