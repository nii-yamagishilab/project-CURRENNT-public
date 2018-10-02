#!/usr/bin/python

import numpy as np
from numpy.linalg import inv

def getARMat(arCoef, Length):
    """ mat = getARMat(arCoef, Length)
    assume arCoef = [1, coef1, coef2] for 1 + coef1 * z^-1 + coef2 * z^-2
    return the matrix of ar transformation in size of [length, length] 
    """
    mat = np.eye(Length)
    order = arCoef.shape[0] - 1
    for n in np.arange(order):
        mat[np.arange(n+1, Length), np.arange(0, Length-(n+1))] = arCoef[n+1]
    return mat

def getInvARMat(arCoef, Length):
    return inv(getARMat(arCoef, Length))

