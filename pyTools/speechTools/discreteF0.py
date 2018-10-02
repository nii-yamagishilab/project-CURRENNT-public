#!/usr/bin/python

import numpy as np
import sys, os
import scipy
from   scipy import io


def f0Conversion(datain, f0Max, f0Min, f0Inter, direction, f0inter=False, f0Threshold=10):
    """ 
    dataout = f0Conversion(datain, f0Max, f0Min, f0Inter, direction, f0Threshold, f0inter):
    f0Max: maximum F0 value in the corpus
    f0Min: minimum F0 value in the corpus
    f0Inter: number of f0 quantized interval
    direction: d2c: from discrete to continuous F0
               c2d: from continuous to discrete F0
    f0Threshold: below which the F0 value will be considered as unvoiced 
    f0inter:     has this F0 trajectory been interpolated
    """
    if direction == 'd2c':

        # Note,
        # For hierarchical softmax, datain = 0, or a number >= 1
        # For plain softmax, datain > 0. In this case, datain < 1 should be considered as unvoiced
        F0VIdx      = datain > 0
        vuv         = np.zeros(datain.shape)
        vuv[F0VIdx] = 1.0
        
        dataout     = datain
        if f0inter:
            dataout[F0VIdx] = (dataout[F0VIdx])/(f0Inter-1.0)*(f0Max - f0Min) + f0Min
        else:
            dataout[F0VIdx] = (dataout[F0VIdx] - 1)/(f0Inter-2.0)*(f0Max - f0Min) + f0Min

    else:
        F0VIdx      = datain >  f0Threshold    # F0 below the threshold is unvoiced
        F0UIdx      = datain <= f0Threshold
        vuv         = np.zeros(datain.shape)
        vuv[F0VIdx] = 1.0
    
        dataout     = datain
        
        if f0inter:
            dataout[F0VIdx] = np.round((dataout[F0VIdx] - f0Min)/(f0Max - f0Min)*(f0Inter-1.0))
            dataout[dataout > (f0Inter-1.0)] = (f0Inter-1.0)
            dataout[dataout < 0] = 0
        else:
            dataout[F0VIdx] = np.round((dataout[F0VIdx] - f0Min)/(f0Max - f0Min)*(f0Inter-2.0))
            dataout[dataout > (f0Inter-2.0)] = (f0Inter-2.0)
            dataout[dataout < 0] = 0
            dataout[F0VIdx] = dataout[F0VIdx] + 1 # shift up and leave the 0th dimension to unvoiced
            dataout[F0UIdx] = 0
            
    return dataout, vuv
