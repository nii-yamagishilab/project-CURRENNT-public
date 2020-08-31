#!/usr/bin/python
"""
This script generates the configuration file for MDN network
Please add PYTHONPATH to pyTools
Please config and python createMDNConfig.py
"""

import sys, os
import numpy as np
from ioTools import readwrite as py_rw

def kmixPara(dim, k, tieVariance):
    if k>0:
        if tieVariance:
            return k*(dim+2)
        else:
            return k*(dim+dim+1)
    elif k<0:
        return -1*k    
    else:
        return 1

## ------ Configuration ---------------------------------
# MDNType: specify the type of Mixture density units
#   N<0: softmax, the range is [0, -N)
#   0:   sigmoid
#   N>0: N mixture of Gaussian
#
#   For example, I want to use a 4-mixture Gaussian model
#   to describe the target feature vector, 
#      MDNType = [4, ]
#   
#   Multiple kinds of distribution can be specified for different part
#   of the target feature vector. I want to use a 4-mixture and 1-mixture
#   components: 
#      MDNType = [4, 1]
#   
#   Personally, I use the configure below.
#   In DATA/data_config.py, it says outScpFile = ['mgc.scp', 'vuv.scp', 'lf0.scp', 'bap.scp']  
#   Thus, I set the MDNType accordingly [2, 0, 2, 1], where 
#   I use 2 mixture for MGC, sigmoid for U/V., 2 mixture for LF0, 1 mixture for BAP
MDNType  = [-128]

# MDNTargetDim
#   specify the dimension range of each MDN component corresponding
#   to the target feature vector
#   For example, I use 4-mixture Gaussian to describe the 1-259th
#   dimension of the data
#   
#   In accord with MDNType above, 
#   I need to specify the dimension range for MGC, VU, Lf0, BAP
#   As DATA/data_config.py says outDim= [dmgcdim, vuvdim, dlf0dim, dbapdim]
#   and dmgcdim = 180, vuvdim = 1, dlf0dim = 3, dbapdim = 75, I need to set:
MDNTargetDim  = [[0,1],]

# tieVariance
#   do you want to share the variance across dimensions ?
#   I don't
tieVariance   = 0

# OutputFile (default)
mdnconfig    = './mdn.config';

## ------ Run ---------------------------------
# MDNNNOutDim (automatically generated. No need specify)
#   specify the dimension range of each MDN component corresponding 
#   to the output of neural network (or, input to MDN layer)
#   Note: the number of dimension is determined by the MDNTargetDim
#   and the distribution used. 
#   For k-mixture Gaussian, the number should be 
#          mixture weight  + shared variance  + mean of all mixtures
#          #mixture        + #mixture +         #mixture * #featureDim
#   For sigmoid, equal to that in MDNTargetDim
#   For softmax, equal to the number of softmax component
bias = 0
MDNNNOutDim = []
for idx, x in enumerate(MDNType):
    temp = kmixPara(MDNTargetDim[idx][1]-MDNTargetDim[idx][0], x, tieVariance)
    MDNNNOutDim.append([bias, bias+temp])
    bias = temp+bias
print MDNNNOutDim

# check and generating the MDN configuration
mdnconfigdata = np.zeros([1+len(MDNType)*5], dtype = np.float32)
mdnconfigdata[0] = len(MDNType)

tmp = 0
for idx, mdntype in enumerate(MDNType):
    mdntarDim = MDNTargetDim[idx]
    mdnoutDim = MDNNNOutDim[idx]
    tmp1 = kmixPara(mdntarDim[1]-mdntarDim[0], mdntype, tieVariance)
    tmp2 = (mdnoutDim[1]-mdnoutDim[0])
        
    if mdntype > 0:
        assert tmp1 == tmp2, "Error in MDN mixture configuraiton"
        tmp = tmp + tmp2
        
    elif mdntype < 0:
        assert mdntarDim[1]-mdntarDim[0]==1, "Softmax to 1 dimension targert"
        tmp = tmp + (mdnoutDim[1]-mdnoutDim[0])
        mdntype = -1 # change it back to -1
    else:
        tmp = tmp + tmp2
    mdnconfigdata[(idx*5+1):((idx+1)*5+1)] = [mdnoutDim[0],mdnoutDim[1],
                                              mdntarDim[0],mdntarDim[1],
                                              mdntype]

print "Dimension of output of NN should be %d" % (tmp)
py_rw.write_raw_mat(mdnconfigdata, mdnconfig)
