
import numpy as np
import os

# ---- prog config 
# type of the data element (default)
dataType = np.float32	
# maximum number of utterances in one single .nc file that will be 
#   generated. Fif the number of utterances is larger than this 
#   threshold, multiple .nc file will be generatd.
#   Be careful, single large .nc may not be supported by classic NETCDF
flushThreshold = 1100   


# ---- data config
labdim  = 382
mgcdim  = 60
vuvdim  = 1
lf0dim  = 1
bapdim  = 25

dlf0dim = lf0dim * 3
dmgcdim = mgcdim * 3
dbapdim = bapdim * 3

# dimension of the input feature (a list)
inDim = [labdim,] 

# dimension of the output features (a list) MUST be set
outDim= [vuvdim]         

# MUST be empty
outScpFile = []                         

# name of each scp for each input features (a list)            
inScpFile = ['lab.scp',]                


# for test set, specify the outputName of each output feature file
outputName = ['lf0']
# whether each output stream has delta component
# 3: use static, delta-delta and delta
# 2: use static, delta
# 1: use static 
outputDelta= [1, ]

# For F0 re-construction
# This is the same information used to quantize the F0
# [maxF0, minF0, F0 levels]
f0Info     = [571.0, 113.0, 128, False]


# ---- output config
# name of the output scp for generating .nc package(default)
allScp = 'all.scp'                     

