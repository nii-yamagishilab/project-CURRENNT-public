
import numpy as np
import os

# ---- prog config 
dataType = np.float32	# type of the data element (default)
flushThreshold = 1100   # maximum number of utterances in one single .nc 
                        # file that will be generated
                        # if the number of utterances is larger than this threshold, 
                        # multiple .nc files will be generatd.
                        # Be careful, single large .nc may not be supported by classic NETCDF
# ---- data config
labdim  = 382
mgcdim  = 60
wedim   = 1
vuvdim  = 1
lf0dim  = 1
bapdim  = 25

dlf0dim = lf0dim * 3
dmgcdim = mgcdim * 3
dbapdim = bapdim * 3

inDim = [labdim, wedim]                     # dimension of the input feature (a list)
outDim= [dmgcdim, vuvdim, dlf0dim, dbapdim] # dimension of the output features (a list) MUST be set

outScpFile = []                             # MUST be empty for output
inScpFile = ['lab.scp','we.scp']            # name of each scp for each input features (a list)


# Configuration for data generation method
#   This is only used for MLPG algorithm in speech synthesis task
#   set it to [1, 1, ...] if you don't know about MLPG
#
#   for test set, specify the outputName of each output feature file
outputName = ['mgc_delta', 'vuv', 'lf0_delta', 'bap_delta']
#   whether each output stream has delta component
#   3: use static, delta-delta and delta
#   2: use static, delta
#   1: use static 
outputDelta= [3, 1, 3, 3]

# Feature Mask:
#    In case only some dimensions of the data are required,
#    use the InMask and outMask to select the span of the data dimension
#    For example, I just want to use 1-292th dimension of each data in 'lab.scp',
#    uncomment the following two lines
#      inMask  = [[0, 292],]
#      outMask = []
#    Note, currently, you can only specify the dimension in a continuous span.
#    If you want 1-10th and 20-30th, duplicate the data configuration
#      inScpFile= ['lab.scp','lab.scp']
#      inDim    = [labdim, labdim]
#      inMask   = [[0, 10], [19, 30]]

# Normalization Mask:
#    Dy default, all dimensions of input and output should be normalized.
#    To prevent normalization on some dimension, use normMask below.
#    normMask specify
#    For example, I don't want to normalize 292-382th dimension, use
#      normMask  = [[291, 382], ]
#    Both input and output in the same normMask, 
#    Makesure, length of normMask is equal to length(inDim) + length(outDim)
normMask  = [[], [0], [], [], [],[]]

# ---- output config
allScp = 'all.scp'                          # name of the output scp (default)

