
import numpy as np
import os

# ---- prog config 
dataType = np.float32	# type of the data element (default)
flushThreshold = 2      # maximum number of utterances in one single .nc file 
                        # that will be generated
                        # if the number of utterances is larger than this threshold, 
                        # multiple .nc files will be generatd.
                        # Be careful, single large .nc may not be supported by classic NETCDF
# ---- data config

# Input and Output 
labdim  = 382
mgcdim  = 60
vuvdim  = 1
lf0dim  = 1
bapdim  = 25

dlf0dim = lf0dim * 3
dmgcdim = mgcdim * 3
dbapdim = bapdim * 3

inDim = [labdim,]                                   # dimension of the input feature (a list)
outDim= [dmgcdim, vuvdim, dlf0dim, dbapdim]         # dimension of the output features (a list)

#    name of the scp for output features (a list)
outScpFile = ['mgc.scp', 'vuv.scp', 'lf0.scp', 'bap.scp']  

#    name of the scp for input features (a list)
inScpFile = ['lab.scp',]                                   
#    to add features as input or output, specify scpFile and dimension 
#    for example, use lab and we data as input feature:
#    inputScpFile = ['lab.scp', 'we.scp']
#    inDim = [labdim, wedim]

# Feature Mask:
#    In case only some dimensions of the data are required,
#    use the InMask and outMask to select the span of the data dimension
#    For example, I just want to use 1-292th dimension of each data in 'lab.scp',
#    uncomment the following two lines
#      inMask  = [[0, 292],]
#      outMask = [[], [], [], []]
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
#      normMask  = [[291, 382], [], [], [], []]
#    Both input and output in the same normMask

# ---- output config
allScp = 'all.scp'                                  # name of the output .scp (default)

