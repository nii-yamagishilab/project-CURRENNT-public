
import numpy as np
import os

# ---- prog config 
# type of the data element (default)
dataType = np.float32	
# maximum number of utterances in one single .nc file that will be generated
#   if the total number of utterances is larger than this threshold, multiple .nc files
#   will be generatd.
#   Be careful, large .nc may not be supported by classic NETCDF
#   Make sure a .nc file is smalled than 3G
flushThreshold = 3000
                        
                         
# ---- data config
labdim  = 1

lf0dim  = 1



# dimension of the input feature (a list)
inDim = [labdim,]                                   

# dimension of the output features (a list)
outDim= [lf0dim,]         

# name of the scp for output features (a list)
outScpFile = []  

# name of the scp for input features (a list)
inScpFile = ['lab.scp',]                                   

# to add features as input or output, specify scpFile and dimension 
# for example, use lab and we data as input feature:
#    inputScpFile = ['lab.scp', 'we.scp']
#    inDim = [labdim, wedim]
normMask = [[0], [0]]

# for test set, specify the outputName of each output feature file
outputName = ['raw']
# whether each output stream has delta component
# 3: use static, delta-delta and delta
# 2: use static, delta
# 1: use static
outputDelta= [1, ]

# ---- output config
# name of the output scp for generating .nc package(keep it as default)
allScp = 'all.scp'                                  

