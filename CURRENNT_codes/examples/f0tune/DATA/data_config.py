
import numpy as np
import os

# ---- prog config 
dataType = np.float32	# type of the data element (default)
flushThreshold = 1100   # maximum number of utterances in one single .nc file that will be generated
                        # if the number of utterances is larger than this threshold, multiple .nc files
                        # will be generatd.
                        # Be careful, single large .nc may not be supported by classic NETCDF
# ---- data config
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

outScpFile = ['mgc.scp', 'vuv.scp', 'lf0.scp', 'bap.scp']  # name of the scp for output features (a list)
inScpFile = ['lab.scp',]                                   # name of the scp for input features (a list)

# to add features as input or output, specify scpFile and dimension 
# for example, use lab and we data as input feature:
#    inputScpFile = ['lab.scp', 'we.scp']
#    inDim = [labdim, wedim]



# ---- output config
allScp = 'all.scp'                                  # name of the output scp for generating .nc package(default)

