#!/usr/bin/python
""" This script extract the raw data from data.nc file
"""
from __future__ import absolute_import
from __future__ import print_function
import scipy
import re, os
from scipy import io
import numpy as np
from ioTools import readwrite as py_rw
from six.moves import range

#####  configuration #########
# what is the parent director of the data.nc
dataDir     = '/work/smg/wang/PROJ/WE/DNNAM/DATA/nancy/nancy_all'

# where to store the extracted data (output directory)
dataOut     = '/work/smg/wang/DATA/speech/nancy/nndata/iplf0/nancy'

# where is the mean and variance data.mv ? set dataMV = None is you don't need to 
#  de-normalize the data
dataMV      = '/work/smg/wang/PROJ/WE/DNNAM/DATA/nancy/nancy_all/'+'data.mv'

dim         = [[181,182],]     # which dimension of the data to be extracted ?
name        = ['.lf0',]        # which suffix to name the extracted data ?
inOutData   = 2                # 1: input data, 
                               # 2: output data
dataPattern = 'data.nc1$'  # the regular pattern to identify the specific data.nc


#####  data extraction #########

# read in the mean and variance
if dataMV is not None:
    mv      = io.netcdf_file(dataMV,'r')
    if inOutData == 1:
        meanVec = mv.variables['inputMeans'][:].copy()
        varVec  = mv.variables['inputStdevs'][:].copy()
    else:
        meanVec = mv.variables['outputMeans'][:].copy()
        varVec  = mv.variables['outputStdevs'][:].copy()
    mv.close()
else:
    print('dataMV is not specified. Extracted data will be normalized data.\n')

# 
dataList = os.listdir(dataDir)
for dataFile in dataList:
    if re.search(dataPattern, dataFile):
        print(dataFile)
if os.path.isdir(dataOut):
    pass
else:
    os.mkdir(dataOut)


for dataFile in dataList:
    if re.search(dataPattern, dataFile):
        data = io.netcdf_file(dataDir+os.path.sep+dataFile)
        uttNum = data.dimensions['numSeqs']
        seqLengths = data.variables['seqLengths'][:].copy()
        seqLengths = np.concatenate((np.array([0]), seqLengths)).cumsum()
        seqTags   = data.variables['seqTags'][:]
        if inOutData == 1:
            dataAll = data.variables['inputs'][:]
        else:
            dataAll = data.variables['targetPatterns'][:]
        for i in range(uttNum):
            outName = dataOut+os.path.sep+''.join(seqTags[i])

            for j, suf in enumerate(name):
                outFile = outName + suf
                tmpdata = dataAll[seqLengths[i]:seqLengths[i+1],dim[j][0]:dim[j][1]].copy()
                if dataMV is not None:
                    tmpdata = tmpdata*varVec[dim[j][0]:dim[j][1]]+meanVec[dim[j][0]:dim[j][1]]
                py_rw.write_raw_mat(tmpdata, outFile)
            print("%s Utt %d" % (dataFile, i))
        del dataAll, seqTags, seqLengths, uttNum
        data.close()

