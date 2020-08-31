#!/usr/bin/python                                       


"""                                                     
Check the statics of F0
"""

import re
import sys
import os
from ioTools import readwrite as py_rw
from speechTools import discreteF0
import numpy as np


# Configuration, same as 00_stats.py
fileScp = './file.lst'
fileDir = '../../RAWDATA'
fileExt = '.lf0_dis'
fileDim = 1

# Output directory
fileOut = '../../RAWDATA'

# Here I use the statistics from the whole Nancy corpus
# Maximum F0 value 
F0Max    = 571.0 #570.733337402
# Minimum F0 value
F0Min    = 113.0 #113.228637695
# U/V threshold. Below this value will be unvoiced
F0Zero   = 10.0
# Number of F0 events (including 1 for unvoiced)
F0Inter  = 128

#default
F0Conti  = 0

################
frameSum = 0
frameMin = 1000
frameMax = 0

if os.path.isdir(fileOut):
    pass
else:
    os.mkdir(fileOut)

with open(fileScp, 'r') as filePtr:
    for idx, fileName in enumerate(filePtr):
        fileName = fileName.rstrip('\n')
        print idx, fileName,
        filePath = fileDir + os.path.sep + fileName + fileExt
        data = py_rw.read_raw_mat(filePath, fileDim)

        if fileDim > 1:
            temp = np.zeros([data.shape[0]])
            temp = data[:,0]
            data = temp

        frameSum = frameSum + data.shape[0]

        F0Idx         = data>10
        dataClas, vuv = discreteF0.f0Conversion(data, F0Max, F0Min, F0Inter, 'c2d', F0Conti)
        dataClas[vuv<1] = 0.0
        #dataClas = np.zeros([data.shape[0]])
        #if F0Conti:
        #    # Continous F0
        #    pass
        #    dataClas[F0Idx] = np.round((data[F0Idx] - F0Min)/(F0Max - F0Min) * (F0Inter - 1))
        #else:
        #    # Discontinuous F0, leave one dimension for unvoiced
        #    dataClas[F0Idx] = np.round((data[F0Idx] - F0Min)/(F0Max - F0Min) * (F0Inter - 2)) + 1
        
        tmpmax = np.max(data[F0Idx])
        tmpmin = np.min(data[F0Idx])
        tmpmax2 = np.max(dataClas[F0Idx])
        tmpmin2 = np.min(dataClas[F0Idx])
        
        print tmpmax, tmpmin, tmpmax2, tmpmin2
        frameMax = np.max([frameMax, tmpmax])
        frameMin = np.min([frameMin, tmpmin])
        
        filePath = fileOut + os.path.sep + fileName + fileExt + '_class'
        py_rw.write_raw_mat(dataClas, filePath)

print "\nmax F0 event, min F0 event, #frame"
print frameMax, frameMin, frameSum
        
