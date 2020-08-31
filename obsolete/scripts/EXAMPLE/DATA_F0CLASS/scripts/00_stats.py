#!/usr/bin/python                                       


"""                                                     
Check the statics of F0
"""

import re
import sys
import os
from ioTools import readwrite as py_rw
import numpy as np

# List of file name
fileScp  = './file.lst'
# Directory of the F0 data 
fileDir  = '../../RAWDATA/'
# File extention
fileExt  = '.lf0_dis'
# Dimension of the bindary raw F0 data
fileDim  = 1
# U/V threshold
F0Zero   = 10.0


###
frameSum = 0
frameMin = 1000
frameMax = 0

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
        data = data[data>F0Zero]
        tmpmax = np.max(data)
        tmpmin = np.min(data)
        print tmpmax, tmpmin
        frameMax = np.max([frameMax, tmpmax])
        frameMin = np.min([frameMin, tmpmin])


print "\nPlease use the information below for F0 quantization"
print "maxF0, minF0, #frame"
print frameMax, frameMin, frameSum
        
