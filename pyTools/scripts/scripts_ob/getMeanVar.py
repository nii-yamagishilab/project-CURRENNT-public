#!/usr/bin/python

from __future__ import absolute_import
from __future__ import print_function
from ioTools import readwrite as py_rw
import numpy as np
from six.moves import range


def WelFord(dataMatrix, dataMv, dataCounter):
    assert dataMatrix.shape[1] == dataMv.shape[1], 'Dimension mismatch'
    for t in range(dataMatrix.shape[0]):
        tmpBias = dataMatrix[t, : ] - dataMv[0]
        dataMv[0] = dataMv[0] + tmpBias * 1.0/(dataCounter + (t + 1))
        dataMv[1] = dataMv[1] + tmpBias * (dataMatrix[t, :] - dataMv[0])
    dataCounter += dataMatrix.shape[0]
    return dataMv, dataCounter


def temprapper(fileName, dim, dataMv, dataCounter):
    data = py_rw.read_raw_mat(fileName, dim)
    return WelFord(data, dataMv, dataCounter)


if __name__ == "__main__":

    # FileList
    fileList = '/work/smg/wang/PROJ/F0MODEL/DATA/F009/lists/split2/test.lst'
    # File directory
    fileDirs = ['/work/smg/takaki/FEAT/F009/mgc']
    # File name extension
    fileExts = ['.mgc']
    # feature Dim
    dataDims = [60]
    
    # Output file
    # The output will be binary float32,
    # it will contains a mean vector and a std vector [mean; var]
    dataOut  = './data.mv.bin'
    
    # std threshold
    # if std < threshold, set std = 1.0 (not use std for normalization)
    stdThresh = 0.0000001
    
    # If mean and std are calculated over multiple types of features,
    # please specify fileDirs, fileExts, dataDims for each type of feature.
    # The output contains a mean/std vector concatenated from all types of features

    
    dataMvBuffer = []
    
    for idx, fileDir in enumerate(fileDirs):
        fileExt = fileExts[idx]
        dataDim = dataDims[idx]
        
        dataMv   = np.zeros([2, dataDim], dtype=np.float64)
        dataCounter = 0
        with open(fileList, 'r') as filePtr:
            for fileName in filePtr:
                fileName = fileName.rstrip('\n')
                print(fileName)
                fileName = fileDir + '/' + fileName + fileExt
                dataMv, dataCounter = temprapper(fileName, dataDim,
                                                 dataMv, dataCounter)
            
                
        dataMv[1] = np.sqrt(dataMv[1]/(dataCounter-1))
        dataMv = np.asarray(dataMv, dtype=np.float32)
        dataMvBuffer.append(dataMv)
    dataMvBuffer = np.concatenate(dataMvBuffer, axis=1)
    dataMvBuffer[1,np.where(dataMvBuffer[1,:] < stdThresh)] = 1.0
    py_rw.write_raw_mat(dataMvBuffer, dataOut)
    
    
