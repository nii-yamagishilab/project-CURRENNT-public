#!/usr/bin/python
from __future__ import absolute_import
from __future__ import print_function
import scipy
import os, sys
from scipy import io
import numpy as np
import time

try:
    from binaryTools import readwriteC2 as py_rw
except ImportError:
    try:
        from binaryTools import readwriteC2_220 as py_rw
    except ImportError:
        try: 
            from ioTools import readwrite as py_rw
        except ImportError:
            print("Please add pyTools to PYTHONPATH")
            raise Exception("Can't not import binaryTools/readwriteC2 or ioTools/readwrite")




def getMeanStd(fileScp, fileDim, stdFloor=0.00001, f0Feature=0):
    """ Calculate the mean and std from a list of files
    """
    meanBuf = np.zeros([fileDim], dtype=np.float64)
    stdBuf  = np.zeros([fileDim], dtype=np.float64)
    timeStep = 0
    fileNum = sum(1 for line in open(fileScp))
    
    with open(fileScp, 'r') as filePtr:
        for idx, fileName in enumerate(filePtr):
            fileName = fileName.rstrip('\n')
            data = py_rw.read_raw_mat(fileName, fileDim)            
                
            sys.stdout.write('\r')
            sys.stdout.write("%d/%d" % (idx, fileNum))

            if f0Feature and fileDim == 1:
                # if this is F0 feature, remove unvoiced region
                data = data[np.where(data>0)]
                if data.shape[0] < 1:
                    continue
                
            # parallel algorithm
            # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
            dataCount = data.shape[0]
            if len(data.shape) == 1:
                meanNew = data.mean()
                stdNew = data.var()
            else:
                meanNew = data.mean(axis=0)
                stdNew = data.var(axis=0)
                
            deltaMean = meanNew - meanBuf
            meanBuf = meanBuf + deltaMean * (float(dataCount) / (timeStep + dataCount))
            
            if timeStep == 0:
                if len(data.shape) == 1:
                    stdBuf[0] = stdNew
                else:
                    stdBuf = stdNew
            else:
                stdBuf = (stdBuf * (float(timeStep) / (timeStep + dataCount)) +
                          stdNew * (float(dataCount)/ (timeStep + dataCount)) +
                          deltaMean * deltaMean  / (float(dataCount)/timeStep +
                                                    float(timeStep)/dataCount + 2.0))
            
            timeStep += data.shape[0]
    sys.stdout.write('\n')
    stdBuf = np.sqrt(stdBuf)

    floorIdx = stdBuf < stdFloor
    stdBuf[floorIdx] = 1.0
    
    meanBuf = np.asarray(meanBuf, dtype=np.float32)
    stdBuf = np.asarray(stdBuf, dtype=np.float32)

    return meanBuf, stdBuf
    
def getMeanStd_merge(fileScps, fileDims, meanStdOutPath, f0Dim=-1):
    assert len(fileScps) == len(fileDims), "len(fileScps) != len(fileDims)"
    
    dimSum  = np.array(fileDims).sum()
    meanStdBuf = np.zeros([dimSum * 2], dtype=np.float32)

    dimCount = 0
    for fileScp, fileDim in zip(fileScps, fileDims):
        print("Mean std on %s" % (fileScp))
        if f0Dim == dimCount:
            # this is the feature dimension for F0, don't count unvoiced region
            tmpM, tmpV = getMeanStd(fileScp, fileDim, f0Feature=1)
        else:
            tmpM, tmpV = getMeanStd(fileScp, fileDim, f0Feature=0)
            
        meanStdBuf[dimCount:dimCount+fileDim] = tmpM
        meanStdBuf[(dimSum + dimCount):(dimSum+dimCount+fileDim)]  = tmpV
        dimCount = dimCount + fileDim

    py_rw.write_raw_mat(meanStdBuf, meanStdOutPath)
        
    
    
if __name__ == "__main__":
    fileScpsString = sys.argv[1]
    fileDimsString = sys.argv[2]
    meanStdOutPath = sys.argv[3]

    fileScps = fileScpsString.split(',')
    fileDims = [int(x) for x in fileDimString.split(',')]

    getMeanStd_merge(fileScps, fileDims, meanStdOutPath)
    
