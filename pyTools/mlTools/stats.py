#!/usr/bin/python
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
        assert 1==0,"Cant find readwrite"



def getMeanStd(fileScp, fileDim, stdFloor=0.00001):
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
            
            for t in xrange(data.shape[0]):
                if len(data.shape) == 1:
                    tmpIn = (data[t]-meanBuf)
                    meanBuf = meanBuf + tmpIn * 1.0/ (timeStep + t + 1)
                    stdBuf  = stdBuf  + tmpIn*(data[t] - meanBuf)
                else:
                    tmpIn = (data[t, :]-meanBuf)
                    meanBuf = meanBuf + tmpIn * 1.0/ (timeStep + t + 1)
                    stdBuf  = stdBuf  + tmpIn*(data[t, :] - meanBuf)
            timeStep += data.shape[0]
    sys.stdout.write('\n')
    stdBuf = np.sqrt(stdBuf/(timeStep-1))

    floorIdx = stdBuf < stdFloor
    stdBuf[floorIdx] = 1.0
    
    meanBuf = np.asarray(meanBuf, dtype=np.float32)
    stdBuf = np.asarray(stdBuf, dtype=np.float32)

    return meanBuf, stdBuf
    
def getMeanStd_merge(fileScps, fileDims, meanStdOutPath):
    assert len(fileScps) == len(fileDims), "len(fileScps) != len(fileDims)"
    
    dimSum  = np.array(fileDims).sum()
    meanStdBuf = np.zeros([dimSum * 2], dtype=np.float32)

    dimCount = 0
    for fileScp, fileDim in zip(fileScps, fileDims):
        print "Mean std on %s" % (fileScp)
        tmpM, tmpV = getMeanStd(fileScp, fileDim)
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
    
