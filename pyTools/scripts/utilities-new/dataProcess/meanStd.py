#!/usr/bin/python
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
from   mlTools import stats
from   ioTools import readwrite as py_rw
import numpy as np

def meanStdNorm(fileScps, fileDims, meanStdOutPath):
    """ Calculate the mean and std from a list of file lists
        fileScpsString: 'file_list_1,file_list_2,...'
        fileDimsString: 'feat_dim_1,feat_dim_2,...'
        meanStdOutPat:  path to save the mean_std vector [mean, std]
    """    
    stats.getMeanStd_merge(fileScps, fileDims, meanStdOutPath)


def meanStdNormMask(fileScps, fileDims, fileNormMask, meanStdOutPath, f0Dim=-1):
    """
    """
    assert len(fileDims) == len(fileNormMask), \
        "Unequal length feature dim & norm mask"
    
    # calcualte the mean/std
    stats.getMeanStd_merge(fileScps, fileDims, meanStdOutPath + '.unmasked', f0Dim)

    meanStdData = py_rw.read_raw_mat(meanStdOutPath + '.unmasked', 1)
    
    assert meanStdData.shape[0] == sum(fileDims) * 2, \
        "%s dimension not %d" % (meanStdOutPath + '.unmasked', sum(fileDims) * 2)

    featDims = []
    startDim = 0
    for dim in fileDims:
        featDims.append([startDim, startDim + dim])
        startDim = startDim + dim
    
    for dimRange, normMask in zip(featDims, fileNormMask):
        if len(normMask) == 0:
            pass
        elif len(normMask) == 1 and (normMask[0] == 0 or normMask[0]=='not_norm'):
            print("mean is set to 0 and std is set to 1")
            meanStdData[dimRange[0]:dimRange[1]] = 0.0
            meanStdData[dimRange[0]+sum(fileDims):dimRange[1]+sum(fileDims)] = 1.0
        elif len(normMask) == 2:
            assert dimRange[0] <= normMask[0], 'normMask range error' % (str(normMask))
            assert dimRange[1] >= normMask[1], 'normMask range error' % (str(normMask))
            meanStdData[normMask[0]:normMask[1]] = 0.0
            meanStdData[normMask[0]+sum(fileDims):normMask[1]+sum(fileDims)] = 1.0
        else:
            print("Wrong format of NormMask %s" % (str(normMask)))
        print('normmask %s' % (str(normMask)))
        
    py_rw.write_raw_mat(meanStdData, meanStdOutPath)
    
if __name__ == "__main__":
    """ 
    """
    fileScps = sys.argv[1]
    fileDims = sys.argv[2]
    fileNormMask = sys.argv[3]
    mvOutputPath = sys.arv[4]

    fileScps = fileScps.split(',')
    fileDims = [int(x) for x in fileDims.split('_')]
    fileNormMask = [int(x) for x in fileNormMask.split('_')]
    meanStdNormMask(fileScps, fileDims, fileNormMask, mvOutputPath)
