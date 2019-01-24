#!/usr/bin/python

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
        elif len(normMask) == 1 and normMask[0] == 0:
            meanStdData[dimRange[0]:dimRange[1]] = 0.0
            meanStdData[dimRange[0]+sum(fileDims):dimRange[1]+sum(fileDims)] = 1.0
        elif len(normMask) == 2:
            assert dimRange[0] <= normMask[0], 'normMask range error' % (str(normMask))
            assert dimRange[1] >= normMask[1], 'normMask range error' % (str(normMask))
            meanStdData[normMask[0]:normMask[1]] = 0.0
            meanStdData[normMask[0]+sum(fileDims):normMask[1]+sum(fileDims)] = 1.0
        else:
            print "Wrong format of NormMask %s" % (str(normMask))
        print 'normmask %s' % (str(normMask))
        
    py_rw.write_raw_mat(meanStdData, meanStdOutPath)

        
if __name__ == "__main__":
    """ python meanStd_wrap_simple.py data_dir data_ext data_dim output_mv_path
          data_dir: path to the directory of data
          data_ext: file name extension of the data  *.data_ext
          data_dim: dimension of the feature vector in one frame
          output_mv_path: path to write the mean/std vector
    """ 
    #dataLst = sys.argv[1]
    acousDirs = sys.argv[1]
    acousExts = sys.argv[2]
    acousDims = sys.argv[3]
    mvoutputPath = sys.argv[4]

    normMask = None
    f0Ext = None
    dataLstDir = './tmp'
    acousDirList = acousDirs.split(',')
    acousExtList = acousExts.split(',')
    acousDimList = [int(x) for x in acousDims.split('_')]
    normMaskList = [[] for dimCnt in acousDimList]

    assert len(acousDirList) == 1, "Error: please check usage"
    assert len(acousDirList) == len(acousExtList), "Error: unequal length of acousDirs, acousExts"
    assert len(acousExtList) == len(acousDimList), "Error: unequal length of acousDims, acousExts"
    assert len(acousExtList) == len(normMaskList), "Error: unequal length of acousDims, normmask"

    fileListsBuff = []
    dimCnt = 0
    f0Dim = -1

    dataFileList = os.listdir(acousDirList[0])
    dataFileList = [x for x in dataFileList if x.endswith(acousExtList[0])]
    
    for acousDir, acousExt, acousDim in zip(acousDirList, acousExtList, acousDimList):

        # confirm the F0 dimension
        if acousExt == f0Ext:
            f0Dim = dimCnt

        # clearn the extension
        if acousExt.startswith('.'):
            acousExt = acousExt[1:]

        # write the file script
        fileOutput = dataLstDir + acousExt + '.scp'
        fileListsBuff.append(fileOutput)
        writePtr = open(fileOutput, 'w')
        for line in dataFileList:
            filename = os.path.splitext(line)[0]
            writePtr.write('%s/%s.%s\n' % (acousDir, filename, acousExt))
        writePtr.close()

        dimCnt = dimCnt + acousDim
    
    
    meanStdNormMask(fileListsBuff, acousDimList, normMaskList, mvoutputPath,
                                f0Dim = f0Dim)
    
    meanstd_data = py_rw.read_raw_mat(mvoutputPath, 1)
    for fileName in fileListsBuff:
        os.system("rm %s" % (fileName))

