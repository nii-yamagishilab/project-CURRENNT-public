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
    """ python meanStd_wrap.py file_name.lst feature1_dir,feature2_dir,... feature1_name_extension,feature2_name_extension, 
                feature1dim_feature2dim_feature3dim normMask(simply set to None) extension_of_f0_file_name 
                output_directory_to_write_file_lst output_path_to_write_mean_std
    """ 
    dataLst = sys.argv[1]
    acousDirs = sys.argv[2]
    acousExts = sys.argv[3]
    acousDims = sys.argv[4]
    normMask = sys.argv[5]
    f0Ext = sys.argv[6]
    dataLstDir = sys.argv[7]
    mvoutputPath = sys.argv[8]
    
    acousDirList = acousDirs.split(',')
    acousExtList = acousExts.split(',')
    acousDimList = [int(x) for x in acousDims.split('_')]
    try:
        normMaskList = [[int(x)] for x in normMask.split('_')]
    except ValueError:
        # by default, normlize every feature dimension
        normMaskList = [[] for dimCnt in acousDimList]
        
    assert len(acousDirList) == len(acousExtList), "Error: unequal length of acousDirs, acousExts"
    assert len(acousExtList) == len(acousDimList), "Error: unequal length of acousDims, acousExts"
    assert len(acousExtList) == len(normMaskList), "Error: unequal length of acousDims, normmask"

    fileListsBuff = []
    dimCnt = 0
    f0Dim = -1
    for acousDir, acousExt, acousDim in zip(acousDirList, acousExtList, acousDimList):

        # confirm the F0 dimension
        if acousExt == f0Ext:
            f0Dim = dimCnt

        # clearn the extension
        if acousExt.startswith('.'):
            acousExt = acousExt[1:]

        # write the file script
        fileOutput = dataLstDir + os.path.sep + acousExt + '.scp'
        fileListsBuff.append(fileOutput)
        writePtr = open(fileOutput, 'w')
        with open(dataLst, 'r') as readfilePtr:
            for line in readfilePtr:
                filename = line.rstrip('\n')
                writePtr.write('%s/%s.%s\n' % (acousDir, filename, acousExt))
        writePtr.close()

        dimCnt = dimCnt + acousDim
    
    
    meanStdNormMask(fileListsBuff, acousDimList, normMaskList, mvoutputPath,
                                f0Dim = f0Dim)
    
    meanstd_data = py_rw.read_raw_mat(mvoutputPath, 1)
    if f0Dim >= 0:
        print "Please note:"
        print "F0 mean: %f" % (meanstd_data[f0Dim])
        print "F0 std: %f" % (meanstd_data[dimCnt+f0Dim])


