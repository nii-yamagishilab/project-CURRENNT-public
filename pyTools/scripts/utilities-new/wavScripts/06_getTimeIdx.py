#!/usr/bin/python
from __future__ import absolute_import
from __future__ import print_function

from ioTools import readwrite as py_rw
from speechTools import htslab
import numpy as np
import multiprocessing
import os
import sys

def generateLabIndex(labfile, outfile, labDim, resolution):
    data = py_rw.read_raw_mat(labfile, labDim)
    
    maxTime = data.shape[0] * resolution
    
    outBuf  = np.zeros([int(maxTime)])
    
    for idx in np.arange(int(data.shape[0])):
        st = idx * resolution
        et = (idx + 1) * resolution
        outBuf[st:et] = idx
    py_rw.write_raw_mat(outBuf, outfile)


def tempWarpper(fileName, labDir, labDim, featExt, outDir, resolution):
    fileName = fileName.rstrip('\n')
    if featExt.startswith('.'):
        inFile  =  labDir + os.path.sep + fileName + featExt
    else:
        inFile  =  labDir + os.path.sep + fileName + '.' + featExt
        
    outFile =  outDir + os.path.sep + fileName + '.bin'
    
    if os.path.isfile(inFile):
        #print fileName
        generateLabIndex(inFile, outFile, labDim, resolution)
    else:
        print("Skip:" + fileName)


if __name__ == "__main__":

    featDir = sys.argv[1]
    outDir  = sys.argv[2]
    fileLst = sys.argv[3]
    featDim = int(sys.argv[4])
    featExt = sys.argv[5]
    resolution = int(sys.argv[6])
    
    
    try:
        os.mkdir(outDir)
    except OSError:
        pass
    pool = multiprocessing.Pool(10)
    with open(fileLst) as filePtr:
        for idx,filename in enumerate(filePtr):
            pool.apply_async(tempWarpper,
                             args=(filename, featDir, featDim, featExt, outDir, resolution))
    pool.close()
    pool.join()
