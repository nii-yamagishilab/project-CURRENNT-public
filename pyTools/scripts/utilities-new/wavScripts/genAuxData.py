#!/usr/bin/python

from __future__ import absolute_import
from __future__ import print_function

from ioTools import readwrite as py_rw
from speechTools import htslab
import numpy as np
import multiprocessing
import os
import sys

def generateData(inFile, outFile, vuvInFile, vuvOutFile, featDim, resolution, thre=0.5):
    if not os.path.isfile(inFile):
        print(inFile)
        return
    if not os.path.isfile(vuvInFile):
        print(vuvInFile)
        return
    
    labData = py_rw.read_raw_mat(inFile, featDim)
    vuvData = py_rw.read_raw_mat(vuvInFile, 1)
    if labData.shape[0] > vuvData.shape[0]:
        labData = labData[0:vuvData.shape[0],:]
    else:
        vuvData = vuvData[0:labData.shape[0]]
    assert labData.shape[0] == vuvData.shape[0], 'Unequal length vuv and lab'
    
    maxTime    = labData.shape[0] * resolution
    labIdxBuf  = np.zeros([int(maxTime)])
    vuvBinBuf  = np.zeros([int(maxTime)])
    for idx in np.arange(labData.shape[0]):
        st = idx * resolution
        et = (idx + 1) * resolution
        labIdxBuf[st:et] = idx
        vuvBinBuf[st:et] = vuvData[idx]
        
        randU = np.random.rand(et-st)
        temp = vuvBinBuf[st:et]
        temp[randU < thre] = 0
        vuvBinBuf[st:et] = temp
        
    py_rw.write_raw_mat(labIdxBuf, outFile)
    py_rw.write_raw_mat(vuvBinBuf, vuvOutFile)


def tempWarpper(fileName, featDim, resolution, vuvDir, outDir, thre):
    fileName          = fileName.rstrip('\n')
    fileDir           = os.path.dirname(fileName)
    fileBase          = os.path.basename(fileName)
    fileBase, fileExt = os.path.splitext(fileBase)

    inFile    = fileName
    outFile   = outDir + os.path.sep + fileBase + os.path.extsep + 'labidx'
    vuvInFile = vuvDir + os.path.sep + fileBase + os.path.extsep + 'vuv'
    vuvOutFile= outDir + os.path.sep + fileBase + os.path.extsep + 'bin'
    generateData(inFile, outFile, vuvInFile, vuvOutFile, featDim, resolution, thre)


if __name__ == "__main__":
    labFile = sys.argv[1]
    vuvDir =  sys.argv[2]
    resolution=int(sys.argv[3])
    featDim=int(sys.argv[4])
    outDir =  sys.argv[5]
    threshold = float(sys.argv[6])
    
    with open(labFile) as filePtr:
        for idx, filename in enumerate(filePtr):
            tempWarpper(filename, featDim, resolution, vuvDir, outDir, threshold)
    
