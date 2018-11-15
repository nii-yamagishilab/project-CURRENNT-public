#!/usr/bin/python
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import numpy as np
from ioTools import readwrite as py_rw

def modulationSpectrum(data, dftL = 4096, avernum = 30):
    """
    """
    datafft = np.fft.fft(data, n = dftL)
    ms      = 2 * np.log(np.abs(datafft))
    ms      = np.concatenate((ms[avernum:0:-1], ms))
    
    if avernum > 0:
        temp    = np.zeros([dftL/2])
        for x in np.arange(avernum, dftL/2 + avernum):
            temp[x - avernum] = np.mean(ms[x-avernum:x])
        return  temp
    else:
        return ms[0:dftL/2]
    

def gv(data):
    """
    """
    data_m = data - np.mean(data)
    data_v = np.power((np.mean(np.power(data_m, 2))), 0.5)
    return data_v


if __name__ == "__main__":

    if sys.argv[1] == 'gv':
        fileDir  = sys.argv[2]
        fileList = sys.argv[3]
        fileExt  = sys.argv[4]
        fileDim  = int(sys.argv[5])
        fileOut  = sys.argv[6]

        cnt = 0
        with open(fileList, 'r') as filePtr:
            for idx, fileName in enumerate(filePtr):
                cnt = cnt + 1

        gvData = np.zeros([cnt, fileDim])
        
        cnt = 0
        with open(fileList, 'r') as filePtr:
            for idx, fileName in enumerate(filePtr):
                fileName = fileName.rstrip('\n')
                data = py_rw.read_raw_mat(fileDir + os.path.sep + fileName + fileExt, fileDim)
                if (fileExt == '.lf0' or fileExt =='.f0') and fileDim == 1:
                    data = data[data>0]
                gvData[cnt, :] = gv(data)
                cnt = cnt + 1
                #print fileName
        py_rw.write_raw_mat(gvData, fileOut + os.path.sep + 'gv.data.bin')
        print(fileOut, '\t', np.median(gvData, axis=0))
                
                
        
    

