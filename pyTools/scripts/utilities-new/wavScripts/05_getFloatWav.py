#!/usr/bin/python

from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import scipy.io.wavfile
from ioTools import readwrite as py_rw
import multiprocessing
import os
import sys
from speechTools import wavTool


def tempWarpper(fileIn, outFile, outWav):
    sr, wavdata = scipy.io.wavfile.read(fileIn)
    if wavdata.dtype == np.int16:
        #print outFile
        transData = np.array(wavdata, dtype=np.float32) / np.power(2.0, 16-1)
        py_rw.write_raw_mat(transData, outFile)
    else:
        print("Unsupported data type")

    

if __name__ == "__main__":

    # File name list
    nameList = sys.argv[1]

    # input file waveform directories
    wavDir = sys.argv[2]

    # output file directories
    outDir = sys.argv[3]
    
    outExt =  '.bin'

    #d
    debug = False
    try:
        os.mkdir(outDir)
    except OSError:
        pass
    
    if debug:
        # debug
        with open(nameList) as filePtr:
            for idx,fileName in enumerate(filePtr):
                fileName = fileName.rstrip('\n')
                inFile  = wavDir + os.path.sep + fileName + '.wav'
                outFile = outDir + os.path.sep + fileName + outExt
                outWav  = outDir + os.path.sep + fileName + '.wav'
                tempWarpper(inFile, outFile, outWav)
                print(inFile)
    else:
        # batch processing
        pool = multiprocessing.Pool(10)
        with open(nameList) as filePtr:
            for idx, fileName in enumerate(filePtr):
                fileName = fileName.rstrip('\n')
                inFile  = wavDir + os.path.sep + fileName + '.wav'
                outFile = outDir + os.path.sep + fileName + outExt
                pool.apply_async(tempWarpper,
                                 args=(inFile,outFile,None))
        pool.close()
        pool.join()
