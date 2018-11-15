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


def tempWarpper(fileIn, outFile, outWav, quanLevel):
    sr, wavdata = scipy.io.wavfile.read(fileIn)
    
    if wavdata.dtype == np.int16:
        transData = wavTool.wavformConvert(wavdata, 16, True, quanLevel)
    elif wavdata.dtype == np.int32:
        transData = wavTool.wavformConvert(wavdata, 32, True, quanLevel)
    else:
        print("Unsupported data type")
    py_rw.write_raw_mat(transData, outFile)
    
    if outWav is not None:
        recoData  = wavTool.wavformDeconvert(transData, quanLevel)
        wavTool.waveSaveFromFloat(recoData, outWav, sr=sr)


if __name__ == "__main__":

    # File name list
    nameList = sys.argv[1]

    # input file waveform directories
    wavDir = sys.argv[2]

    # output file directories
    outDir = sys.argv[3]
    
    # number of bits for quantization
    #  Level of quantization (2 ^ N - 1), N is the number of bits
    muQuanLevel = np.power(2, int(sys.argv[4])) - 1
    
    outExt =  '.bin'

    
    try:
        os.mkdir(outDir)
    except OSError:
        pass
        
    
    if True is False:
        # debug
        with open(nameList) as filePtr:
            for idx,fileName in enumerate(filePtr):
                fileName = fileName.rstrip('\n')
                inFile  = wavDir + os.path.sep + fileName + '.wav'
                outFile = outDir + os.path.sep + fileName + outExt
                outWav  = outDir + os.path.sep + fileName + '.wav'
                tempWarpper(inFile, outFile, outWav, muQuanLevel)
                print(inFile)
                break
    else:
        # batch processing
        pool = multiprocessing.Pool(10)
        with open(nameList) as filePtr:
            for idx, fileName in enumerate(filePtr):
                fileName = fileName.rstrip('\n')
                inFile  = wavDir + os.path.sep + fileName + '.wav'
                outFile = outDir + os.path.sep + fileName + outExt
                pool.apply_async(tempWarpper,
                                 args=(inFile, outFile, None, muQuanLevel))
        pool.close()
        pool.join()
