import numpy as np
import scipy.io.wavfile
from ioTools import readwrite as py_rw
import multiprocessing
import os
from speechTools import wavTool


def tempWarpper(fileIn, outFile, outWav, quanLevel):
    sr, wavdata = scipy.io.wavfile.read(inFile)
    if wavdata.dtype == np.int16:
        transData = wavTool.wavformConvert(wavdata, 16, True, quanLevel)
    elif wavdata.dtype == np.int32:
        transData = wavTool.wavformConvert(wavdata, 32, True, quanLevel)
    else:
        print "Unsupported data type"
    py_rw.write_raw_mat(transData, outFile)

    if outWav is not None:
        recoData  = wavTool.wavformDeconvert(transData, quanLevel)
        wavTool.waveSaveFromFloat(recoData, outWav, sr=sr)


if __name__ == "__main__":

    # Level of quantization
    muQuanLevel = 1024 - 1
    # Wavefomr directory
    wavDir = './wav16k' 
    # output file directory
    outDir =  './temp'
    # name list
    nameList = './file.lst'
    
    if True is True:
        # Test sample
        with open('./temp.lst') as filePtr:
            for idx,fileName in enumerate(filePtr):
                fileName = fileName.rstrip('\n')
                inFile  = wavDir + os.path.sep + fileName + '.wav'
                outFile = outDir + os.path.sep + fileName + '.raw'
                outWav  = outDir + os.path.sep + fileName + '.wav'
                tempWarpper(inFile, outFile, outWav, muQuanLevel)
                print inFile
    else:
        # batch processing
        pool = multiprocessing.Pool(5)
        with open(nameList) as filePtr:
            for idx, filename in enumerate(filePtr):
                fileName = fileName.rstrip('\n')
                inFile  = wavDir + os.path.sep + fileName + '.wav'
                outFile = outDir + os.path.sep + fileName + '.raw'
                pool.apply_async(tempWarpper,
                                 args=(inFile,
                                       outFile,
                                       None,
                                       muQuanLevel))
        pool.close()
        pool.join()
