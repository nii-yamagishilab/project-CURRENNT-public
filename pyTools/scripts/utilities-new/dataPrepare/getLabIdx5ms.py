#!/usr/bin/python


import numpy as np
import multiprocessing
import os
import sys

try:
    from binaryTools import readwriteC2 as py_rw
except ImportError:
    try:
        from binaryTools import readwriteC2_220 as py_rw
    except ImportError:
        try: 
            from ioTools import readwrite as py_rw
        except ImportError:
            print "Please add pyTools to PYTHONPATH"
            raise Exception("Can't not import binaryTools/readwriteC2 or ioTools/readwrite")
        
def generateLabIndex(labfile, outfile, featDim):
    if os.path.isfile(labfile):
        labFile = py_rw.read_raw_mat(labfile, featDim)
        outBuf = np.arange(labFile.shape[0])
        py_rw.write_raw_mat(outBuf, outfile)
    else:
        print "Not found %s" % (labfile)
    

def tempWarpper(fileName, labDir, labExt, outDir, outExt, featDim):
    fileName = fileName.rstrip('\n')
    inFile  =  labDir + os.path.sep + fileName + '.' + labExt
    outFile =  outDir + os.path.sep + fileName + '.' + outExt
    generateLabIndex(inFile, outFile, featDim)


if __name__ == "__main__":

    # full_align directory
    labDir = sys.argv[1]
    labExt = sys.argv[2]
    featDim = int(sys.argv[3])
    
    # outptu directory
    outDir = sys.argv[4]
    outExt = sys.argv[5]

    # file list
    nameList = sys.argv[6]

    if True is False:
        with open(nameList) as filePtr:
            for idx,filename in enumerate(filePtr):
                tempWarpper(filename, labDir, labExt, outDir, outExt, featDim)
                
    else:
        pool = multiprocessing.Pool(10)
        with open(nameList) as filePtr:
            for idx,filename in enumerate(filePtr):
                pool.apply_async(tempWarpper,
                                 args=(filename, labDir, labExt, outDir, outExt, featDim))
        pool.close()
        pool.join()
