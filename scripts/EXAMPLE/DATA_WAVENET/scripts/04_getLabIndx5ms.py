from ioTools import readwrite as py_rw
from speechTools import htslab
import numpy as np
import multiprocessing
import os


def generateLabIndex(labfile, outfile, resolution = 80, res=50000):
    sTime, eTime, labEntry=htslab.read_full_lab(labfile, res)
    maxTime = eTime[-1] * resolution
    outBuf  = np.zeros([int(maxTime)])
    for idx in np.arange(int(eTime[-1])):
        st = idx * resolution
        et = (idx + 1) * resolution
        outBuf[st:et] = idx
    py_rw.write_raw_mat(outBuf, outfile)


def tempWarpper(fileName, labDir, outDir, resolution):
    fileName = fileName.rstrip('\n')
    inFile  =  labDir + os.path.sep + fileName + '.lab'
    outFile =  outDir + os.path.sep + fileName + '.labidx'
    generateLabIndex(inFile, outFile, resolution)


if __name__ == "__main__":

    # full_align directory
    labDir = './full_align'

    # outptu directory
    outDir = './temp'

    # resolution
    resolution = 1  

    # name list
    nameList = './temp.lst'

    if True is True:
        with open(nameList) as filePtr:
            for idx,filename in enumerate(filePtr):
                tempWarpper(filename, labDir, outDir, resolution)
                
    else:
        pool = multiprocessing.Pool(10)
        with open(nameList) as filePtr:
            for idx,filename in enumerate(filePtr):
                pool.apply_async(tempWarpper,
                                 args=(filename, labDir, outDir, resolution))
        pool.close()
        pool.join()
