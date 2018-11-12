#!/usr/bin/python

from __future__ import absolute_import
from ioTools import readwrite as py_rw
import numpy as np
import multiprocessing
import os
import sys
from speechTools import discreteF0


def f0convert(f0File, qF0Output, vuvOutputFile, f0Zero, f0Max, f0Min, f0Inter, f0Conti, f0Type):
    if f0Type == 0:
        data  = py_rw.read_raw_mat(f0File, 1)
        idx   = data > 0
        data[idx] = 1127.0 * np.log(data[idx]/700.0 + 1)
    elif f0Type == 1:
        data  = py_rw.read_raw_lf0(f0File, 1)
        idx   = data > 0
    F0Idx = data>f0Zero
    dataClas, vuv = discreteF0.f0Conversion(data.copy(), f0Max, f0Min, f0Inter, 'c2d', f0Conti)
    dataClas[vuv<1] = 0.0
    py_rw.write_raw_mat(dataClas, qF0Output)
    py_rw.write_raw_mat(vuv, vuvOutputFile)


def tempWarpper(fileName, f0Dir, f0OutDir, f0Type, f0Zero, f0Max, f0Min, f0Inter, f0Conti):
    fileName          = fileName.rstrip('\n')
    fileDir           = os.path.dirname(fileName)
    fileBase          = os.path.basename(fileName)
    fileBase, fileExt = os.path.splitext(fileBase)

    inFile    = fileName
    if f0Type == 0:
        f0File    = f0Dir + os.path.sep + fileBase + os.path.extsep  + 'f0'
        qF0Output = f0OutDir + os.path.sep + fileBase + os.path.extsep + 'qf0'
        vuvOutFile= f0OutDir + os.path.sep + fileBase + os.path.extsep + 'vuv'
        f0convert(f0File, qF0Output, vuvOutFile, f0Zero, f0Max, f0Min, f0Inter, f0Conti, f0Type)

    if f0Type == 1:
        f0File    = f0Dir + os.path.sep + fileBase + os.path.extsep + 'lf0'
        qF0Output = f0OutDir + os.path.sep + fileBase + os.path.extsep + 'qf0'
        vuvOutFile= f0OutDir + os.path.sep + fileBase + os.path.extsep + 'vuv'
        f0convert(f0File, qF0Output, vuvOutFile, f0Zero, f0Max, f0Min, f0Inter, f0Conti, f0Type)


if __name__ == "__main__":
    fileScp = sys.argv[1]
    f0Dir   = sys.argv[2]
    f0OutDir= sys.argv[3]
    f0Type  = int(sys.argv[4])
    
    f0Zero  = float(sys.argv[5])
    f0Max   = float(sys.argv[6])
    f0Min   = float(sys.argv[7])
    f0Inter = int(sys.argv[8])
    f0Conti = int(sys.argv[9])
    
    with open(fileScp) as filePtr:
        for idx, filename in enumerate(filePtr):
            tempWarpper(filename, f0Dir, f0OutDir, f0Type, f0Zero, f0Max, f0Min, f0Inter, f0Conti)
    
