#!/usr/bin/python
from __future__ import absolute_import
from __future__ import print_function

from ioTools import readwrite as py_rw
import numpy as np
import os
import sys



def split(fileName, inDir, uvDir, uvT, f0Ext, vuExt):
    conlf0Name = inDir + os.path.sep + fileName + f0Ext
    vuName     = uvDir + os.path.sep + fileName + vuExt
    print(fileName, end=' ')
    if os.path.isfile(conlf0Name) and os.path.isfile(vuName):
        conlf0 = py_rw.read_raw_mat(conlf0Name, 1)
        vu     = py_rw.read_raw_mat(vuName, 1)
        assert conlf0.shape[0] == vu.shape[0], ': lf0 uv unequal length'
        conlf0[vu < uvT] = -1.0e+10
        py_rw.write_raw_mat(conlf0, conlf0Name)
        print(': done')
    else:
        print(': not found')
    

if __name__ == "__main__":
    fileDir = sys.argv[1]
    if len(sys.argv) > 2:
        uvT = float(sys.argv[2])
    else:
        uvT = 0.5

    if len(sys.argv) > 3:
        f0Ext = sys.argv[3]
    else:
        f0Ext = '.lf0'

    if len(sys.argv) > 4:
        vuExt = sys.argv[4]
    else:
        vuExt = '.vuv'

    if len(sys.argv) > 5:
        uvDir = sys.argv[5]
    else:
        uvDir = fileDir
        
    fileNames  = os.listdir(fileDir)
    for fileName in [x for x in fileNames if x.endswith(f0Ext)]:
        fileName = fileName.split(f0Ext)[0]
        split(fileName, fileDir, uvDir, uvT, f0Ext, vuExt)
