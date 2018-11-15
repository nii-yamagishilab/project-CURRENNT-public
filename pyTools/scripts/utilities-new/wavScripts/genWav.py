#!/usr/bin/python
from __future__ import absolute_import
from __future__ import print_function

from speechTools import wavTool
from scipy.io import wavfile
from ioTools import readwrite as py_rw
import os
import sys
import numpy as np

dirPath = sys.argv[1]
quantiBitNum = int(sys.argv[2])
samplingRate = int(sys.argv[3])

try:
    trimLength = int(sys.argv[4])
except IndexError:
    trimLength = 0

fileList = py_rw.read_txt_list(dirPath + '/gen.scp')
for fileName in fileList:
    fileName = fileName.rstrip('\n')
    nameHtk  = dirPath + os.path.sep + os.path.basename(fileName).rstrip('.htk') + '.htk'
    nameRaw  = dirPath + os.path.sep + os.path.basename(fileName).rstrip('.htk') + '.raw'
    nameWav  = dirPath + os.path.sep + os.path.basename(fileName).rstrip('.htk') + '.wav'
    print(nameRaw, nameWav)
    data = py_rw.read_htk(nameHtk, 'f4', 'b')
    if trimLength < data.shape[0]:
        data = data[trimLength:data.shape[0]-trimLength]
        
    if quantiBitNum > 0:
        quantiLevel = np.power(2, quantiBitNum)-1
        py_rw.write_raw_mat(data, nameRaw)
        wavTool.raw2wav(nameRaw, nameWav, quantiLevel, samplingRate=samplingRate)
    else:
        wavfile.write(nameWav, samplingRate, data)
