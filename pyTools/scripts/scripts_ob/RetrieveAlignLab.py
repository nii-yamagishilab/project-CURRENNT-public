#!/usr/bin/python

import scipy
from scipy import io
import os
from ioTools import readwrite as py_rw
import numpy as np
 
ncFile = "/home/smg/wang/PROJ/DL/RNNJP/DATA/test_align/F009A/data.nc1"
labdir = "/home/smg/takaki/FEAT/F009/data/ver01/full"
labout = "/home/smg/wang/DATA/speech/F009A/nndata/labels/full_align/test_set"
prefix = "ATR_Ximera_F009A_"
resolu = 50000

ncData = io.netcdf_file(ncFile, 'r')
sentNm = ncData.dimensions['numSeqs']
sentNa = ncData.variables['seqTags'][:].copy()
sentTi = ncData.variables['seqLengths'][:].copy()

start = 0
for id, sentId in enumerate(sentNa):
    sentId = ''.join(sentId)
    labinpfile = labdir+os.path.sep+sentId+'.lab'
    laboutfile = labout+os.path.sep+sentId+'.lab'
    labentrys  = py_rw.read_txt_list(labinpfile)
    stime, etime = start, start+sentTi[id]
    data = ncData.variables['inputs'][stime:etime, 0:-3].copy()
    data = (data*data).sum(axis=1)
    difd = np.diff(data)
    indx = np.concatenate((np.array([0]), np.argwhere(difd).flatten(), np.array([etime])))
    if len(indx)==len(labentrys)+1:
        temp = ''
        for x in xrange(len(labentrys)):
            st = indx[x]*resolu
            et = indx[x+1]*resolu
            lab = labentrys[x].split()
            temp += "%d %d %s\n" % (st, et, lab[2])
        fil = open(laboutfile, 'w')
        fil.write(temp[0:-1])
        fil.close()
        print "Writing to %s" % (laboutfile)
    else:
        print "Unmatched %s" % (labinpfile)
    
    start = etime

ncData.close()
