#!/usr/bin/python
""" This script extract the phoneme labels from data.nc
    
"""
from __future__ import absolute_import
from __future__ import print_function
import scipy
import re, os
from   scipy   import io
import numpy   as np
from   ioTools import readwrite as py_rw

#####  configuration #########
# what is the parent director of the data.nc
dataDir     = '/work/smg/wang/PROJ/DL/RNNJP/DATA/test_align/F009A'

# where to store the extracted data
dataOut     = '/work/smg/wang/DATA/speech/F009A/testset/full_align'

# where is the mean and variance data.mv ? set dataMV = None is you don't need to 
#  de-normalize the data
dataMV      = '/work/smg/wang/PROJ/DL/RNNJP/DATA/test_align/F009A/data.nc1'

# phone list
# JVOICE
phonList    = ['xx', 'A','E','I','N','O','U','a','b','by','ch','cl','d','dy','e','f','g','gy','h','hy','i','j','k','ky','m','my','n','ny','o','p','pau','py','r','ry','s','sh','sil','t','ts','ty','u','v','w','y','z']
#  actually, pau and sil should be considered as the same silent symbol
phonList    = ['xx', 'A','E','I','N','O','U','a','b','by','ch','cl','d','dy','e','f','g','gy','h','hy','i','j','k','ky','m','my','n','ny','o','p','#','py','r','ry','s','sh','#','t','ts','ty','u','v','w','y','z']

# 
dim         = [[0, 220],]     # which dimension of the data to be extracted ?
name        = ['.lab',]        # which suffix to name the extracted data ?
inOutData   = 1                # 1: input data, 
                               # 2: output data
dataPattern = 'data.nc1$'    # the regular pattern to identify the specific data.nc
resolu      = 50000

#
phoneContext= [[0,44], [44,88], [88,132], [132,176], [176,220]] # the start-end dimension for the phonetic context
phoneDim    = 45

##### function

def genLabel(bdata, outfile):
    phoneBag    = []
    quinphoneID = np.zeros([bdata.shape[0]])
    for idx, phone in enumerate(phoneContext):
        phonemat = bdata[:, phone[0]:phone[1]]>0.5
        maxindex = np.argmax(phonemat, axis=1)
        claimed0 = (maxindex == 0) * (phonemat[:, 0]==False)
        maxindex[claimed0] = -1
        maxindex = maxindex + 1
        phoneBag.append(maxindex)
        quinphoneID = quinphoneID + maxindex * (phoneDim ^ idx)
    
    changeidx = np.where(np.abs(np.diff(quinphoneID))>0)[0] + 1
    changeidx = np.insert(changeidx, 0, 0)
    changeidx = np.insert(changeidx, changeidx.shape[0], bdata.shape[0])
    
    fileptr = open(outfile, 'w')
    for idx, et in enumerate(changeidx):
        if idx == 0:
            continue
        else:
            st = changeidx[idx-1]
            fileptr.write('%d %d %s^%s-%s+%s=%s\n' % 
                          (st*resolu, et*resolu, 
                           phonList[phoneBag[0][st]],
                           phonList[phoneBag[1][st]],
                           phonList[phoneBag[2][st]],
                           phonList[phoneBag[3][st]],
                           phonList[phoneBag[4][st]]))
    fileptr.close()

#####  data extraction #########

# read in the mean and variance
if dataMV is not None:
    mv      = io.netcdf_file(dataMV,'r')
    if inOutData == 1:
        meanVec = mv.variables['inputMeans'][:].copy()
        varVec  = mv.variables['inputStdevs'][:].copy()
    else:
        meanVec = mv.variables['outputMeans'][:].copy()
        varVec  = mv.variables['outputStdevs'][:].copy()
    mv.close()
else:
    print('dataMV is not specified. Extracted data will be normalized data.\n')

# 
dataList = os.listdir(dataDir)
for dataFile in dataList:
    if re.search(dataPattern, dataFile):
        print(dataFile)
if os.path.isdir(dataOut):
    pass
else:
    os.mkdir(dataOut)


for dataFile in dataList:
    if re.search(dataPattern, dataFile):
        data = io.netcdf_file(dataDir+os.path.sep+dataFile)
        uttNum = data.dimensions['numSeqs']
        seqLengths = data.variables['seqLengths'][:].copy()
        seqLengths = np.concatenate((np.array([0]), seqLengths)).cumsum()
        seqTags   = data.variables['seqTags'][:]
        if inOutData == 1:
            dataAll = data.variables['inputs'][:]
        else:
            dataAll = data.variables['targetPatterns'][:]

        for i in range(uttNum):
            outName = dataOut+os.path.sep+''.join(seqTags[i])

            for j, suf in enumerate(name):
                outFile = outName + suf
                tmpdata = dataAll[seqLengths[i]:seqLengths[i+1],dim[j][0]:dim[j][1]].copy()
                if dataMV is not None:
                    tmpdata = tmpdata*varVec[dim[j][0]:dim[j][1]]+meanVec[dim[j][0]:dim[j][1]]
                genLabel(tmpdata, outFile)
            print("%s Utt %d" % (dataFile, i))
        del dataAll, seqTags, seqLengths, uttNum
        data.close()

