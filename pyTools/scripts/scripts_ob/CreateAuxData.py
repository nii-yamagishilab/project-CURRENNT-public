#!/usr/bin/python
from   ioTools     import readwrite as py_rw
from   speechTools import htslab 
import numpy       as np
import os


####################
# NOTE: This script is exclusively used for a specific format of alignment labels from Flite
#       Please re-write the script for your alignment labels.
#
#       The boundary information is generated as np.int8. Thus, at most 8 types of boundary
#       is supported currently.
#
#       Currently, I use 5 types: frame, phoneme, syllable, word, phrase. The lowest 5 bits
#       denote the boundary information. 1: updating, 0: not updating
#
#       The output for one utterance is a binary vector of np.int8, where each number encodes
#       the 5 types boundary updating information.
#
#       Use py_rw.read_raw_mat(PATH_TO_DATA, 1, 'u1', 'l') to read the output data
#

# File name list
DataList  = '/work/smg/wang/PROJ/WE/WORDE/data/nancy/file.scp'

# Output dir
DataDir   = '/work/smg/wang/DATA/speech/nancy/nndata/aux/allLevelAlign'

# Phoneme alignment
PhoAlign  = '/work/smg/wang/DATA/speech/nancy/nndata/full_align/full_align'

# Syllable alignment
SylAlign  = '/work/smg/wang/PROJ/WE/WORDE/data/nancy/syllable'

# word alignment
WorAlign  = '/work/smg/wang/PROJ/WE/WORDE/data/nancy/word'

# phrase marker
phraseSym = '##'

# resolution specify (bit)
bitInfo   = {'fra':np.int8(1),     # the first 
             'pho':np.int8(2), 
             'syl':np.int8(4), 
             'wor':np.int8(8), 
             'phr':np.int8(16)}

# print additional information ?
CheckBinary = 0 

####################

def CreateTimeMatrix(DataFile):    
    """
    """
    phofile  = PhoAlign + os.path.sep + DataFile + '.lab'
    sylfile  = SylAlign + os.path.sep + DataFile + '.lab'
    worfile  = WorAlign + os.path.sep + DataFile + '.lab'
    
    phodata  = htslab.read_full_lab(phofile)
    syldata  = htslab.read_full_lab(sylfile)
    wordata  = htslab.read_full_lab(worfile)
    
    if len(wordata[1]) != len(syldata[1]) or wordata[1][-1] != syldata[1][-1]:
        print "\t Unequal Length %s" % (DataFile)
        return 0
    
    DataTime = np.int(syldata[1][-1])
    
    # default, update at every frame level
    dataMat  = np.bitwise_or(np.zeros([DataTime], dtype=np.int8), bitInfo['fra'])
    
    preSyl = ''
    preWor = ''
    
    for idx1 in xrange(len(syldata[0])):
        frameStart = syldata[0][idx1]
        frameEnd   = syldata[1][idx1]
        
        # update the phoneme state
        dataMat[frameStart] = np.bitwise_or(dataMat[frameStart], bitInfo['pho'])
        
        syllabel   = syldata[2][idx1]
        worlabel   = wordata[2][idx1]
        
        if syllabel !=  preSyl:
            dataMat[frameStart] = np.bitwise_or(dataMat[frameStart], bitInfo['syl'])
        if worlabel !=  preWor:
            dataMat[frameStart] = np.bitwise_or(dataMat[frameStart], bitInfo['wor'])
        if len(preWor)==0 or preWor == phraseSym or worlabel == phraseSym:
            dataMat[frameStart] = np.bitwise_or(dataMat[frameStart], bitInfo['phr'])
        preSyl = syllabel
        preWor = worlabel
        if CheckBinary:
            pholabel   = phodata[2][idx1]
            for t in range(np.int(frameStart), np.int(frameEnd)):
                print "%d, %s [%s %s %s]" % (t,np.binary_repr(dataMat[t], len(bitInfo)),
                                             pholabel[0:6], syllabel[0:6], worlabel[0:6])

    py_rw.write_raw_mat(dataMat, DataDir+os.path.sep+DataFile+'.bin', 'u1')
    return DataTime


DataFiles    = py_rw.read_txt_list(DataList)
frameNum     = 0
for idx1, DataFile in enumerate(DataFiles):
    print "Process %s (%d / %d)" % (DataFile, idx1+1, len(DataFiles))
    frameNum = frameNum + CreateTimeMatrix(DataFile)

print "Total %d frames" % (frameNum)    
 



