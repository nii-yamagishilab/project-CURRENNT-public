from __future__ import print_function
from __future__ import absolute_import


import sys, os
import numpy as np

def read_full_lab(labfile, res=50000, state=1):

    nline = 0
    with open(labfile, 'r') as filePtr:
        for line in filePtr:
            nline += 1
    print("%d state for %s" % (nline, labfile))
    sTime = np.zeros([nline])
    eTime = np.zeros([nline])
    labEntry = []
    nline=0
    with open(labfile, 'r') as filePtr:
        for line in filePtr:
            temp = line.split()
            assert len(temp)>=3, "Invalid label entry %s" % (line)
            labEntry.append(temp[2])
            sTime[nline] = int(temp[0])/res
            eTime[nline] = int(temp[1])/res
            nline += 1

    if state > 1:
        assert nline % state == 0, "Error: incompatible state numbers"
        sTime = sTime[::state]
        eTime = eTime[state-1::state]
        labEntry = [x for idx,x in enumerate(labEntry) if idx % state == 0]
        assert sTime.shape[0] == eTime.shape[0], "Error: fail to extract state"
        assert sTime.shape[0] == len(labEntry), "Error: fail to extract state"
        
    return sTime, eTime, labEntry
                         



def phoneExtract(labs, leftB='+', rightB='-'):
    phones = []
    for labLine in labs:
        temp = labLine.split(leftB)[0]    
        temp = temp.split(rightB)[1]
        phones.append(temp)
    return phones
