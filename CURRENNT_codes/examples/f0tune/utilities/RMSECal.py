

##############################
#
import numpy as np

import re
import sys
import scipy
from scipy import stats


try:
    from binaryTools import readwriteC2 as funcs
except ImportError:
    try:
        from binaryTools import readwriteC2_220 as funcs
    except ImportError:
        print "Please add ~/CODE/pyTools to PYTHONPATH"
        try: 
            from ioTools import readwrite as funcs
        except ImportError:
            print "Can't not import binaryTools/readwriteC2 or funcs"


def RMSECalcore(file1, file2, dim):
    data1 = funcs.read_raw_mat(file1, dim)
    data2 = funcs.read_raw_mat(file2, dim)
    if data1.shape[0]>data2.shape[0]:
        if dim==1:
            data1 = data1[0:data2.shape[0]]
        else:
            data1 = data1[0:data2.shape[0],:]
        #else:
        #    assert 1==0, "Unknown dimension"
    elif data1.shape[0]<data2.shape[0]:
        if dim==1:
            data2 = data2[0:data1.shape[0]]
        else:
            data2 = data2[0:data1.shape[0],:]
            #assert 1==0, "Unknown dimension"
    
    #if data1.ndim==1:
    #    data1 = data1.reshape([data1.shape[0],1])
    #    data2 = data2.reshape([data2.shape[0],1])
    
    if dim==1:
        # This is F0 
        diff = np.zeros([data1.shape[0], 3])
        temp1 = data1>0
        temp2 = data2>0
        indp = (temp1 *temp2)             # all voiced
        indn = (temp1 - temp2)             # u/v different
        voiceFrame = sum(indp)
        if voiceFrame>0:
            diff[indp,0] = data1[indp]-data2[indp] #
            diff[indn,1] = 1                       # 
            diff[indp,2] = 1
            pow2 = diff*diff
            corr = scipy.stats.pearsonr(data1[indp],data2[indp])
        else:
            corr = [np.nan,0]
            pow2 = diff*np.nan
        
    else:
        diff = data1 - data2
        pow2 = diff*diff
        voiceFrame = data1.shape[0]
        corr = -10

    return pow2, data1.shape[0], corr
    
def RMSECal(scp1, dim, g_resolu, g_escape):
    timeStep = np.array([])
    fileNM   = 0
    with open(scp1, "r") as filePtr:
        for line in filePtr:
            fileNM += 1
    
    errorSum = np.zeros([fileNM+1, dim+1])
    #errorMean= np.zeros([fileNM+1, dim+1])
    corrStack = []
    fileNM = 0
    
    with open(scp1, "r") as filePtr:
        for idx,line in enumerate(filePtr):
            [file1, file2, file3] = line.split()
            [tSum, times, corr] = RMSECalcore(file1, file2, dim)
            if dim==1:
                corrStack.append(corr[0])
            frame=0
            voiceFrame=0
            with open(file3, "r") as filePtr2:
                for line2 in filePtr2:
                    [sTime, eTime, labEntry] = line2.split()
                    #if 1:
                    if not re.search(g_escape, labEntry):
                        sTime = int(sTime)/g_resolu
                        eTime = int(eTime)/g_resolu
                        if dim==1:
                            # F0
                            errorSum[fileNM,0] += tSum[sTime:eTime,0].sum()
                            errorSum[fileNM,1] += tSum[sTime:eTime,1].sum()
                            voiceFrame += sum(tSum[sTime:eTime,2])
                        else: 
                            errorSum[fileNM,0:dim] += tSum[sTime:eTime,:].sum(axis=0)
                            errorSum[fileNM,dim] += tSum[sTime:eTime,:].sum()
                        frame += eTime-sTime
                        #errorMean[fileNM,0:dim+1]+= tSum[sTime:eTime,:].mean(axis=0)
                        #errorMean[fileNM,dim]+= tS
                    else:
                        # silence
                        pass
            if dim==1:
                #F0:
                errorSum[fileNM,0] = np.sqrt(errorSum[fileNM,0]/voiceFrame)
                errorSum[fileNM,1] = errorSum[fileNM,1]/frame
            else:
                errorSum[fileNM,:] = np.sqrt(errorSum[fileNM,:]/frame)
            fileNM += 1
    
    errorSum[fileNM,:] = errorSum[0:fileNM,:].mean(axis=0)
    corrStack = np.array(corrStack)
    if dim==1:
        print "RMSE: %f\tCor: %f\t VU:%f\t" % (
            errorSum[fileNM,0], 
            corrStack.mean(), 
            errorSum[fileNM,1]),
        #print "Corre: %f\t" % ()
        corrStack = np.concatenate((corrStack, np.array([corrStack.mean()])))
        corrStack = np.expand_dims(corrStack, axis=1)
        errorSum = np.concatenate((errorSum, corrStack), axis=1)
    else:
        print "RMSE: %f\t" % (errorSum[fileNM,0]),
    return errorSum
    
    
if __name__ == "__main__":
    scpFile = sys.argv[1]
    dim     = sys.argv[2]
    g_resolu = int(sys.argv[3])
    g_escape = sys.argv[4]
    output = sys.argv[5]
    #sourdir = 
    funcs.write_raw_mat(RMSECal(scpFile, int(dim), g_resolu, g_escape), output)
    #print ""
    
