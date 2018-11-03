

##############################
#
import numpy as np
import os
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
        try: 
            from ioTools import readwrite as funcs
        except ImportError:
            print "Please add ~/CODE/pyTools to PYTHONPATH"
            print "Can't not import binaryTools/readwriteC2 or funcs"

def F0Transform(data):
    #data_out = (np.exp(data/1127.0)-1)*700
    data_out = data
    return data_out


def RMSECalcore(file1, file2, dim):
    data1 = funcs.read_raw_mat(file1, dim)
    data2 = funcs.read_raw_mat(file2, dim)

    # check the data length
    if np.abs(data1.shape[0] - data2.shape[0]) * 2.0 / (data1.shape[0] + data2.shape[0]) > 0.2:
        print "Warning: length mis-match: %s %d, %s %d" % (file1, data1.shape[0],
                                                           file2, data2.shape[0])
        
    # slightly change the length of data
    if data1.shape[0]>data2.shape[0]:
        if dim==1:
            data1 = data1[0:data2.shape[0]]
        else:
            data1 = data1[0:data2.shape[0],:]
    elif data1.shape[0]<data2.shape[0]:
        if dim==1:
            data2 = data2[0:data1.shape[0]]
        else:
            data2 = data2[0:data1.shape[0],:]
    
    
    if dim==1:
        # This is F0 
        diff = np.zeros([data1.shape[0], 3])
        temp1 = data1 > 0
        temp2 = data2 > 0
        
        # all voiced time steps
        indp = (temp1 * temp2)
        
        # u/v different time steps
        indn = np.bitwise_xor(temp1, temp2)

        # number of voiced frames
        voiceFrame = sum(indp)               
        
        if voiceFrame>0:
            data1 = F0Transform(data1[indp])
            data2 = F0Transform(data2[indp])
            diff[indp,0] = data1-data2 
            diff[indn,1] = 1           
            diff[indp,2] = 1
            rmse = diff*diff
            
            #corr = scipy.stats.pearsonr(data1,data2)
            corr = scipy.stats.spearmanr(data1, data2)
            
        else:
            corr = [np.nan, 0]
            rmse = diff * np.nan
        
    else:
        diff = data1 - data2
        rmse = diff*diff
        voiceFrame = data1.shape[0]
        corr = -10

    return rmse, data1.shape[0], corr


def RMSECal(scp1, dim, g_resolu, g_escape):
    
    timeStep = np.array([])
    fileNM   = 0
    
    with open(scp1, "r") as filePtr:
        for line in filePtr:
            fileNM += 1
    
    errorSum = np.zeros([fileNM+1, dim+1])
    corrStack = []
    fileNM = 0
    
    with open(scp1, "r") as filePtr:
        for idx,line in enumerate(filePtr):
            [file1, file2, file3] = line.split()
            [rmse, data_length, corr] = RMSECalcore(file1, file2, dim)
            
            if dim==1:
                corrStack.append(corr[0])
                
            frame=0
            voiceFrame=0

            # read force-aligned lab and remove sil part
            if os.path.isfile(file3):
                with open(file3, "r") as filePtr2:
                    for line2 in filePtr2:
                        [sTime, eTime, labEntry] = line2.split()
                        
                        if not re.search(g_escape, labEntry):
                            sTime = int(sTime)/g_resolu
                            eTime = int(eTime)/g_resolu
                            if dim==1:
                                errorSum[fileNM,0] += rmse[sTime:eTime,0].sum()
                                errorSum[fileNM,1] += rmse[sTime:eTime,1].sum()
                                voiceFrame += sum(rmse[sTime:eTime,2])
                            else: 
                                errorSum[fileNM,0:dim] += rmse[sTime:eTime,:].sum(axis=0)
                                errorSum[fileNM,dim] += rmse[sTime:eTime,:].sum()
                            frame += eTime-sTime
            else:
                sTime = 0
                eTime = rmse.shape[0]
                if dim==1:
                    errorSum[fileNM,0] += rmse[sTime:eTime,0].sum()
                    errorSum[fileNM,1] += rmse[sTime:eTime,1].sum()
                    voiceFrame += sum(rmse[sTime:eTime,2])
                else: 
                    errorSum[fileNM,0:dim] += rmse[sTime:eTime,:].sum(axis=0)
                    errorSum[fileNM,dim] += rmse[sTime:eTime,:].sum()
                frame += eTime-sTime

            # summarize results
            if dim==1:
                # F0 rmse
                errorSum[fileNM,0] = np.sqrt(errorSum[fileNM,0]/voiceFrame)
                # F0 UV/
                errorSum[fileNM,1] = errorSum[fileNM,1]/frame
            else:
                # RMSE
                errorSum[fileNM,:] = np.sqrt(errorSum[fileNM,:]/frame)
                
            fileNM += 1

    # write mean of all utterances to the final row of the data matrix
    errorSum[fileNM,:] = errorSum[0:fileNM,:].mean(axis=0)
    corrStack = np.array(corrStack)
    
    if dim==1:
        # errorSum = [RMSE, UV]
        print "Average RMSE: %f\tCor: %f\t VU:%f\t" % (
            errorSum[fileNM,0], 
            corrStack.mean(), 
            errorSum[fileNM,1]),
        corrStack = np.concatenate((corrStack, np.array([corrStack.mean()])))
        corrStack = np.expand_dims(corrStack, axis=1)
        errorSum = np.concatenate((errorSum, corrStack), axis=1)
    else:
        # errorSum[RMSE 1st dim, RMSE 2nd dim, ..., RMSE average over dim]
        print "Average RMSE: %f\t" % (errorSum[fileNM, -1]),
    return errorSum

def showRMSE(dim, rmseFile):
    if dim == 1:
        # F0
        data = funcs.read_raw_mat(rmseFile, 3)
        print "RMSE: %f\tCor: %f\t VU:%f\t" % (
            data[-1,0], 
            data[-1,2], 
            data[-1,1]),
    else:
        # MGC
        data = funcs.read_raw_mat(rmseFile, dim+1)
        print "RMSE: %f\t" % (data[-1,-1]),
        
    
if __name__ == "__main__":
    scpFile  = sys.argv[1]
    dim      = sys.argv[2]
    g_resolu = int(sys.argv[3])
    g_escape = sys.argv[4]
    output   = sys.argv[5]
    calFlag  = int(sys.argv[6])
    
    if calFlag > 0:
        funcs.write_raw_mat(RMSECal(scpFile, int(dim), g_resolu, g_escape), output)
    else:
        showRMSE(int(dim), output)
        
