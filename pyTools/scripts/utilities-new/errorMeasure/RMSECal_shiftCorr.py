

##############################
#
from __future__ import absolute_import
from __future__ import print_function
import numpy as np

import re
import sys
import scipy
from scipy import stats
from six.moves import range


try:
    from binaryTools import readwriteC2 as funcs
except ImportError:
    try:
        from binaryTools import readwriteC2_220 as funcs
    except ImportError:
        try: 
            from ioTools import readwrite as funcs
        except ImportError:
            print("Please add ~/CODE/pyTools to PYTHONPATH")
            print("Can't not import binaryTools/readwriteC2 or funcs")

def F0Transform(data):
    #data_out = (np.exp(data/1127.0)-1)*700
    data_out = data
    return data_out


def RMSECalcore(file1, file2, dim):
    data1 = funcs.read_raw_mat(file1, dim)
    data2 = funcs.read_raw_mat(file2, dim)

    shift_max = np.abs(data2.shape[0] - data1.shape[0])
    if data1.shape[0]>data2.shape[0]:
        valid_length = data2.shape[0]
        fixed_data   = data2
        shift_data   = data1
    else:
        valid_length = data1.shape[0]
        fixed_data   = data1
        shift_data   = data2

    max_v_cover = 0
    max_corr = -1.0
    min_rmse = 1000000
    min_rmse_buf = []
    max_corr_buf = []
    shift_pos = 0

    if shift_max == 0:
        shift_max = 1
    for shift_t in range(shift_max):        
        if dim==1:
            shift_data_temp = shift_data[shift_t:shift_t + valid_length].copy()
            fixed_data_temp = fixed_data.copy()
            
            # This is F0 
            diff = np.zeros([shift_data_temp.shape[0], 3])
            temp1 = shift_data_temp > 0
            temp2 = fixed_data_temp > 0
            indp = (temp1 *temp2)             
            indn = np.bitwise_xor(temp1, temp2)
            voiceFrame = sum(indp)
        
            if voiceFrame>0:
                shift_data_temp = F0Transform(shift_data_temp[indp])
                fixed_data_temp = F0Transform(fixed_data_temp[indp])
                diff[indp,0] = shift_data_temp - fixed_data_temp
                diff[indn,1] = 1                        
                diff[indp,2] = 1
                pow2 = diff*diff
                corr = scipy.stats.spearmanr(shift_data_temp, fixed_data_temp)
            
            else:
                corr = [np.nan,0]
                pow2 = diff*np.nan

            v_cover = voiceFrame * 1.0 / valid_length
                
        else:
            print('Only for F0 data')
            
        if v_cover > max_v_cover:
            max_corr = corr[0]
            max_corr_buf = corr
            min_rmse_buf = pow2
            shift_pos = shift_t
            max_v_cover = v_cover

    return min_rmse_buf, valid_length, max_corr_buf
    
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

            sTime = 0
            eTime = times
            if dim==1:
                # F0
                errorSum[fileNM,0] += tSum[0:times,0].sum()
                errorSum[fileNM,1] += tSum[0:times,1].sum()
                voiceFrame += sum(tSum[sTime:eTime,2])
            else: 
                errorSum[fileNM,0:dim] += tSum[sTime:eTime,:].sum(axis=0)
                errorSum[fileNM,dim] += tSum[sTime:eTime,:].sum()
            frame += eTime-sTime

            if voiceFrame > 0:
                if dim==1:
                    #F0:
                    errorSum[fileNM,0] = np.sqrt(errorSum[fileNM,0]/voiceFrame)
                    errorSum[fileNM,1] = errorSum[fileNM,1]/frame
                    print(file1, errorSum[fileNM,1], corr[0])
                else:
                    errorSum[fileNM,:] = np.sqrt(errorSum[fileNM,:]/frame)
                
            else:
                if dim==1:
                    #F0:
                    errorSum[fileNM,0] = np.nan
                    errorSum[fileNM,1] = np.nan
                    print(file1, 'nan')
                else:
                    errorSum[fileNM,:] = np.nan
            fileNM += 1
            
    errorSum[fileNM,:] = errorSum[0:fileNM,:].mean(axis=0)
    corrStack = np.array(corrStack)

    
    if dim==1:
        print("Average: RMSE: %f\tCor: %f\t VU:%f\t\n" % (
            errorSum[fileNM,0], 
            corrStack.mean(), 
            errorSum[fileNM,1]), end=' ')
        corrStack = np.concatenate((corrStack, np.array([corrStack.mean()])))
        corrStack = np.expand_dims(corrStack, axis=1)
        errorSum = np.concatenate((errorSum, corrStack), axis=1)
    else:
        print("Average: RMSE: %f\t" % (errorSum[fileNM, -1]), end=' ')
        
    return errorSum

def showRMSE(dim, rmseFile):
    if dim == 1:
        # F0
        data = funcs.read_raw_mat(rmseFile, 3)
        print("RMSE: %f\tCor: %f\t VU:%f\t" % (
            data[-1,0], 
            data[-1,2], 
            data[-1,1]), end=' ')
    else:
        # MGC
        data = funcs.read_raw_mat(rmseFile, dim+1)
        print("RMSE: %f\t" % (data[-1,-1]), end=' ')
    
        
    
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
        
