#!/usr/bin/python
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import os
from six.moves import range


def preRec(mat, verBose = True):
    """ 
    row is the predicted results
    col is the golden-standard
    """
    matNew = np.zeros([mat.shape[0]+1, mat.shape[1]+1])
    matNew[:mat.shape[0],:mat.shape[1]] = mat
    matNew = np.array(matNew, dtype=np.float32)
    matNew[-1, 0:-1] = mat.sum(axis=0)
    matNew[0:-1, -1] = mat.sum(axis=1)
    matNew[-1,-1] = mat.sum().sum()
   
    rec = mat[list(range(mat.shape[0])), list(range(mat.shape[0]))]/matNew[-1,:mat.shape[0]]
    pre = mat[list(range(mat.shape[0])), list(range(mat.shape[0]))]/matNew[:mat.shape[0],-1]
    acc = mat[list(range(mat.shape[0])), list(range(mat.shape[0]))].sum()/matNew[-1,-1]
    f0  = 2*rec*pre/(rec+pre)
    strtmp = "recall\t" + str(rec) + "\tprecision\t" + str(pre) + "\taccuracy\t" + str(acc)
    if verBose:
        print(strtmp)
        print("f0\t" + str(f0))
    return np.concatenate((rec, pre, f0)), acc
    

def MakeIndx(ix1, ix2):
    """ [0,1,3] [4,5,6]
    ==> [0,0,0,1,1,1,3,3,3][4,5,6,4,5,6,4,5,6]
    """ 
    ou1 = []
    ou2 = []
    for id1, idx1 in enumerate(ix1):
        for id2, idx2 in enumerate(ix2):
            ou1.append(idx1)
            ou2.append(idx2)
    return ou1, ou2 

def mergeMatrix(mat, mergeMap, mergeInfo=None, verBose =True):
    """ only 2-d matrix
    """
    resultStack = []
    accStack = []
    for idx, mer in enumerate(mergeMap):
        if mergeInfo is not None and verBose:
            print(mergeInfo[idx])
        tmpMat = np.zeros([len(mer), len(mer)], dtype=np.int32)
        
        for idr, rowIndx in enumerate(mer):
            for idc, colIndx in enumerate(mer):
                rowIndx, colIndx = MakeIndx(rowIndx,colIndx)
                tmpMat[idr, idc] = mat[rowIndx, colIndx].sum()
                # print mat[rowIndx, colIndx]
        if verBose:
            print(np.array2string(tmpMat, separator='\t'))
        result, acc = preRec(np.array(tmpMat), verBose)
        resultStack.append(result)
        accStack.append(acc)
    return resultStack, accStack

            
def confusionMatrix(result, resultMap=None, verBose=True):
    """ we assume that result
    result: #samples * 2
    confusion matrix
    row: predicted
    col: golden-tag
    """
    result = np.array(result, dtype=np.int32)
    mat = np.zeros([len(list(resultMap.keys())), len(list(resultMap.keys()))], dtype=np.int32)
    for idx in range(result.shape[0]):
        mat[result[idx,0], result[idx,1]] += 1
    if verBose:
        print(resultMap)
        print(np.array2string(mat,separator='\t'))
    return mat


if __name__ == "__main__":
    # data 
    # res = confusionMatrix(data, cfg.ToBIMapping)
    # mergeMatrix(res, cfg.MergeTag, cfg.MergeInfo) 
    pass
