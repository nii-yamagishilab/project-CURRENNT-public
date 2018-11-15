#!/usr/bin/python
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
from ioTools import readwrite as py_rw


def parseSize(data):
    weightNum = data(1)
    tSize     = data(2)
    pSize     = data(3)
    inPerBlock= data(4)
    inTernalW = data(5)
    return weightNum, tSize, pSize, inPerBlock, inTernalW

def getWeightNum(tSize, pSize, inPerBlock, inTernalW):
    # Transformation matrix
    # for feedforward, it is the Input2Hidden
    # for LSTM, it includes Input to every gate
    wNum            = tSize * (inPerBlock * (pSize))

    # bias
    # for feedforward, it is for Input2Hidden
    # for LSTM, it includes bias to every gate
    bNum            = tSize * (inPerBlock * (1))

    # other
    # for LSTM: Hidden2Hidden
    iNum            = tSize * inTernalW;
    weightNumCheck  = wNum + bNum + iNum

    return [weightNumCheck, wNum, bNum, iNum]
    

def ReadCURRENNTWeight(fileName, numFormat = 'f4', swap = 'l'):
    networkFile = py_rw.read_raw_mat(fileName, 1, numFormat, swap)
    layerNum    = int(networkFile[0])
    layerSize   = networkFile[1:1+layerNum*5]
    weights     = networkFile[1+layerNum*5:]

    weightMats  = []
    startPos    = 0
    for layerId in np.arange(0, layerNum):
        weightNum = int(layerSize[layerId * 5])
        tSize     = int(layerSize[layerId * 5 + 1])
        pSize     = int(layerSize[layerId * 5 + 2])
        inPerBlock= int(layerSize[layerId * 5 + 3])
        interW    = int(layerSize[layerId * 5 + 4])
        [weightNumCheck, wNum, bNum, iNum]  = getWeightNum(tSize, pSize, inPerBlock, interW)

        assert weightNumCheck == weightNum, "incompatible of the weight format. Please check CURRENNT version"
        weightMat = weights[startPos:startPos+weightNum]
        startPos  = startPos + weightNum
        if pSize * tSize == wNum:
            
            # normal feed-forward matrix
            W = weightMat[0:wNum].reshape([tSize, pSize]);
            b = weightMat[wNum:(wNum + bNum)].reshape([1, tSize]);
            Inter = weightMat[(wNum+bNum):];
        else:
            # blstm or weights for other architecture
            # need to re-write for blstm matrix
            W = weightMat[0 : wNum].reshape([inPerBlock * tSize, pSize]);
            b = weightMat[wNum:(wNum + bNum)].reshape([inPerBlock, tSize]);
            Inter = weightMat[wNum + bNum:];
        
        weightMats.append([W,b,Inter])

    return weightMats

def ref2filter(refcoeff):
    """ fiter_coefficients = ref2filter(reflection_coefficients):
    convert the reflection coefficients into the filter coefficients
    Note: 
       [a_1, ..., a_K] for filter 1 - a_1 z^-1 + ... - a_K z^K
    """
    order = len(refcoeff)
    mat   = np.eye(order + 1)
    for i in np.arange(order):
        # the orderIdx-th order filter
        orderIdx = i + 1
        mat[orderIdx, 0] = refcoeff[i]
        if orderIdx > 1:
            for j in np.arange(1, orderIdx):
                posIdx1 = j - 1
                posIdx2 = (orderIdx - 1) + 1 - j - 1
                mat[orderIdx, j] = mat[orderIdx-1, posIdx1] - refcoeff[i] * mat[orderIdx-1, posIdx2]            
    return mat[order, 0:-1][::-1]


def sarFilterGet(weightCoeff, arOrder, filtertype, tanh):
    """ filter_cof = sarFilterGet(weightCoeff, arOrder, type, tanh):
    filter_cof:  coefficients of filters, for each dimension of the feature
                 format [dim, order + 1]

    weightCoeff: coefficients from CURRENNT
                 assume input format is [dim, order]
    
    arOrder:     order of the AR

    tanh:        0. no tanh on pole parameter
                 1. tanh on pole parameter
                 tanh is only used when type == 2

    type:        1. classical form [1, a_1, a_2, a_n] -> 1 - a_1z^-1 - a_2z^-2 ...
                 2. cascade form real pole [1, a_1, a_2, a_n] -> (1 - a_1z^-1)(1 - a_2z^-2) ...
                 3. cascade complex form (single real pole + conjugated complex poles)
                 4. cascade complex form (only conjudagated complex poles)
                 5. reflection coefficients
    """
    if len(weightCoeff.shape) == 1:
        # turn [a_1, a_2, a_order] into [[a_1, a_2, a_order]]
        weightCoeff = np.reshape(weightCoeff, [1, weightCoeff.shape[0]])

    if filtertype is 1:
        # [a_1, a_2, ...] => [1, -a_1, -a_2]
        return np.concatenate([np.ones([weightCoeff.shape[0], 1]), -1 * weightCoeff], axis=1)
    
    elif filtertype is 2:
        if tanh is 1:
            weightCoeff = np.tanh(weightCoeff)
        tmpCoe = np.zeros([weightCoeff.shape[0], arOrder + 1])
        coeff  = np.zeros([weightCoeff.shape[0], arOrder + 1])
        coeff[:,0] = 1
        coeff[:,1] = -1 * weightCoeff[:, 0]
        for n in np.arange(arOrder-1):
            # iterative algorithm to convert filter coefficients
            dimIdx = n+1
            tmpCoe[:, 1:] = coeff[:,0:arOrder] * -1 * weightCoeff[:, dimIdx:dimIdx+1]
            coeff[:,0:(arOrder+1)] = coeff[:,0:(arOrder+1)] + tmpCoe
        return coeff[:,0:(arOrder+1)]
    
    elif filtertype is 3 or filtertype is 4:
        featDim  = weightCoeff.shape[0]
        casOrder = int(np.ceil(arOrder / 2))
        
        tmpCoe   = np.reshape(weightCoeff, [featDim, casOrder*2])

        # convert from beta, alpha to cascade 2-order filter coefficients
        tmpCoe[:, 1:casOrder*2:2] = 1/(1+np.exp(-1.0 * tmpCoe[:, 1:casOrder*2:2]))
        tmpCoe[:, 0:casOrder*2:2] = np.tanh(tmpCoe[:, 0:casOrder*2:2]) * np.sqrt(tmpCoe[:, 1:casOrder*2:2])

        if filtertype is 3:
            tmpCoe[:, 1] = 0;  # void dimension for the real pole
            tmpCoe[:, 0] = np.tanh(tmpCoe[:, 0]) # real pole dimension
            
        outCoef = np.zeros([featDim, arOrder + 1])
        
        # convert from cascade form to classical form
        for dimIdx in np.arange(featDim):
            # for each data dimension
            filterCoef = np.zeros([2, casOrder*2+2])
            filterCoef[0,0] = 1

            # for the first 2-order filter
            filterCoef[0,1:(1+2)] = tmpCoe[dimIdx, 0:2]

            for n in np.arange(casOrder-1):
                filterCoef[1,1:(1+casOrder*2)] = filterCoef[0,0:(casOrder*2)] * tmpCoe[dimIdx, (n+1)*2]
                filterCoef[1,2:(2+casOrder*2)] = filterCoef[1,2:(2+casOrder*2)] + filterCoef[0,0:(0+casOrder*2)] * tmpCoe[dimIdx, (n+1)*2+1]
                filterCoef[0,:] = filterCoef[0,:] + filterCoef[1,:]
                
            # copy to output buffer
            outCoef[dimIdx, :] = filterCoef[0, 0:arOrder+1].copy()
        
        return outCoef
    elif filtertype is 5:
        featureDim, order = weightCoeff.shape[0], weightCoeff.shape[1]
        outCoeff = np.zeros([featureDim, order+1])
        outCoeff[:, 0] = 1.0
        for featIdx in np.arange(featureDim):
            outCoeff[featIdx, 1:(1+order)] = -1.0*ref2filter(np.tanh(weightCoeff[featIdx, :]))
        return outCoeff
        
    else:
        print("Unknown type")
        print("Please check help message of this function")
        return []
    
def sarWeightArange(weightCoeff, filterDim, arOrder):
    """ sarWeightArange(weightCoeff, filterDim, arOrdeer)
    weightCoeff: a vector of filter coefficients learned by CURRENNT
    filterDim:   dimenison of the feature
    arOrder:     order of the AR filter, 1-a_1 z^1 ...- a_K z^K, K-order filter
    This function just format the vector into a matrix
    return filter_coefficients [dim, order], bias vector
    """
    assert weightCoeff.shape[0] == filterDim * (arOrder + 1), 'invalid config'
    bias = weightCoeff[filterDim * arOrder :]
    weightCoeff = np.reshape(weightCoeff[:filterDim * arOrder], [arOrder, filterDim])
    return np.transpose(weightCoeff), bias

    

if __name__ == "__main__":
    weightCoeff = np.array([ 0.180554,  0.180554])
    temp = sarFilterGet(weightCoeff, 2, 2, 1)
    print(temp)
