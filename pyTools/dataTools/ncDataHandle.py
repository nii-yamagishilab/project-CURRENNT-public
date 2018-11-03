#!/usr/bin/python
import scipy
import os, sys
from scipy import io
import numpy as np
import time
#try:
#    import funcs as py_rw
#except ImportError:
#    pass

#try:
#    import funcs_220 as py_rw
#except ImportError:
#    print "Searching for binary io in pypath"
#    pass

try:
    from binaryTools import readwriteC2 as py_rw
except ImportError:
    try:
        from binaryTools import readwriteC2_220 as py_rw
    except ImportError:
        assert 1==0,"Cant find readwrite"

g_VOIDFILE     = '#'
g_MinMaxRange  = 0.0 # normalize the value to [g_minmaxrange, 1-g_minmaxrange]

def errorMes(datafiles, errorType):
    error=''
    if errorType==0:  
        # unequal dimension
        error = error + "Unequal dimension of input feature and data mean/var \n"
    elif errorType==1:  
        # unequal dimension
        error = error + "Unequal dimension of output feature and data mean/var \n"
    elif errorType==2:
        error = error + "Unequal dimension of data \n"
    elif errorType==4:
        error = error + "Configuration file error\n"
    elif errorType==3:
        # unequal time length
        error = error + "Unequal number of data frames\n"
        
    error = error + 'Please check:\n'
    for data in datafiles:
        error = error + '\t' + data + '\n'
    return error
        


def readmv(data):
    data = io.netcdf_file(data)
    try:
        mi = data.variables['inputMeans'][:].copy()
        vi = data.variables['inputStdevs'][:].copy()
    except KeyError:
        print "Can't find mv input"
        mi, vi = None, None
        
    try:
        mo = data.variables['outputMeans'][:].copy()
        vo = data.variables['outputStdevs'][:].copy()
    except KeyError:
        print "Can't find mv output"
        mo, vo = None, None
    data.close()
    return mi, vi, mo, vo
        
def _write(f):
    f.fp.seek(0)
    f.fp.write(b'CDF')
    f.fp.write(np.array(f.version_byte, '>b').tostring())

    # Write headers and data.
    f._write_numrecs()
    f._write_dim_array()
    f._write_gatt_array()
    f._write_var_array()


    
def norm(ncFile, mvFile, ncTarget=None, mask=None, flagKeepOri=1, 
         addMV=1, flushT=400000, waitT=5, stdT=0.000001, reverse=0):
    """ 
    normalizing the data, 
    specify the operation and call ncFileNanipulate
    mask: 1: normalize it
          0: not normalize it
    """
    print "norm %s " % (ncFile)
    if mask is not None:
        maskData = py_rw.read_raw_mat(mask, 1)
        if reverse:
            maskData = 1-maskData
        operation = lambda x,y,z,m: (x-y*m)/(z**m)
    else:
        maskData = None
        operation = lambda x,y,z: (x-y)/z
    ncFileManipulate(ncFile, mvFile, operation, ncTarget, flagKeepOri, 
                     addMV, flushT, waitT, stdT, maskData)

def denorm(ncFile, mvFile, ncTarget=None, mask=None, flagKeepOri=1, 
           addMV=1, flushT=400000, waitT=5, stdT=0.000001, reverse=0):
    """ 
    denormalizing the data, 
    specify the operation and call ncFileNanipulate
    mask: 1: denormalize it
          0: not denormalize it
    """
    print "denorm %s" % (ncFile)
    if mask is not None:
        maskData = py_rw.read_raw_mat(mask, 1)
        if reverse:
            maskData = 1-maskData
        operation = lambda x,y,z,m: (x*(z**m)+y*m)
    else:
        maskData = None
        operation = lambda x,y,z: (x*z+y)
    ncFileManipulate(ncFile, mvFile, operation, ncTarget, flagKeepOri, 
                     addMV, flushT, waitT, stdT, maskData)

def normnom(ncFile, mvFile, ncTarget=None, mask=None, reverse=0, flagKeepOri=1, 
            addMV=1, flushT=400000, waitT=5, stdT=0.000001):
    """ 
    normalizing without shifting mean of certain dimension
    specify the operation and call ncFileNanipulate
    mask: 1: normalize it without shifting mean
          0: normalize it
    """
    print "normlizing without mean shift %s" % (ncFile)
    if mask is not None:
        maskData = py_rw.read_raw_mat(mask, 1)
        if reverse:
            maskData = 1-maskData
        operation = lambda x,y,z,m: (x-y*(1-m))/z
    else:
        maskData = None
        operation = lambda x,y,z: (x)/z
    ncFileManipulate(ncFile, mvFile, operation, ncTarget, flagKeepOri, 
                     addMV, flushT, waitT, stdT, maskData)

def compennom(ncFile, mvFile, ncTarget=None, mask=None, reverse=0, flagKeepOri=1, 
              addMV=1, flushT=400000, waitT=5, stdT=0.000001):
    """ 
    Add the mean/std back to some dimensions
    mask: 1: compensente it
          0: not compensente
    """
    print "compensente %s" % (ncFile)
    if mask is not None:
        maskData = py_rw.read_raw_mat(mask, 1)
        if reverse:
            maskData = 1-maskData
        operation = lambda x,y,z,m: x+y/z*m
    else:
        maskData = None
        operation = lambda x,y,z: x+y/z
    ncFileManipulate(ncFile, mvFile, operation, ncTarget, flagKeepOri, 
                     addMV, flushT, waitT, stdT, maskData)


def ncFileManipulate(ncFile, mvFile, operation, ncTarget=None, flagKeepOri=1, 
                     addMV=1, flushT=400000, waitT=5, stdT=0.000001, maskData=None):
    """
    This is a warper for handling the nc data,
    'operation' specify the operation on each time step of the data, the arg of 
    'operation' is (data, mean, std)
    by default, ncTarget = ncFile, and the original file will be deleted
    """
    if ncTarget is None:
        ncTarget = ncFile+'Manipulated'
    if ncTarget == ncFile:
        print "the same input and output ncFile name detected"
        ncTarget = ncFile+'Manipulated'
        print "use %s as output name if you don't turn on flagKeepOri" % (ncTarget)
    dataOut = io.netcdf.netcdf_file(ncTarget,'w',version=2)
    dataIn  = io.netcdf_file(ncFile)
    dataMV  = io.netcdf_file(mvFile)
    
    # create New dimension
    for dimName in dataIn.dimensions:
        dataOut.createDimension(dimName, dataIn.dimensions[dimName])
    # create variable
    for varName in dataIn.variables:
        dataOut.createVariable(
            varName, dataIn.variables[varName].typecode(), 
            dataIn.variables[varName].dimensions
        )
        dataOut.variables[varName][:] = dataIn.variables[varName][:].copy()
    
    meanIn  = dataMV.variables['inputMeans'][:].copy()
    stdIn   = dataMV.variables['inputStdevs'][:].copy()
    meanOut = dataMV.variables['outputMeans'][:].copy() 
    stdOut  = dataMV.variables['outputStdevs'][:].copy()
    
    # make sure no singula std in division
    stdIn[stdIn<stdT] = 1.0
    stdOut[stdOut<stdT] = 1.0
    
    # normlize the data
    numTimes = dataIn.dimensions['numTimesteps']
    inputPatSize = dataIn.dimensions['inputPattSize']
    outPatSize   = dataIn.dimensions['targetPattSize']
    assert meanIn.shape[0] == inputPatSize, errorMes([mvFile, ncFile], 0)
    assert stdIn.shape[0] == inputPatSize, errorMes([mvFile, ncFile], 0)
    assert meanOut.shape[0] == outPatSize, errorMes([mvFile, ncFile], 1)
    assert stdOut.shape[0] == outPatSize, errorMes([mvFile, ncFile], 1)
    
    if maskData is not None:
        print "manipulate with Mask %s %s" % (maskData.shape[0], inputPatSize+outPatSize)
        assert maskData.shape[0]==inputPatSize+outPatSize, errorMes(['data_config.py'], 2) + \
            "\nThe length of mask is not equal to input and output dim %d %d" % \
            (maskData.shape[0], inputPatSize+outPatSize)
    count = 0
    # This is stupid, to loop over all frames !!!
    """for t in xrange(numTimes):
        if maskData is None:
            dataOut.variables['inputs'][t, :] = operation(
                dataIn.variables['inputs'][t, :], meanIn, stdIn
            )
            dataOut.variables['targetPatterns'][t, :] = operation(
                dataIn.variables['targetPatterns'][t, :], meanOut, stdOut
            )
        else:
            dataOut.variables['inputs'][t, :] = operation(
                dataIn.variables['inputs'][t, :], meanIn, stdIn, maskData[0:inputPatSize]
            )
            dataOut.variables['targetPatterns'][t, :] = operation(
                dataIn.variables['targetPatterns'][t, :], meanOut, stdOut, maskData[inputPatSize:]
            )
        count += 1
        if count>=flushT:
            count = 0
            dataOut.flush()
            print "%d/%d, Let's wait netCDF for %d(s)" % (numTimes, t, waitT)
            #raw_input("Enter")
            for x in xrange(waitT):
                print "*",
                sys.stdout.flush()
                time.sleep(1)
            print "*"
    """
    dataBatchNum = (numTimes / flushT + ((numTimes % flushT)>0))
    for t in xrange(dataBatchNum):
        st = 0
        et = 0
        if t == (dataBatchNum - 1):
            st = t * flushT
            et = numTimes
        else:
            st = t * flushT
            et = (t+1)*flushT
            
        if maskData is None:
            dataOut.variables['inputs'][st:et, :] = operation(
                dataIn.variables['inputs'][st:et, :], meanIn, stdIn
            )
            dataOut.variables['targetPatterns'][st:et, :] = operation(
                dataIn.variables['targetPatterns'][st:et, :], meanOut, stdOut
            )
        else:
            dataOut.variables['inputs'][st:et, :] = operation(
                dataIn.variables['inputs'][st:et, :], meanIn, stdIn, maskData[0:inputPatSize]
            )
            dataOut.variables['targetPatterns'][st:et, :] = operation(
                dataIn.variables['targetPatterns'][st:et, :], meanOut, stdOut, maskData[inputPatSize:]
            )
            
        dataOut.flush()
        print "%d/(%d-%d), Let's wait netCDF for %d(s)" % (numTimes, st, et, waitT)
        #raw_input("Enter")
        for x in xrange(waitT):
            print "*",
            sys.stdout.flush()
            time.sleep(1)
        print "*"
           
    
    # add MV is necessary
    if addMV == 1:
        # for both input and output
        for varName in dataMV.variables:
            #if varName[0:6]=="output":
            dataOut.createVariable(varName, dataMV.variables[varName].typecode(), 
                                   dataMV.variables[varName].dimensions)
            dataOut.variables[varName][:] = dataMV.variables[varName][:].copy()
    dataMV.close()
    dataOut.flush()
    dataOut.close()
    if flagKeepOri==0:
        os.system("rm %s" % (ncFile))
        os.system("mv %s %s" % (ncTarget, ncFile))
        print "*** %s prepared (original file deleted) " % (ncFile)
    else:
        print "*** %s prepared " % (ncTarget)
    

def meanStd(ncScp, mvFile, normMethod=None):
    """
    calculate the mean and variance over all .nc in ncScp
    Welford's one line algorithm on mean and population variance
    """
    timeStep = 0
    with open(ncScp, 'r') as filePtr:
        for idx, ncFile in enumerate(filePtr):
            ncFile = ncFile.rstrip('\n')
            data = io.netcdf_file(ncFile)
            print "Processing %s" % (ncFile)
            if idx==0:
                # for the first file, get the dimension of data
                # create the buffer
                inputSize = data.dimensions['inputPattSize']
                outSize   = data.dimensions['targetPattSize']
                meanInBuf = np.zeros([inputSize], dtype=np.float64)
                stdInBuf  = np.zeros([inputSize], dtype=np.float64)
                meanOutBuf = np.zeros([outSize], dtype=np.float64)
                stdOutBuf  = np.zeros([outSize], dtype=np.float64)
                
                # if max min normalization is used, get the max and min value
                if normMethod is not None:
                    maxminInBuf      = np.zeros([inputSize, 2], dtype=np.float64)
                    maxminInBuf[:,0] = data.variables['inputs'][:].max(axis = 0)
                    maxminInBuf[:,1] = data.variables['inputs'][:].min(axis = 0)
                    print "Input max %f\tmin %f" % (maxminInBuf[:,0].max(), maxminInBuf[:,1].min())
                    maxminOutBuf      = np.zeros([outSize, 2], dtype=np.float64)
                    maxminOutBuf[:,0] = data.variables['targetPatterns'][:].max(axis = 0)
                    maxminOutBuf[:,1] = data.variables['targetPatterns'][:].min(axis = 0)
                    print "Output max %f\tmin %f" % (maxminOutBuf[:,0].max(), 
                                                     maxminOutBuf[:,1].min())
                #
            else:
                # for the remaining data files
                if normMethod is not None:
                    tmp = data.variables['inputs'][:].max(axis = 0)
                    maxminInBuf[:,0] = np.maximum(tmp, maxminInBuf[:,0])
                    tmp = data.variables['inputs'][:].min(axis = 0)
                    maxminInBuf[:,1] = np.minimum(tmp, maxminInBuf[:,1])
                    print "Input max %f\tmin %f" % (maxminInBuf[:,0].max(), maxminInBuf[:,1].min())
                    tmp = data.variables['targetPatterns'][:].max(axis = 0)
                    maxminOutBuf[:,0] = np.maximum(tmp, maxminOutBuf[:,0])
                    tmp = data.variables['targetPatterns'][:].min(axis = 0)
                    maxminOutBuf[:,1] = np.minimum(tmp, maxminOutBuf[:,1])
                    print "Output max %f\tmin %f" % (maxminOutBuf[:,0].max(), 
                                                     maxminOutBuf[:,1].min())
            
            numTimes = data.dimensions['numTimesteps']
            print "Processing %s of %s frames" % (ncFile, numTimes)
            print "Input max %f\tmin %f"  % (data.variables['inputs'][:].max(),
                                             data.variables['inputs'][:].min())
            print "Output max %f\tmin %f" % (data.variables['targetPatterns'][:].max(),
                                             data.variables['targetPatterns'][:].min())
            
            for t in xrange(numTimes):
                tmpIn = (data.variables['inputs'][t, :]-meanInBuf)
                meanInBuf = meanInBuf + tmpIn*1.0/(timeStep+t+1)
                tmpOut = (data.variables['targetPatterns'][t, :]-meanOutBuf)
                meanOutBuf = meanOutBuf + tmpOut*1.0/(timeStep+t+1)
                stdInBuf = stdInBuf + tmpIn*(data.variables['inputs'][t, :]-meanInBuf)
                stdOutBuf = stdOutBuf + tmpOut*(data.variables['targetPatterns'][t, :]-meanOutBuf)
            timeStep += numTimes
            data.close()
    stdOutBuf = np.sqrt(stdOutBuf/(timeStep-1))
    stdInBuf  = np.sqrt(stdInBuf/(timeStep-1))
    

    # create MV and save
    f = io.netcdf.netcdf_file(mvFile, 'w')
    f.createDimension('inputPattSize', inputSize)
    f.createDimension('targetPattSize', outSize)
    f.createVariable('inputMeans', 'f', ('inputPattSize',))
    f.createVariable('inputStdevs', 'f', ('inputPattSize', ))
    f.createVariable('outputMeans', 'f', ('targetPattSize',))
    f.createVariable('outputStdevs', 'f', ('targetPattSize', ))
    
    
    if normMethod is not None:
        normIdx = py_rw.read_raw_mat(normMethod, 1, 'i4', 'l')
        assert normIdx.shape[0] == (inputSize + outSize), errorMes([normMethod], 2) 
        inNormIdx  = normIdx[0:inputSize]
        outNormIdx = normIdx[inputSize:(inputSize+outSize)]
        
        f.createVariable('inputMeans_ori',   'f', ('inputPattSize', ))
        f.createVariable('inputStdevs_ori',  'f', ('inputPattSize', ))
        f.createVariable('outputMeans_ori',  'f', ('targetPattSize',))
        f.createVariable('outputStdevs_ori', 'f', ('targetPattSize',))
        meanInBuf_ori, stdInBuf_ori        = meanInBuf.copy(), stdInBuf.copy()
        meanOutBuf_ori, stdOutBuf_ori      = meanOutBuf.copy(), stdOutBuf.copy()
        f.variables['inputMeans_ori'][:]   = np.asarray(meanInBuf_ori, np.float32)
        f.variables['inputStdevs_ori'][:]  = np.asarray(stdInBuf_ori, np.float32)
        f.variables['outputMeans_ori'][:]  = np.asarray(meanOutBuf_ori, np.float32)
        f.variables['outputStdevs_ori'][:] = np.asarray(stdOutBuf_ori, np.float32)
        
        f.createVariable('inputMax_ori',   'f', ('inputPattSize', ))
        f.createVariable('inputMin_ori',  'f', ('inputPattSize', ))
        f.createVariable('outputMax_ori',  'f', ('targetPattSize',))
        f.createVariable('outputMin_ori', 'f', ('targetPattSize',))
        maxInBuf,  minInBuf  = maxminInBuf[:,0].copy(),  maxminInBuf[:,1].copy()
        maxOutBuf, minOutBuf = maxminOutBuf[:,0].copy(), maxminOutBuf[:,1].copy()
        f.variables['inputMax_ori'][:]   = np.asarray(maxminInBuf[:,0],  np.float32)
        f.variables['inputMin_ori'][:]   = np.asarray(maxminInBuf[:,1],  np.float32)
        f.variables['outputMax_ori'][:]  = np.asarray(maxminOutBuf[:,0], np.float32)
        f.variables['outputMin_ori'][:]  = np.asarray(maxminOutBuf[:,1], np.float32)
        
        #if min(inNormIdx) < 0:
        #    negIdx = np.unique(inNormIdx[inNormIdx<0]) # the negative method
        #    for idx in negIdx:
        #        dataIdx   = np.where(inNormIdx == idx)
        #        assert len(dataIdx)>0, 'Impossible error in normMethod'
        #        tempInBuf = stdInBuf.copy()
        #        tempInBuf[np.where(inNormIdx != idx)] = 0
        #        inNormIdx[dataIdx] = np.argmax(tempInBuf)
                
        #if min(outNormIdx) < 0:
        #    negIdx = np.unique(outNormIdx[outNormIdx<0]) # the negative method
        #    for idx in negIdx:
        #        dataIdx   = np.where(outNormIdx == idx)
        #        assert len(dataIdx)>0, 'Impossible error in normMethod'
        #        tempOutBuf = stdOutBuf.copy()
        #        tempOutBuf[np.where(outNormIdx != idx)] = 0
        #        outNormIdx[dataIdx] = np.argmax(tempOutBuf)    
        
        #maxIn, minIn   = max(inNormIdx), min(inNormIdx)
        #maxOut, minOut = max(outNormIdx), min(outNormIdx)
        #assert (maxIn>=0  and maxIn<inputSize),  'inNormIdx out of bound. Please check normMethod'
        #assert (maxOut>=0 and maxOut<inputSize), 'outNormIdx out of bound. Please check normMethod'
        #assert (minIn>=0  and minIn<inputSize),  'inNormIdx out of bound. Please check normMethod'
        #assert (minOut>=0 and minOut<inputSize), 'outNormIdx out of bound. Please check normMethod'
        if min(inNormIdx) < 0:
            tmpMin   = ((1-g_MinMaxRange) * minInBuf - g_MinMaxRange * maxInBuf)/(1-2*g_MinMaxRange)
            tmpMax   = ((1-g_MinMaxRange) * maxInBuf - g_MinMaxRange * minInBuf)/(1-2*g_MinMaxRange)
            
            maxminIndex  = inNormIdx < 0
            meanInBuf[maxminIndex] = tmpMin[maxminIndex]
            stdInBuf[maxminIndex]  = tmpMax[maxminIndex]-tmpMin[maxminIndex]
        if min(outNormIdx) < 0:
            tmpMin   = ((1-g_MinMaxRange)*minOutBuf-g_MinMaxRange*maxOutBuf)/(1-2*g_MinMaxRange)
            tmpMax   = ((1-g_MinMaxRange)*maxOutBuf-g_MinMaxRange*minOutBuf)/(1-2*g_MinMaxRange)

            maxminIndex = outNormIdx < 0
            meanOutBuf[maxminIndex] = tmpMin[maxminIndex]
            stdOutBuf[maxminIndex]  = tmpMax[maxminIndex]-tmpMin[maxminIndex]
        print "Combing maxmin done"

    f.variables['inputMeans'][:] = np.asarray(meanInBuf, np.float32)
    f.variables['inputStdevs'][:] = np.asarray(stdInBuf, np.float32)
    f.variables['outputMeans'][:] = np.asarray(meanOutBuf, np.float32)
    f.variables['outputStdevs'][:] = np.asarray(stdOutBuf, np.float32)

    f.flush()
    f.close()
    print "*** please check max/min above\n"
    print "*** writing done %s\n" % (mvFile) 
    
    

def pre_process(fileScp, maskFile = None):
    """
    return several global informaiton of the data in fileScp:
    1. timeSteps
    2. numSeqs
    3. inputPattSize
    4. targetPattSize
    5. maxSeqLength
    """
    timeSteps = 0
    numSeqs   = 0
    allTxtLength   = 0
    numInputFile   = -1
    inputPattSize  = -1
    numOutputFile  = -1
    targetPattSize = -1
    maxSeqLength   = -1
    maxTxtLength   = -1
    txtPatSize     = -1
    with open(fileScp,'r') as filePtr:
        for idx, fileLine in enumerate(filePtr):
            lineSlots = fileLine.split()
            # log the seq tag length
            if len(lineSlots[0]) > maxSeqLength:
                maxSeqLength = len(lineSlots[0])
            # timeSteps
            timeSteps += int(lineSlots[3])
            # numSeqs  
            numSeqs += 1

            # inputPattSize
            if numInputFile<0:
                numInputFile = int(lineSlots[1])
                # initialize the dim stack
                inputDim = np.zeros([numInputFile, 3], dtype=np.int32)
            elif numInputFile!=int(lineSlots[1]):
                assert 1==0, errorMes(['data_config.py', fileScp], 4) + \
                    "Number of input files incompatible %s" % (fileLine)

            slotBias = 4 # start from the [4] slot
            tmpInputDim = 0
            for idx2 in xrange(numInputFile):
                tmp = int(lineSlots[slotBias+idx2*2])
                tmpInputDim += tmp
                if idx==0:
                    inputDim[idx2, 1] = tmp
                    inputDim[idx2, 2] = tmp
                    
            if inputPattSize<0:
                inputPattSize = tmpInputDim
            elif inputPattSize!=tmpInputDim:
                assert 1==0, errorMes(['data_config.py'], 0) + \
                    "Error, incompatible input data dimension %s" % (fileLine)
            
            # outputPattSize
            if numOutputFile<0:
                numOutputFile = int(lineSlots[2])
                outputDim =np.zeros([numOutputFile, 3], dtype=np.int32)
            elif numOutputFile!=int(lineSlots[2]):
                assert 1==0, errorMes(['data_config.py'], 1) + \
                    "Error, incompatible output data dimension %s" % (fileLine)

            slotBias = 4+numInputFile*2 # skip the head and input part
            tmpOutputDim = 0
            for idx2 in xrange(numOutputFile):
                tmp = int(lineSlots[slotBias+idx2*2])
                tmpOutputDim += tmp
                if idx==0:
                    outputDim[idx2, 1] = tmp
                    outputDim[idx2, 2] = tmp
            if targetPattSize<0:
                targetPattSize = tmpOutputDim
            elif targetPattSize!=tmpOutputDim:
                assert 1==0, "Error, incompatible outputdim %s" % (fileLine)

            if len(lineSlots)==(2*((numOutputFile) + (numInputFile))+4+3):
                # text input data input here
                txtNum = int(lineSlots[2*((numOutputFile) + (numInputFile))+4])
                txtDim = int(lineSlots[2*((numOutputFile) + (numInputFile))+4+1])
                if txtNum > maxTxtLength:
                    maxTxtLength = txtNum
                if txtPatSize < 0:
                    txtPatSize = txtDim
                else:
                    assert txtPatSize==txtDim, "Unequal dimension of txt data %s" % (fileLine)
                allTxtLength = allTxtLength + txtNum                    

    # read the mask
    if maskFile is not None:
        with open(maskFile, mode='r') as filePtr:
            for idx, dim in enumerate(filePtr):
                assert idx<(numInputFile+numOutputFile), "Mask line larger than data file number"
                [sDim, eDim] = dim.split()
                if idx<numInputFile:
                    assert int(eDim)<=inputDim[idx,1],"Mask input larger than dim %s" % (dim)
                    inputDim[idx,0],inputDim[idx,1] = int(sDim), int(eDim)
                    
                else:
                    assert int(eDim)<=outputDim[idx-numInputFile,1],\
                        "Mask output larger than dim %s" % (dim)
                    outputDim[idx-numInputFile,0] = int(sDim)
                    outputDim[idx-numInputFile,1] = int(eDim)
        inputPattSize = (inputDim[:,1]-inputDim[:,0]).sum()
        targetPattSize = (outputDim[:,1]-outputDim[:,0]).sum()
        
    # convert the information to a new format
    # [[startDim in feature vec.][endDim in feature vec.][startDim in raw data][endDim in raw data]]
    inputDim[:,2]   = inputDim[:,1]   - inputDim[:,0]
    tmp = np.concatenate((np.zeros([1], dtype=np.int32), inputDim[:,2].cumsum()))
    inputDimSE = np.zeros([4, numInputFile], dtype=np.int32)
    inputDimSE[2,:] = inputDim.T[0,:]
    inputDimSE[3,:] = inputDim.T[1,:]
    inputDimSE[0,:] = tmp[0:numInputFile]
    inputDimSE[1,:] = inputDim.T[2,:] + inputDimSE[0,:]
    
    outputDim[:,2]  = outputDim[:,1]  - outputDim[:,0]
    tmp = np.concatenate((np.zeros([1], dtype=np.int32), outputDim[:,2].cumsum()))
    outDimSE = np.zeros([4, numOutputFile], dtype=np.int32)
    outDimSE[2,:]   = outputDim.T[0,:]
    outDimSE[3,:]   = outputDim.T[1,:]
    outDimSE[0,:]   = tmp[0:numOutputFile]
    outDimSE[1,:]   = outputDim.T[2,:] + outDimSE[0,:]
    
    return numSeqs, timeSteps, maxSeqLength, inputPattSize, targetPattSize, \
    inputDim, outputDim, inputDimSE.T, outDimSE.T, allTxtLength, maxTxtLength, txtPatSize
    
    
def bmat2nc_sub1(fileScp, outputfile, maskFile=None, flushT=300, waitT=30):
    """ Package the data into .nc file
    one row, one frame of data 
    maskFile: to discard certain dimension of data. Text file, each line specify 
             the start and end column of the single input of output data
              e.g. 0 180   # read the 0-180th column of data1
                   0 3     # read the 0-3th column of data2
                   10 12   # read the 10-12th column of data3
    flushT: after reading this number of utterances, nc block will be flushed to the disk
    waitT:  the number of seconds to wait for the flush process (
            to avoid read and write the disk at the same time)
    """
    numSeqs, timeSteps, maxSeqLength, inputPattSize, outputPattSize, \
        inputDim, outputDim, inputDimSE, outDimSE, \
        allTxtLength, maxTxtLength, txtPatSize  = pre_process(fileScp, maskFile)
    print "Creating nc file %s" % (outputfile)
    print "Input dimension:  %s\n output dimension: %s" % (str(inputPattSize), str(outputPattSize))

    # create the dimension
    if os.path.exists(outputfile):
        print "*** %s exists. It will be overwritten" % (outputfile)
    f = io.netcdf.netcdf_file(outputfile, mode = 'w',version=2)
    f.createDimension('numSeqs', numSeqs)
    f.createDimension('numTimesteps', timeSteps)
    f.createDimension('inputPattSize', inputPattSize)
    f.createDimension('targetPattSize', outputPattSize)
    f.createDimension('maxSeqTagLength', maxSeqLength+1)
    
    tagsVar   = f.createVariable('seqTags', 'S1', ('numSeqs','maxSeqTagLength'))
    seqLVar   = f.createVariable('seqLengths', 'i', ('numSeqs',))
    inputVar  = f.createVariable('inputs', 'f', ('numTimesteps', 'inputPattSize'))
    outVar    = f.createVariable('targetPatterns', 'f', ('numTimesteps', 'targetPattSize'))
    
    #seqLVar  = np.zeros([numSeqs])
    
    seqLVar[:] = 0
    tagsVar[:] = ''
    timeStart  = 0
    count      = 0
    
    with open(fileScp, 'r') as filePtr:
        for idx1, line in enumerate(filePtr):
            
            temp          = line.split()
            print "Reading %s" % (temp[0])
            seqFrame      = int(temp[3])
            seqLVar[idx1] = seqFrame #int(temp[3])
            
            tagsVar[idx1,0:len(temp[0])] = list(temp[0]) #charSeq
            inputFileNum  = int(temp[1])
            outFileNum    = int(temp[2])
            slotBias      = 4
            
            for idx2 in xrange(inputFileNum):
                [sDim, eDim] = inputDimSE[idx2,2:4]         # start, end dimension in raw data
                dim          = int(temp[slotBias+(idx2)*2]) # raw data dim
                datafile     = temp[slotBias+(idx2)*2+1]    # path to raw data
                
                [dS, dE]     = inputDimSE[idx2,0:2]         # start, end dimension in package data
                tS,tE        = timeStart,(timeStart+seqFrame)
                
                if datafile == g_VOIDFILE:
                    data = np.zeros([seqFrame, dim])
                else:
                    data = py_rw.read_raw_mat(datafile, dim)
                assert (data.shape[0]-seqFrame)<seqFrame*0.3, \
                    errorMes([datafile], 3) + "This data has less number of frames" % (datafile)
                if dim==1 and data.ndim==1:
                    #data = data[0:seqFrame]
                    inputVar[tS:tE,dS]    = data[0:seqFrame].copy()
                else:
                    #data = data[0:seqFrame,sDim:eDim]
                    inputVar[tS:tE,dS:dE] = data[0:seqFrame, sDim:eDim].copy()

            slotBias = 4+inputFileNum*2
            for idx2 in xrange(outFileNum):
                [sDim, eDim] = outDimSE[idx2,2:4]
                dim          = int(temp[slotBias+(idx2)*2])
                datafile     = temp[slotBias+(idx2)*2+1]
                [dS,   dE]   = outDimSE[idx2,0:2]
                tS, tE       = timeStart, (timeStart+seqFrame)
                
                if datafile  == g_VOIDFILE:
                    data = np.zeros([seqFrame, dim])
                else:
                    data = py_rw.read_raw_mat(datafile, dim)

                assert (data.shape[0]-seqFrame)<seqFrame*0.1, \
                    errorMes([datafile], 3) + "This data has less number of frames" % (datafile)
                if dim==1 and data.ndim==1:
                    outVar[tS:tE,dS] =data[0:seqFrame].copy()
                else:
                    outVar[tS:tE,dS:dE] = data[0:seqFrame,sDim:eDim].copy()

            #print idx1
            del data
            if count > flushT:
                count = 0
                _write(f) #.flush()
                print "Have read %d. Let's wait netCDF for %d(s)" % (idx1, waitT)
                #raw_input("Enter")
                for x in xrange(waitT):
                    print "*",
                    sys.stdout.flush()
                    time.sleep(1)
            count += 1
            timeStart += seqFrame
    print "Read and write done\n"
    f.close()


def bmat2nc_sub2(fileScp, outputfile, shiftInput, shiftOutput, maskFile=None, flushT=300, waitT=30):
    """ Package the data into .nc file
    one row, one frame of data 
    maskFile: to discard certain dimension of data. Text file, each line specify 
             the start and end column of the single input of output data
              e.g. 0 180   # read the 0-180th column of data1
                   0 3     # read the 0-3th column of data2
                   10 12   # read the 10-12th column of data3
    flushT: after reading this number of utterances, nc block will be flushed to the disk
    waitT:  the number of seconds to wait for the flush process (
            to avoid read and write the disk at the same time)
    """

    numSeqs, timeSteps, maxSeqLength, inputPattSize, outputPattSize, \
        inputDim, outputDim, inputDimSE, outDimSE, \
        allTxtLength, maxTxtLength, txtPatSize  = pre_process(fileScp, maskFile)
    print "Data format input: %s, output: %s" % (str(inputPattSize), str(outputPattSize))
    print "Creating nc file %s" % (outputfile)
    if txtPatSize > 0 and maxTxtLength > 0:
        print "Using txt data, maxlength and dimension %d %d" % (maxTxtLength, txtPatSize)
    
    
    # create the dimension
    if os.path.exists(outputfile):
        print "*** %s exists. It will be overwritten" % (outputfile)
    f = io.netcdf.netcdf_file(outputfile, mode = 'w',version=2)
    f.createDimension('numSeqs', numSeqs)
    f.createDimension('numTimesteps', timeSteps)
    f.createDimension('inputPattSize', inputPattSize)
    f.createDimension('targetPattSize', outputPattSize)
    f.createDimension('maxSeqTagLength', maxSeqLength+1)
    
    if txtPatSize>0 and maxTxtLength > 0:
        f.createDimension('txtLength',   allTxtLength)
        f.createDimension('txtPattSize', txtPatSize)
    

    tagsVar  = f.createVariable('seqTags', 'S1', ('numSeqs','maxSeqTagLength'))
    seqLVar  = f.createVariable('seqLengths', 'i', ('numSeqs',))
    inputVar = f.createVariable('inputs', 'f', ('numTimesteps', 'inputPattSize'))
    outVar   = f.createVariable('targetPatterns', 'f', ('numTimesteps', 'targetPattSize'))

    if txtPatSize>0 and maxTxtLength > 0:
        txtVar   = f.createVariable('txtData',    'i', ('txtLength', 'txtPattSize'))
        txtLVar  = f.createVariable('txtLengths', 'i', ('numSeqs',))
    
        
    
    #seqLVar  = np.zeros([numSeqs])
    seqLVar[:] = 0
    tagsVar[:] = ''
    timeStart = 0
    count = 0

    txtStart  = 0

    with open(fileScp, 'r') as filePtr:
        for idx1, line in enumerate(filePtr):
            temp = line.split()
            print "Reading %s" % (temp[0])
            seqFrame      = int(temp[3])
            seqLVar[idx1] = seqFrame #int(temp[3])
            
            tagsVar[idx1,0:len(temp[0])] = list(temp[0]) #charSeq
            inputFileNum = int(temp[1])
            outFileNum   = int(temp[2])
            slotBias = 4
            
            if txtPatSize>0 and maxTxtLength > 0:
                txtLength = int(temp[slotBias + 2*(inputFileNum + outFileNum)])
                txtDim    = int(temp[slotBias + 2*(inputFileNum + outFileNum)+1])
                txtFile   = temp[slotBias + 2*(inputFileNum + outFileNum) + 2]
                data      = py_rw.read_raw_mat(txtFile, txtDim)
                if txtDim == 1:
                    txtVar[txtStart:(txtStart+txtLength),0] = data.copy()
                else:
                    txtVar[txtStart:(txtStart+txtLength),:] = data.copy()
                txtStart = txtStart + txtLength
                txtLVar[idx1] = txtLength

            for idx2 in xrange(inputFileNum):
                [sDim, eDim] = inputDim[idx2,0:2]
                dim = int(temp[slotBias+(idx2)*2])
                datafile = temp[slotBias+(idx2)*2+1]

                #data_raw = readwrite.FromFile(datafile)
                #m,n = data_raw.size/dim, dim
                #assert m*n==data_raw.size, "dimension mismatch %s %s" % (line, datafile)
                #data = data_raw.reshape((m,n))
                
                # store the data
                tS,tE,dS,dE = timeStart, (timeStart+seqFrame), inputDimSE[idx2][0], \
                              inputDimSE[idx2][1] 
                if datafile == g_VOIDFILE:
                    data = np.zeros([seqFrame, dim])
                else:
                    data = py_rw.read_raw_mat(datafile, dim)
                assert (data.shape[0]-seqFrame)<seqFrame*0.1, \
                    errorMes([datafile], 3) + "This data has less number of frames" % (datafile)
                if dim==1 and data.ndim==1:
                    data = data[0:seqFrame]
                    inputVar[tS:tE,dS] = data[0:seqFrame].copy()
                    
                else:
                    data = data[0:seqFrame,sDim:eDim]
                    
                    inputVar[tS:tE,dS:dE] = data[0:seqFrame, \
                                                 inputDimSE[idx2][2]:inputDimSE[idx2][3]].copy()

            slotBias = 4+inputFileNum*2
            for idx2 in xrange(outFileNum):
                [sDim, eDim] = outputDim[idx2,0:2]
                dim = int(temp[slotBias+(idx2)*2])
                datafile = temp[slotBias+(idx2)*2+1]
                
                #data_raw = readwrite.FromFile(datafile)
                #m,n = data_raw.size/dim, dim
                #assert m*n==data_raw.size, "dimension mismatch %s %s" % (line, datafile)
                #data = data_raw.reshape((m,n))
                
                # read and store the output data
                tS,tE,dS,dE = timeStart, (timeStart+seqFrame), outDimSE[idx2][0], \
                              outDimSE[idx2][1] 

                if datafile == g_VOIDFILE:
                    data = np.zeros([seqFrame, dim])
                else:
                    data = py_rw.read_raw_mat(datafile, dim)

                assert (data.shape[0]-seqFrame)<seqFrame*0.1, \
                    errorMes([datafile], 3) + "This data has less number of frames" % (datafile)
                if dim==1 and data.ndim==1:
                    data = data[0:seqFrame]
                    outVar[tS:tE,dS] =data[0:seqFrame].copy()
                    
                else:

                    data = data[0:seqFrame,sDim:eDim]
                    if shiftOutput != 0:
                        outVar[tS:tE,dS:dE] = np.roll(data, shiftOutput, axis=0)[0:seqFrame, \
                                                       outDimSE[idx2][2]:outDimSE[idx2][3]].copy()
                    else:
                        outVar[tS:tE,dS:dE] = data[0:seqFrame, \
                                                       outDimSE[idx2][2]:outDimSE[idx2][3]].copy()
                    

            #print idx1
            del data
            if count > flushT:
                count = 0
                _write(f) #.flush()
                print "Have read %d. Let's wait netCDF for %d(s)" % (idx1, waitT)
                #raw_input("Enter")
                for x in xrange(waitT):
                    print "*",
                    sys.stdout.flush()
                    time.sleep(1)
            count += 1
            timeStart += seqFrame
    print "Reading and writing done " 
    f.close()


def bmat2nc(fileScp, outputfile, maskFile=None, shiftInput=-1, shiftOutput=-1, flushT=1000, waitT=5):
    if shiftInput < 0 and shiftOutput < 0:
        bmat2nc_sub1(fileScp, outputfile, maskFile, flushT, waitT)
    else:
        bmat2nc_sub2(fileScp, outputfile, shiftInput, shiftOutput, maskFile, flushT, waitT)

if __name__ == "__main__":

    #var1 = '/work/smg/wang/TEMP/code/CURRENNT/examples/mdn/data_speech_gen/data/all.scp1'
    #var2 = '/work/smg/wang/TEMP/code/CURRENNT/examples/mdn/data_speech_gen/data/data.nc1'
    #bmat2nc(var1, var2, None, 0, 1)
    pass
    #fileScp = "/home/smg/wang/PROJ/WE/DNNAM/DATA/nancy/nancy_phone/all.scp11train"
    #fileScp = 'all.scps'
    #fileScp  = "all_1.scp"
    #outputfile = "temp2.nc"
    #mask =  None
    #bmat2nc(fileScp, outputfile, mask)
    #meanStd("data.scp", "temp2.mv")
    #norm("data/data.nc1", "temp2.mv", "data/data.nc1norm")
    #compennom('/home/smg/wang/PROJ/WE/DNNAM/DATA/nancy/testset_align/testset_phone_rnnwe_a/data.nc1', '/home/smg/wang/PROJ/WE/DNNAM/DATA/nancy/nancy_phone_rnnwe_a/data.mv', "data/data.nc1nomshift", '/home/smg/wang/PROJ/WE/DNNAM/DATA/nancy/nancy_phone_nomshift/meanMask')
    
    #mode = sys.argv[1]
    #scp = '/home/smg/wang/PROJ/WE/DNNAM/DATA/nancy/testset_align/testset_phone_rnnwe_a/all.scp1'
    #datamv = '/home/smg/wang/PROJ/WE/DNNAM/DATA/nancy/nancy_phone_rnnwe_a/data.mv'
    #datamask = '/home/smg/wang/PROJ/WE/DNNAM/DATA/nancy/testset_align/testset_phone_rnnwe_a/mask.txt'
    #out = '~/TEMP/data/test.nc1nomsfhit'
    #mask = '/home/smg/wang/PROJ/WE/DNNAM/DATA/nancy/nancy_phone_nomshift/meanMask'
    #bmat2nc(scp, out, datamask)
    #norm(out, datamv, flagKeepOri=0)
    #compennom(out, datamv, None, flagKeepOri=0, mask=mask)
    
