import numpy as np
import scipy
from scipy import io
import sys
import os
import pickle

#import funcs

try:
    from binaryTools import readwriteC2 as funcs
except ImportError:
    try:
        from binaryTools import readwriteC2_220 as funcs
    except ImportError:
        try: 
            from ioTools import readwrite as funcs
        finally:
            print "Please add ~/CODE/pyTools to PYTHONPATH"
            raise Exception("Can't not import binaryTools/readwriteC2 or funcs")

sys.path.append(os.path.dirname(sys.argv[1]))
cfg = __import__(os.path.basename(sys.argv[1])[:-3])
dataType = cfg.dataType
flushThreshold = cfg.flushThreshold
scpdir = sys.argv[2]


def PrepareScp(InScpFile, OutScpFile, inDim, outDim, allScp, datadir, txtScp, txtDim):
    """ Count the number of frames of each input and output file.
    """
    assert len(InScpFile)==len(inDim), \
        "Unequal length of input scp and in dim"
    assert len(OutScpFile)==0 or len(OutScpFile)==len(outDim), \
        "Unequal length of output scp and in dim"
    
    numSeque = 0                                    # number of sequence
    numFrame = 0                                    # number of total frame
    inPatDim = 0                                    # dimension of input 
    ouPatDim = 0                                    # dimension of output
    maxSeqLe = 0                                    # maximum length of name sequence
    fileLabBuffer = ['',]
    fileInBuffer = ['',]
    fileOutBuffer = ['',]
    seqLenBuffer = [0,]
    
    # Pre-process the input file
    #  check the duration of each input file
    #  keep the shortest duration of input files for one entry
    
    print "\nNote: Different feature files of one utterance may contain different number of frames."
    print "Trim value shows how many frames are discarded in order to match the shortest file."
    print "Large Trim value indicates that the dimension in data_config.py or the extracted feature"
    print "file may be ill. Please check it carefully if it happens!\n"
    print "Processing the input file"
    for scpFile, dim, scpIndex in zip(InScpFile, inDim, xrange(len(InScpFile))):
        fPtr = open(scpFile,'r')
        fileCtr = 0
        
        while 1:
            line = fPtr.readline()
            if len(line.strip())==0:
                break
            
            if scpIndex==0:
                numSeque = numSeque + 1
            fileline = line.strip()
            
            # only check for relative path
            if not os.path.isfile(fileline):
                fileline = datadir + os.path.sep + fileline                
            assert os.path.isfile(fileline), "Can't find file"+fileline

            if scpIndex==0:                         # loading lab file
                fileLabBuffer.append(fileline)
                if len(fileline) >  maxSeqLe:
                    maxSeqLe = len(fileline)
            
            fileInBuffer.append(fileline)           # check the time step of file
            
            if scpIndex>0 and fileline==fileLabBuffer[fileCtr+1]:
                pass                     # if this file is the same as the input lab file
                                         # pass it (for the case when several columns of the 
                                         # input lab will be extracted based on inputMask)
            else:
                tempFrame = funcs.Bytes(fileline, dim)/np.dtype(dataType).itemsize
                if scpIndex==0:
                    numFrame = numFrame + tempFrame
                    seqLenBuffer.append(tempFrame)
                else:
                    if seqLenBuffer[fileCtr+1]>tempFrame:
                        addiFrame = seqLenBuffer[fileCtr+1]-tempFrame
                        seqLenBuffer[fileCtr+1]=tempFrame
                        print "Trim %d to fit %s" % (addiFrame,fileline)
                    elif seqLenBuffer[fileCtr+1]<tempFrame:
                        addiFrame = -1*seqLenBuffer[fileCtr+1]+tempFrame
                        print "Trim %d from %s" % (addiFrame,fileline)
                    
                        
            fileCtr = fileCtr + 1
            print "Input %d %d\r" % (scpIndex, fileCtr),
            sys.stdout.flush()
            #sys.stdout.write("\rInput:"+str(fileCtr))
        print ""
        if fPtr.tell() == os.fstat(fPtr.fileno()).st_size:
            flagTer = True                            # all files have been processed
        fPtr.close()    
    assert len(fileInBuffer)-1==(len(inDim)*numSeque), "Unequal file input numbers"
    
    
    # Pre-process the output file
    #  check the duration of output file
    print "Processing the output file"
    if len(OutScpFile)==0:                    # void output files
        for dim in outDim:
            for x in range(numSeque):
                fileOutBuffer.append('#')
    else:                                    # multiple output files
        for scpFile, dim, scpIndex in zip(OutScpFile, outDim, xrange(len(OutScpFile))):
            fPtr = open(scpFile,'r')
            fileCtr = 0
        
            while 1:
                line = fPtr.readline()
                fileline = line.strip()
                if len(line.strip())==0:
                    break
                
                # only check for relative path
                if not os.path.isfile(fileline):
                    fileline = datadir + os.path.sep + fileline
                assert os.path.isfile(fileline), "Can't find file"+fileline

                #print line                
                fileOutBuffer.append(fileline)
                tempFrame = funcs.Bytes(fileline, dim)/np.dtype(dataType).itemsize
                if seqLenBuffer[fileCtr+1]>tempFrame:
                    addiFrame = seqLenBuffer[fileCtr+1]-tempFrame
                    seqLenBuffer[fileCtr+1]=tempFrame
                    print "Trim %d to fit %s " % (addiFrame, fileline)
                elif seqLenBuffer[fileCtr+1]<tempFrame:
                    addiFrame = -1*seqLenBuffer[fileCtr+1]+tempFrame
                    print "Trim %d from %s" % (addiFrame,fileline)
                
                fileCtr = fileCtr + 1
                print "Output %d %d\r" % (scpIndex, fileCtr),
                sys.stdout.flush()
            print ""
            fPtr.close()
        assert len(fileOutBuffer)-1==(len(outDim)*numSeque), "Unequal file output numbers"

    # if text scp exists, check
    if len(txtScp) > 0:
        textBuffer = []
        with open(txtScp, 'r') as filePtr:
            for i, fileline in enumerate(filePtr):
                filename = fileline.rstrip('\n')
                name1 = os.path.splitext(os.path.basename(filename))[0]
                name2 = os.path.splitext(os.path.basename(fileLabBuffer[i+1]))[0]
                assert name1==name2, "textScpFile unmatch %s, %s, %d-th line" % (name1, name2, i)
                textBuffer.append(filename)
        assert len(textBuffer)==(len(fileLabBuffer)-1), "textScpFile, unmatched length"
    else:
        textBuffer = []
        
    # Write the scp for packaging data
    scpFileCtr = 1
    fileCtr = 0
    fPtr = open(allScp+str(scpFileCtr), mode='w')
    nameBuf = ['', ]
    numFrameBuf = [0, ]
    numUttBuf  = [0, ]
    frameBuf = 0
    assert numSeque, "Found no utterance to pack"
        
    for i in xrange(numSeque):
        outputline = "%s %d %d %d" %  \
                     (
                         os.path.splitext(os.path.basename(fileLabBuffer[i+1]))[0], 
                         len(inDim), 
                         len(outDim), 
                         seqLenBuffer[i+1]
                     )
        frameBuf = frameBuf + seqLenBuffer[i+1]
        
        for j in xrange(len(inDim)):
            index = (j)*numSeque+i+1
            outputline = outputline + " %d %s" % (inDim[j], fileInBuffer[index])
        for j in xrange(len(outDim)):
            index = (j)*numSeque+i+1
            outputline = outputline + " %d %s" % (outDim[j], fileOutBuffer[index])
        fileCtr = fileCtr + 1
        fPtr.write(outputline)
        
        if len(textBuffer) > 0:
            assert os.path.isfile(textBuffer[i]), "Can't find %s" % (textBuffer[i])
            itemNum = funcs.Bytes(textBuffer[i], txtDim)/np.dtype(dataType).itemsize
            temp = " %d %d %s" % (itemNum, txtDim, textBuffer[i])
            fPtr.write(temp)

        fPtr.write("\n")
        flagLock = False
        
        if fileCtr >= flushThreshold:
            numUttBuf.append(fileCtr)
            numFrameBuf.append(frameBuf)
            fileCtr = 0
            frameBuf = 0        
            fPtr.close()
            nameBuf.append(allScp+str(scpFileCtr))
            scpFileCtr = scpFileCtr + 1
            fPtr = open(allScp+str(scpFileCtr), mode='w')
            flagLock = True
    if flagLock==False:    
        nameBuf.append(allScp+str(scpFileCtr))
        numUttBuf.append(fileCtr)
        numFrameBuf.append(frameBuf)
    fPtr.close()    

    # Return
    assert sum(numUttBuf)==numSeque, "Unequal utterance number"
    return numSeque, numFrame, maxSeqLe, numFrameBuf[1:], numUttBuf[1:], nameBuf[1:]


def PreProcess(tmp_inDim, tmp_outDim, tmp_inScpFile, tmp_outScpFile, tmp_inMask, 
    tmp_outMask, scpdir):
    """ 
    Generating the Mask.txt for feature extraction
    
    """
    assert len(tmp_inDim)==len(tmp_inScpFile), "Unequal inDim and inScpFile"
    assert len(tmp_inDim)==len(tmp_inMask), "Unequal inDim and tmp_inMask"
    assert len(tmp_outDim)==len(tmp_outMask), "Unequal outDim and tmp_outMask"
    inDim      = []
    outDim     = []
    inScpFile  = []
    outScpFile = []
    validInDim = []
    validOutDim= []
    filePtr = open(scpdir + os.path.sep + 'mask.txt', 'w')
    for index, fileMask in enumerate(tmp_inMask):
        if len(fileMask)==0:
            inDim.append(tmp_inDim[index])
            inScpFile.append(scpdir+ os.path.sep + tmp_inScpFile[index])
            tempLine   = "0 %d\n" % tmp_inDim[index]
            filePtr.write(tempLine)
            validInDim.append(tmp_inDim[index])
        else:
            assert len(fileMask)%2==0, "inMask should have even number of data"
            for index2, tmpDim in enumerate(fileMask[0::2]):
                inDim.append(tmp_inDim[index])
                inScpFile.append(scpdir+ os.path.sep + tmp_inScpFile[index])
                tempLine = "%d %d\n" % (fileMask[2*index2], fileMask[2*index2+1])
                filePtr.write(tempLine)
                validInDim.append((fileMask[2*index2+1] - fileMask[2*index2]))
                
    for index, fileMask in enumerate(tmp_outMask):
        if len(fileMask)==0:
            outDim.append(tmp_outDim[index])
            if len(tmp_outScpFile)>0:
                outScpFile.append(scpdir+ os.path.sep + tmp_outScpFile[index])
            tempLine    = "0 %d\n" % tmp_outDim[index]
            filePtr.write(tempLine)
            validOutDim.append(tmp_outDim[index])
        else:
            assert len(fileMask)%2==0, "inMask should have even number of data"
            assert len(tmp_outScpFile)>0, "mask is used but no output file scp"
            for index2, tmpDim in enumerate(fileMask[0::2]):
                outDim.append(tmp_outDim[index])
                outScpFile.append(scpdir+ os.path.sep + tmp_outScpFile[index])
                tempLine = "%d %d\n" % (fileMask[2*index2], fileMask[2*index2+1])
                filePtr.write(tempLine) 
                validOutDim.append(fileMask[2*index2+1] - fileMask[2*index2])
    filePtr.close()
    return inDim, inScpFile, outDim, outScpFile, validInDim, validOutDim



def normMaskGen(inDim, outDim, normMask):
    """ Generating the normMask for packaging the data
    """
    assert len(inDim)+len(outDim)==len(normMask), "Unequal length normMask and inDim outDim"
    inDim = np.array(inDim)
    outDim = np.array(outDim)
    dimAll = np.concatenate((inDim, outDim))
    
    dimVec = np.ones([dimAll.sum()])
    dimS = 0
    for idx, dim in enumerate(normMask):
        if len(dim)==2:
            # [start end] 
            nS, nE = dim[0] + dimS, dim[1] + dimS
            assert nS>=dimS, "start of normMask smalled than feature dimension"
            assert nE<=(dimS+dimAll[idx]), "end of normMask larger than feature dimension"
            dimVec[nS:nE] = 0
        elif len(dim)==1 and dim[0]==0:
            # [0] all to zero
            dimVec[dimS:(dimS+dimAll[idx])] = 0
        else:
            # nothing []
            pass
        dimS = dimS + dimAll[idx]
    
    print dimVec
    print "Mask shape:"+str(dimVec.shape)
    funcs.write_raw_mat(dimVec, scpdir + os.path.sep + 'normMask')
    print "Writing norMask to %s " % (scpdir + os.path.sep + 'normMask')


def normMethodGen(inDim, outDim, inNormIdx, outNormIdx, scpdir):
    """ Generating the idx file for normalization method
        input:  inNormIdx and outNormIdx: [normMethod, [parameter_1], ...]
        output: 
    """
    inNormIdxData  = np.arange(0, sum(inDim),  dtype=np.int32)
    outNormIdxData = np.arange(0, sum(outDim), dtype=np.int32)
    
    for configIn in inNormIdx:
        assert len(configIn)==3, 'inNormIdx element should be [start_d, end_d, h] format'
        assert configIn[0]>=0, 'start_d in [start_d, end_d, h] is below 0'
        assert configIn[0]<=sum(inDim), 'start_d in [start_d, ...] is larger than input dimension'
        assert configIn[1]>=0, 'start_e in [start_d, end_d, h] is below 0'
        assert configIn[1]<=sum(inDim), 'start_e in [start_d, ...] is larger than input dimension'
        if configIn[2]>=0:
            assert configIn[2]<sum(inDim), 'h in [start_d, end_d, h] is larger than input dimension'
            inNormIdxData[configIn[0]:configIn[1]] = configIn[2]
        else:
            inNormIdxData[configIn[0]:configIn[1]] = configIn[2]

    for configIn in outNormIdx:
        assert len(configIn)==3, 'inNormIdx element should be [start_d, end_d, h] format'
        assert configIn[0]>=0, 'start_d in [start_d, end_d, h] is below 0'
        assert configIn[0]<=sum(outDim), 'start_d in [start_d, ...] is larger than output dim'
        assert configIn[1]>=0, 'start_e in [start_d, end_d, h] is below 0'
        assert configIn[1]<=sum(outDim), 'start_e in [start_d, ...] is larger than input dim'
        if configIn[2]>=0:
            assert configIn[2]<sum(outDim), 'h in [start_d, end_d, h] is larger than output dim'
            outNormIdxData[configIn[0]:configIn[1]] = configIn[2]
        else:
            outNormIdxData[configIn[0]:configIn[1]] = configIn[2]

    normIdxData = np.append(inNormIdxData, outNormIdxData)
    funcs.write_raw_mat( normIdxData, scpdir + os.path.sep +  'normMethod', 'i4', 'l')
    #funcs.write_raw_mat(outNormIdxData, scpdir + os.path.sep + 'outNormMethod', 'i4', 'l')
    

if __name__ == "__main__":
    
    # Generating the feature mask file
    print "====== Generating the Scp File ======"
    if 'inMask' in dir(cfg) and 'outMask' in dir(cfg):
        print "Generating the Mask file"
        [inDim, inScpFile, outDim, outScpFile, valInDim, valOutDim] = PreProcess(
            cfg.inDim, cfg.outDim, cfg.inScpFile, cfg.outScpFile, 
            cfg.inMask, cfg.outMask, scpdir)
    else:
        print "No Mask configuration"
        inDim    =  cfg.inDim
        outDim   =  cfg.outDim
        valInDim =  cfg.inDim
        valOutDim=  cfg.outDim
        inScpFile = []
        outScpFile = []
        for fileName in cfg.inScpFile:
            inScpFile.append(scpdir + os.path.sep + fileName)
        for fileName in    cfg.outScpFile:
            outScpFile.append(scpdir + os.path.sep + fileName)
    
    # Generating the normlization mask file
    if 'normMask' in dir(cfg):
        print "Generating normMask"
        normMaskGen(valInDim, valOutDim, cfg.normMask)
    else:
        print "No normMask configuration"
        if os.path.isfile("./normMask"):
            os.system("rm ./normMask")

    # Generating the normaization method idx
    if 'inNormIdx' in dir(cfg) or 'outNormIdx' in dir(cfg):
        if not 'inNormIdx' in dir(cfg):
            inNormIdx = []
        else:
            inNormIdx = cfg.inNormIdx
        if not 'outNormIdx' in dir(cfg):
            outNormIdx = []
        else:
            outNormIdx = cfg.outNormIdx
        normMethodGen(valInDim, valOutDim, inNormIdx, outNormIdx, scpdir)
    else:
        print "No normMethod configuration"
        if os.path.isfile("./normMethod"):
            os.system("rm ./normMethod")
            
    # Generating the txt file (optional)
    if 'textScpFile' in dir(cfg) and 'textDim' in dir(cfg):
        print "Found input text file"
        txtScp = scpdir + os.path.sep + cfg.textScpFile
        txtDim = cfg.textDim
    else:
        txtScp = ''
        txtDim = 0

    # Generating the scp files for data packaging
    allScp = scpdir + os.path.sep + cfg.allScp

    if os.path.exists(allScp + ".info"):
        [numSeque, numFrame, maxSeqLe, 
         FrameBuf, numUttBuf, nameBuf] = pickle.load(open(allScp+".info", "rb"))
    else:
        [numSeque, numFrame, maxSeqLe, 
         FrameBuf, numUttBuf, nameBuf] = PrepareScp(inScpFile, outScpFile, inDim, outDim, allScp,
                                                    scpdir, txtScp, txtDim)
        pickle.dump([numSeque, numFrame, maxSeqLe, FrameBuf, numUttBuf, nameBuf], 
                    open(allScp+".info", "wb"))
    print "\n======     Data statistics     ======"
    print "Number of utternaces:          "+str(numSeque)
    print "Number of frames:              "+str(numFrame)
    print "Max utterance lab length:      "+str(maxSeqLe)
    print "\n"
    print "Generating scps to:            "
    for x in nameBuf:
        print "\t%s" % (x)
    print "Number of frames per file:     "+ str(FrameBuf)
    print "Number of utterances per file: "+ str(numUttBuf)
    print "\n"
    
    if len(nameBuf)>0:
        filePtr = open(allScp, 'w')
        for files in nameBuf:
            filePtr.write(files+'\n')
        filePtr.close()
        print "Generating " + allScp + " as batch file"
    
    # no longer needed
    #PrepareData(nameBuf, inDim, outDim, maxSeqLe, FrameBuf, numUttBuf, ncFile)
    #GetStatistics(nameBuf, FrameBuf, sum(inDim), sum(outDim), ncFile)
    #NormalizeData(nameBuf, FrameBuf, sum(inDim), sum(outDim), ncFile)
