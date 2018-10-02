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


def PrepareScp(InScpFile, OutScpFile, inDim, outDim, allScp):

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
                        print "Trim %d for %s" % (addiFrame,fileline)
                        
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
            
                #print line                
                fileOutBuffer.append(fileline)
                tempFrame = funcs.Bytes(fileline, dim)/np.dtype(dataType).itemsize
                if seqLenBuffer[fileCtr+1]>tempFrame:
                    addiFrame = seqLenBuffer[fileCtr+1]-tempFrame
                    seqLenBuffer[fileCtr+1]=tempFrame
                    print "Trim %d for %s " % (addiFrame, fileline)
                
                fileCtr = fileCtr + 1
                print "Output %d %d\r" % (scpIndex, fileCtr),
                sys.stdout.flush()
            print ""
            fPtr.close()
        assert len(fileOutBuffer)-1==(len(outDim)*numSeque), "Unequal file output numbers"

    # Write the scp for packaging data
    scpFileCtr = 1
    fileCtr = 0
    fPtr = open(allScp+str(scpFileCtr), mode='w')
    nameBuf = ['', ]
    numFrameBuf = [0, ]
    numUttBuf  = [0, ]
    frameBuf = 0
    for i in xrange(numSeque):
        outputline = "%s %d %d %d" %  \
                     (
                         os.path.basename(fileLabBuffer[i+1])[:-4], 
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
    Generating the Mask.txt
    
    """
    assert len(tmp_inDim)==len(tmp_inScpFile), "Unequal inDim and inScpFile"
    assert len(tmp_inDim)==len(tmp_inMask), "Unequal inDim and tmp_inMask"
    assert len(tmp_outDim)==len(tmp_outMask), "Unequal outDim and tmp_outMask"
    inDim = []
    outDim = []
    inScpFile = []
    outScpFile = []
    filePtr = open(scpdir + os.path.sep + 'mask.txt', 'w')
    for index, fileMask in enumerate(tmp_inMask):
        if len(fileMask)==0:
            inDim.append(tmp_inDim[index])
            inScpFile.append(scpdir+ os.path.sep + tmp_inScpFile[index])
            tempLine = "0 %d\n" % tmp_inDim[index]
            filePtr.write(tempLine)
        else:
            assert len(fileMask)%2==0, "inMask should have even number of data"
            for index2, tmpDim in enumerate(fileMask[0::2]):
                inDim.append(tmp_inDim[index])
                inScpFile.append(scpdir+ os.path.sep + tmp_inScpFile[index])
                tempLine = "%d %d\n" % (fileMask[2*index2], fileMask[2*index2+1])
                filePtr.write(tempLine)
        if len(tmp_outScpFile)==0:
            pass
            outDim = tmp_outDim
        else:
            assert len(tmp_outDim)==len(tmp_outScpFile), "Unequal outDim and outScpFile"
    for index, fileMask in enumerate(tmp_outMask):
        if len(fileMask)==0:
            outDim.append(tmp_outDim[index])
            outScpFile.append(scpdir+ os.path.sep + tmp_outScpFile[index])
            tempLine = "0 %d\n" % tmp_outDim[index]
            filePtr.write(tempLine)
        else:
            assert len(fileMask)%2==0, "inMask should have even number of data"
            for index2, tmpDim in enumerate(fileMask[0::2]):
                outDim.append(tmp_outDim[index])
                outScpFile.append(scpdir+ os.path.sep + tmp_outScpFile[index])
                tempLine = "%d %d\n" % (fileMask[2*index2], fileMask[2*index2+1])
                filePtr.write(tempLine) 
    filePtr.close()
    return inDim, inScpFile, outDim, outScpFile






if __name__ == "__main__":
    
    print "Generating the Scp File"
    if 'inMask' in dir(cfg) and 'outMask' in dir(cfg):
        [inDim, inScpFile, outDim, outScpFile] = PreProcess(
            cfg.inDim, cfg.outDim, cfg.inScpFile, cfg.outScpFile, 
            cfg.inMask, cfg.outMask, scpdir)
    else:
        inDim = cfg.inDim
        outDim = cfg.outDim
        inScpFile = []
        outScpFile = []
        for file in cfg.inScpFile:
            inScpFile.append(scpdir + os.path.sep + file)
        for file in    cfg.outScpFile:
            outScpFile.append(scpdir + os.path.sep + file)
            
    allScp = scpdir + os.path.sep + cfg.allScp

    if os.path.exists(allScp + ".info"):
        [numSeque, numFrame, maxSeqLe, 
         FrameBuf, numUttBuf, nameBuf] = pickle.load(open(allScp+".info", "rb"))
    else:
        [numSeque, numFrame, maxSeqLe, 
         FrameBuf, numUttBuf, nameBuf] = PrepareScp(inScpFile, outScpFile, inDim, outDim, allScp)
        pickle.dump([numSeque, numFrame, maxSeqLe, FrameBuf, numUttBuf, nameBuf], 
                    open(allScp+".info", "wb"))
    
    print "Number of utternaces: "+str(numSeque)
    print "Number of frames: "+str(numFrame)
    print "Max utterance lab length: "+str(maxSeqLe)
    print "Generating: "+ str(nameBuf)
    print "Number of frames per file: "+ str(FrameBuf)
    print "Number of utterances per file: "+ str(numUttBuf)
    
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
