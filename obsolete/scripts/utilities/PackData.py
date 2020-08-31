#!/usr/bin/python
"""
This script create the ncFile locally.
Take care when the size of data is huge

INPUT:
    filescp: the scp that stores the path to scripts of feature
             e.g. 
                path_to_scp1
                path_to_scp2
    mask:    the text binary mask file to mask part of the dimension
             of the data (this mask is not used for normalization)
    buff:    remote local disk that will store the .nc file 
             if buff='-', the same directory as filescp will be used to
             store the data
OUTPUT:
    datascp: the path to the script of nc file that will be generated
    mv:      the path to the mean and variance data that will be generated

CONFIGURATION:
    step1:   1. pack data
             0. not pack data
    step2:   1. calculate the mean and variance
             0. not calculate the mean and variance
    step3:   1. normalise the data
             0. not normalize the data

"""
import re, sys, os, traceback
try:
    from dataTools import ncDataHandle as nc
except ImportError:
    raise Exception("*** Add the path of pyTools to PYTHONPATH ")
    
if __name__ == "__main__":
    filescp = sys.argv[1]
    mv      = sys.argv[2]
    buff    = sys.argv[3]
    
    datascp = re.sub(r'all.scp', r'data.scp', filescp)
    step1   = int(sys.argv[4])
    step2   = int(sys.argv[5])
    step3   = int(sys.argv[6])
    mask    = sys.argv[7]
    if mask=='None':
        mask = None
        
    if len(sys.argv)<9:
        addMV = 0
    else:
        addMV = int(sys.argv[8])

    if len(sys.argv)<10:
        normMask = None
    else:
        normMask = sys.argv[9]
        if normMask =='None':
            normMask = None
    
    if len(sys.argv)<11:
        normMethod = None
    else:
        normMethod = sys.argv[10]
        if normMethod == 'None':
            normMethod =  None
        
    if step1==1:
        print "===== Reading and loading data ====="
        filePtr2 = open(datascp, 'w') 
        with open(filescp, 'r') as filePtr:
            for idx, line in enumerate(filePtr):
                flagDataValid = True
                line = line.rstrip('\n')

                if buff != '-':
                    dataline = os.path.basename(line)
                    dataline = re.sub(r'all.scp', r'data.nc', dataline)
                    dataline = buff + os.path.sep + dataline
                else:
                    dataline = re.sub(r'all.scp', r'data.nc', line)
                assert os.path.isfile(line), "Can't find %s" % (line)

                try:
                    nc.bmat2nc(line, dataline, mask)
                except:
                    flagDataValid = False
                    print "Unexpected error:", sys.exc_info()[0]
                    print traceback.extract_tb(sys.exc_info()[-1])
                    raise Exception("*** Fail to pack data %s" % (line))
                
            
                if flagDataValid is True:
                    filePtr2.write(dataline+'\n')
        filePtr2.close()
    else:
        print "===== Skip reading and loading data ====="
        
    if step2==1:
        print "===== Calculating mean and variance ====="
        try:
            nc.meanStd(datascp, mv, normMethod)
        except:
            print "Unexpected error:", sys.exc_info()[0]
            print traceback.extract_tb(sys.exc_info()[-1])
            print "*** Fail to generate %s" % (mv)
            raise Exception("*** Fail to get mean and std. %s" % (line))

    else:
        print "===== Skip calculating mean and variance ====="

    if step3==1:
        print "===== Normalize data.nc ====="
        assert os.path.isfile(mv), "*** Fail to locate %s" % (mv)

        with open(datascp, 'r') as filePtr:
            for idx, line in enumerate(filePtr):
                flagDataValid = True
                line = line.rstrip('\n')
                assert os.path.isfile(line), "Can't find %s" % (line)
                try:
                    nc.norm(line, mv, flagKeepOri=0, addMV=addMV, mask=normMask)
                except:
                    print "Unexpected error:", sys.exc_info()[0]
                    print traceback.extract_tb(sys.exc_info()[-1])
                    raise Exception("*** Fail to norm %s" % (line))
                    flagDataValid = False
                    print "*** Not sure where %s is still valid" % (line) 
    else:
        print "===== skip Normalizing data.nc ====="            
