#!/usr/bin/python
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import multiprocessing
import os
import sys


def tempWarpper(scriptDir, fileLst, batch, mlpgFlag, mainDataDir, nndataDir, batchNum, postFilter, vocoder):
    if vocoder == 'WORLD':
        command = "perl %s/wavGen_WORLD/Synthesis.pl %s/wavGen_WORLD/Config.pm" % (scriptDir, scriptDir)
        command = command + " %s/wavGen_WORLD/Utils.pm DNN_GNWAV %d" % (scriptDir, batch)
    elif vocoder == 'STRAIGHT':
        command = "perl %s/wavGen_STRAIGHT/Synthesis.pl %s/wavGen_STRAIGHT/Config.pm" % (scriptDir, scriptDir)
        command = command + " %s/wavGen_STRAIGHT/Utils.pm DNN_GNWAV %d" % (scriptDir, batch)
    else:
        print("Unknown vocoder type %s" % (vocoder))
        
    command = command + " %d 0 %s %s"  % (mlpgFlag, fileLst, mainDataDir)
    command = command + " %s %d 1 0"   % (nndataDir, batchNum)
    command = command + " %s %s %s %f" % (mainDataDir, mainDataDir, mainDataDir, postFilter)
    print(command)
    os.system(command)
    
if __name__ == "__main__":
    
    
    mainDataDir = sys.argv[1]
    postFilter  = float(sys.argv[2])

    mgcDir      = sys.argv[3]
    lf0Dir      = sys.argv[5]
    bapDir      = sys.argv[7]
    mlpgFlag    = [int(sys.argv[4]), int(sys.argv[6]), int(sys.argv[8])]

    fileLst     = sys.argv[9]
    vocoder     = sys.argv[10]
    nndataDir   = sys.argv[11]
    
    try:
        scriptDir =sys.argv[12]
    except IndexError:
        scriptDir   = "./utilities"
        
    batchNum    = 10

    
    try:
        os.system("rm %s/*.wav" % (mainDataDir))
    except OSError:
        pass
    
    with open(fileLst, 'r') as filePtr:
        for idx, fileName in enumerate(filePtr):
            fileName = fileName.rstrip('\n')
            #fileName = '.'.join(fileName.split('.')[0:-1])
            fileBaseName = os.path.splitext(os.path.basename(fileName))[0]
            if mgcDir != mainDataDir:
                if mlpgFlag[0] == 0:
                    os.system("ln -f -s %s/%s.mgc %s/%s.mgc" % (mgcDir, fileBaseName, mainDataDir, fileBaseName));
                else:
                    os.system("ln -f -s %s/%s.mgc_delta %s/%s.mgc_delta" % (mgcDir, fileBaseName, mainDataDir, fileBaseName));
            else:
                if mlpgFlag[0] == 0:
                    os.system("rm %s/%s.mgc_delta" % (mgcDir, fileBaseName));
                else:
                    os.system("rm %s/%s.mgc" % (mgcDir, fileBaseName));
                    
                    
            if lf0Dir != mainDataDir:
                if mlpgFlag[1] == 0:
                    os.system("ln -f -s %s/%s.lf0 %s/%s.lf0" % (lf0Dir, fileBaseName, mainDataDir, fileBaseName));
                else:
                    os.system("ln -f -s %s/%s.lf0_delta %s/%s.lf0_delta" % (lf0Dir, fileBaseName, mainDataDir, fileBaseName));
                    os.system("ln -f -s %s/%s.vuv %s/%s.vuv" % (lf0Dir, fileBaseName, mainDataDir, fileBaseName));
            else:
                if mlpgFlag[1] == 0:
                    os.system("rm %s/%s.lf0_delta" % (lf0Dir, fileBaseName));
                else:
                    os.system("rm %s/%s.lf0" % (lf0Dir, fileBaseName));
                    
                    
            if bapDir != mainDataDir:
                if mlpgFlag[2] == 0:
                    os.system("ln -f -s %s/%s.bap %s/%s.bap" % (bapDir, fileBaseName, mainDataDir, fileBaseName));
                else:
                    os.system("ln -f -s %s/%s.bap_delta %s/%s.bap_delta" % (bapDir, fileBaseName, mainDataDir, fileBaseName));
            else:
                if mlpgFlag[2] == 0:
                    os.system("rm %s/%s.bap_delta" % (bapDir, fileBaseName));
                else:
                    os.system("rm %s/%s.bap" % (bapDir, fileBaseName));
    batch = 1
    pool = multiprocessing.Pool(batchNum)
    if np.sum(np.array(mlpgFlag)) > 0:
        mlpgFlag = 1
    else:
        mlpgFlag = 0
    while batch <= batchNum:
    
        #tempWarpper(command)
        pool.apply_async(tempWarpper,args=(scriptDir, fileLst, batch, mlpgFlag, mainDataDir,
                                           nndataDir, batchNum, postFilter, vocoder))
        batch = batch + 1

    
    pool.close()
    pool.join()
    os.system("rm %s.*" % (fileLst))
