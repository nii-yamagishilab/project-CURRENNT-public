#!/usr/bin/python

from ioTools import readwrite as py_rw

fileScp  =  '/work/smg/wang/PROJ/NNWAV/nancy/DATA/lst/train_random.lst'
dataDirs = ['/home/smg/takaki/FEAT/nancy/data_sptk_32k/sp',
            '/work/smg/wang/PROJ/NNWAV/nancy/DATA2/diff_sp_log_model05_tr1',
            '/home/smg/takaki/FEAT/nancy/nndata/iplf0',
            '/home/smg/takaki/FEAT/nancy/nndata/vuv',
            '/home/smg/takaki/FEAT/nancy/nndata/lab']
fileExts = ['.sp','.sp','.lf0','.vuv','.lab']
scpNames = ['/work/smg/wang/PROJ/NNWAV/nancy/DATA/lst/sp.scp',
            '/work/smg/wang/PROJ/NNWAV/nancy/DATA2/lst/sp.scp',
            '/work/smg/wang/PROJ/NNWAV/nancy/DATA/lst/lf0.scp',
            '/work/smg/wang/PROJ/NNWAV/nancy/DATA/lst/vuv.scp',
            '/work/smg/wang/PROJ/NNWAV/nancy/DATA/lst/lab.scp']


for idx2, fileScpOut in enumerate(dataDirs):
    fileScpOut = open(scpNames[idx2], 'w')
    fileExt    = fileExts[idx2]
    fileDir    = dataDirs[idx2]
    
    with open(fileScp, 'r') as filePtr:
        for idx1, fileName in enumerate(filePtr):
            fileName = fileName.split('\n')[0]
            print "Process %s (%d)" % (fileName, idx1)
            fileScpOut.write("%s/%s%s\n" % (fileDir, fileName, fileExt))

    fileScpOut.close()
