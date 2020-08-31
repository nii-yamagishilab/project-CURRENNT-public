from ioTools import readwrite as py_rw
from speechTools import discreteF0
import numpy as np
import multiprocessing
import os

# parameter of F009
# check: /work/smg/wang/PROJ/F0MODEL/DATA/F009/scripts/F0Class/01_converF0.py
F0Max    = 529.0 # mean + 3*std (F0Max from 00_stats.py has double-valued F0, it is not accurate)
F0Min    = 66.0  # F0Min from 00_stats.py
F0Zero   = 10.0  # F0<10.0 will be treated as unvoiced F0
F0Inter  = 256   # quantization level
F0Conti  = 0     # 0: the F0 was not interpolated



def generate(inFile1, inFile2, outfile):
    data1 = np.asarray(py_rw.read_raw_mat(inFile1, 1), dtype=np.int32)
    data2 = py_rw.read_raw_mat(inFile2, 1)
    temp,_ = discreteF0.f0Conversion(data2.copy(), F0Max, F0Min, F0Inter, 'c2d', F0Conti)
    data3 = np.zeros(data1.shape)
    data3[data2[data1]>0] = 1
    py_rw.write_raw_mat(data3, outfile)


def tempWarpper(fileName):
    fileName = fileName.rstrip('\n')
    inFile1  = './labIndex16k/' + fileName + '.labidx'
    inFile2  = '/work/smg/wang/PROJ/F0MODEL/MODEL_F009/F0CLASS/FB1S/002/output_testset_trained_network_mdn1.000000' + os.path.sep + fileName + '.lf0'
    #inFile1  = '/work/smg/wang/PROJ/WE/DNNAM/MODEL/F009/ARRMDN/001/output_trainset_epoch001_mdn0.001000' + os.path.sep + fileName + '.mgc_delta'
    #inFile2  = '/work/smg/wang/PROJ/F0MODEL/MODEL_F009/F0CLASS/FB1S/002/output_valset_trained_network_mdn1.000000' + os.path.sep + fileName + '.lf0'
    outFile = './genControl'  + os.path.sep + fileName + '.bin'
    print inFile2,outFile
    generate(inFile1, inFile2, outFile)


if __name__ == "__main__":
    pool = multiprocessing.Pool(10)
    with open('./test.lst') as filePtr:
        for idx,filename in enumerate(filePtr):
            #tempWarpper(filename)
            pool.apply_async(tempWarpper, args=(filename, ))
    pool.close()
    pool.join()
