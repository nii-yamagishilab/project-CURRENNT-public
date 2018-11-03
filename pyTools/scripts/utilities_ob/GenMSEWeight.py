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


if __name__ == "__main__":

    outDim  = 256                                # dimension of the output
    outFile = './mseWeight'                      # where to write the output vector?
    
    # write the vector
    data = np.ones([outDim], dtype = np.float32) # prepare the weight vector
    data[100:180] = 0.5 # I'd like the 100th-179th dimension with weight 0.5
    funcs.write_raw_mat(data, outFile)

    # test
    data = funcs.read_raw_mat(outFile, outDim)
    print data
