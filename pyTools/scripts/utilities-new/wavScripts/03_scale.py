#!/usr/bin/python
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import sys
from scipy.io import wavfile

if __name__ == "__main__":
    
    samp, data = wavfile.read(sys.argv[1])
    if data.dtype == np.int16:
        # int16 should be fine
        pass
    else:
        # float32 should be scaled
        data = data / (np.abs(data).max()*1.001)
    wavfile.write(sys.argv[2], samp, data)
