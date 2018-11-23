#!/usr/bin/python
import os
import sys
import numpy as np
from ioTools import readwrite as py_rw

def lf02f0(data):
    return (np.exp(data/1127.0) - 1) * 700.0

def f02lf0(data):
    return np.log(data / 700.0 + 1) * 1127.0

def lf02f0_file(data_file, out_file, thres=10.0):
    data = py_rw.read_raw_lf0(data_file, 1)
    data[data>thres] = lf02f0(data[data>thres])
    data[data<=thres] = 0.0
    py_rw.write_raw_mat(data, out_file)

def f02lf0_file(data_file, out_file, thres=10.0, unvoiced_value=-1.0e+10):
    data = py_rw.read_raw_lf0(data_file, 1)
    data[data>thres] = f02lf0(data[data>thres])
    data[data<=thres] = unvoiced_value
    py_rw.write_raw_mat(data, out_file)


if __name__ == "__main__":
   
    data_path = sys.argv[1]
    try:
        lf0ext = sys.argv[2]
    except IndexError:
        lf0ext = '.lf0'
    try:
        f0ext = sys.argv[3]
    except IndexError:
        f0ext = '.f0'
        
    if os.path.isdir(data_path):
        data_list = os.listdir(data_path)
        lf0_data_names = [x for x in data_list if x.endswith(lf0ext)]
        for lf0_data_name in lf0_data_names:
            inpath = os.path.join(data_path, lf0_data_name)
            outpath = os.path.join(data_path, lf0_data_name.split(lf0ext)[0]) + f0ext
            lf02f0_file(inpath, outpath)
            print("Processed %s" % (lf0_data_name))
