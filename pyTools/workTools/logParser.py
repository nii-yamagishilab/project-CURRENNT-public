#!/usr/bin/python

from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import os
import re

#if __name__ == "__main__" and __package__ is None:
#    import os, sys
#    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#############
# functions

def read_log_err(file_path, train_num, val_num):
    """ data_train, data_val = read_log_err(path_to_log_err, num_train_utt, num_val_utt)
    path_to_log_err: path to the log_err file
    num_train_utt: how many training utterances
    num_val_utt: how many validation utterances
    
    data_train: average error values per epoch on training set
    data_val: average error values per epoch on valiation set
    """
    
    data_str = []
    with open(file_path, 'r') as file_ptr:
        for line in file_ptr:
            try:
                tmp = int(line[0])
                data_str.append(line)
            except ValueError:
                pass

    row = len(data_str)
    col = len(np.fromstring(data_str[0], dtype=np.float32, sep=','))
    
    data = np.zeros([row,col])
    for idx, line in enumerate(data_str):
        data[idx, :] = np.fromstring(line, dtype=np.float32, sep=',')
    
    print(data.shape[0])
    total_num = train_num + val_num
    epoch_num = data.shape[0] / total_num
    data_train = np.zeros([epoch_num, data.shape[1]])
    data_val = np.zeros([epoch_num, data.shape[1]])
    
    for x in range(epoch_num):
        temp_data = data[x * total_num:(x+1)*total_num, :]
        train_part = temp_data[0:train_num,:]
        val_part = temp_data[train_num:(train_num+val_num),:]
        data_train[x, :] = np.mean(train_part, axis=0)
        data_val[x, :] = np.mean(val_part, axis=0)
    
    return data_train, data_val


def read_log_train(file_path):
    """ data_train, data_val, time_per_epoch = read_log_train(path_to_log_train)
    path_to_log_train: path to the log_train file
    
    data_train: error values per epoch on training set
    data_val: error values per epoch on valiation set
    time_per_epoch: training time per epoch
    """
    read_flag = False
    
    data_str = []
    with open(file_path, 'r') as file_ptr:
        for line in file_ptr:
            if read_flag and line.count('|'):
                data_str.append(line)
            if line.startswith('----'):
                read_flag = True
            
    row = len(data_str)

    data_train = np.zeros([row, 3])
    data_val   = np.zeros([row, 3])
    time_per_epoch = np.zeros(row)
    for idx, line in enumerate(data_str):
        time_per_epoch[idx] = float(line.split('|')[1])
        trn_data = line.split('|')[2].split('/')
        val_data = line.split('|')[3].split('/')
        for idx2 in np.arange(len(trn_data)):
            data_train[idx, idx2] = float(trn_data[idx2])
            data_val[idx,idx2] = float(val_data[idx2])

    return data_train, data_val, time_per_epoch
    
