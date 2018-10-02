# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 11:20:07 2015

@author: wangx
"""

import os
import numpy as np
#import data_raw_handle as drh
from ioTools import readwrite as drh
import cPickle
import math
import random

def prepare_data_s_b(data_dir, data_in_name, data_out_name, batch_num, dim_in, 
                     dim_out, buffer_size, out_data_in_name, out_data_out_name,
                     p_format='<f4', val_set=0.2):
    '''prepare_data_s_b: prepare the data in Split and Batch_mode
    
    '''
    buffer_in = np.zeros([1, dim_in], dtype=np.float32)
    buffer_out= np.zeros([1, dim_out], dtype=np.float32)
    
    data_counter = 0
    for i in xrange(batch_num):
        # the batch data is assumed to start from data1
        data_tmp_in = os.path.join(data_dir, data_in_name, str(i+1)) 
        data_tmp_out= os.path.join(data_dir, data_out_name, str(i+1)) 
        assert os.path.isfile(data_tmp_in) and os.path.isfile(data_tmp_out), \
        'not found %s and %s' % (data_tmp_in, data_tmp_out)
        data_tmp_in = drh.read_raw_mat(data_tmp_in, dim_in, p_format=p_format)
        data_tmp_out= drh.read_raw_mat(data_tmp_out, dim_out, p_format=p_format)
        buffer_in = np.append(buffer_in, data_tmp_in)
        buffer_out= np.append(buffer_out, data_tmp_out)
        print 'processing %d/%d batch' % (i,batch_num)
    
    # randomlize
    data_idx = np.random.permutation(buffer_in.shape[0]-1) + 1
    buffer_in = buffer_in[data_idx]
    buffer_out= buffer_out[data_idx]
    
    # output as cPickled file
    train_size = int(math.floor((1-val_set)*buffer_in.shape[0]/100))*100
    val_size = buffer_in.shape[0] - train_size
    tmp1 = int(math.floor(train_size/buffer_size))
    tmp2 = train_size - buffer_size*tmp1
    index1 = np.arange(tmp1)*buffer_size
    index2 = index1 + buffer_size
    index1.append(index2[-1])
    index2.append(train_size)
    
    file_counter = 1
    for s_idx, e_idx in zip(index1,index2):
        buffer_data = (buffer_in[s_idx:e_idx], buffer_out[s_idx:e_idx])
        fid = open(os.path.join(data_dir,out_data_in_name,str(file_counter)),'rb')
        cPickle.dump(fid,buffer_data)
        fid.cloe()
        file_counter += 1
    
    if val_set > 0:
        buffer_data = (buffer_in[train_size:], buffer_out[train_size:])
        fid = open(os.path.join(data_dir,out_data_in_name,'val'),'rb')
        cPickle.dump(fid,buffer_data)
        fid.cloe()
    
    print 'Processing Done'


def data_normalize(dataFile, dim, mask=None, ran=False):
    
    data = drh.read_raw_mat(dataFile, dim)
    if ran==True:
        ranfile = os.path.dirname(dataFile)+os.path.sep+'random_index'
        if os.path.isfile(ranfile):
            print "Found random index %s" % (ranfile)
            randidx = np.asarray(drh.read_raw_mat(ranfile,1), dtype=np.int32)
            if randidx.shape[0]!=data.shape[0]:
                print "But it unmatches the data. New random_index will be generated"
                randidx = np.array(range(data.shape[0]))
                random.shuffle(randidx)
                drh.write_raw_mat(randidx, ranfile)
            else:
                pass
        else:
            randidx = np.array(range(data.shape[0]))
            random.shuffle(randidx)
            drh.write_raw_mat(randidx, ranfile)
        data = data[randidx,:]

    meanData = data.mean(axis=0)
    stdData  = data.std(axis=0)
    if mask is None:
        pass
    else:
        stdData[mask>0] = 0.0 # set to zero
    
    idx = stdData>0.000001   # a threshold to normalize
    data[:,idx] = (data[:,idx]-meanData[idx])/stdData[idx]
    
    drh.write_raw_mat(data,dataFile+'.norm')
    drh.write_raw_mat(np.concatenate((meanData, stdData)), dataFile+'.mv')


if __name__ == '__main__':
    mask = np.zeros([411], dtype=np.int32)
    mask[-1] = 1
    data_normalize('/home/smg/wang/DATA/BURNC_PART/datapool/data_v1', 411, mask=mask)
