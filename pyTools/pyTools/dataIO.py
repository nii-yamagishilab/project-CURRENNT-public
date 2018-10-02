# -*- coding: utf-8 -*-
"""

Created on Thu Mar 12 09:30:46 2015
@author: wangx
"""

import cPickle
import gzip
import os
import sys
import numpy as np

def read_scp2list(listfile):
    ''' Read scp and get list'''
    assert os.path.isfile(listfile), "Can't find file %s" % (listfile)
    f = open(listfile,'r')
    temp = f.readline().split('\n')[0]
    scplist = []
    while len(temp) > 0:
        scplist.append(temp)
        temp = f.readline().split('\n')[0]
    f.close()
    return scplist

def read_raw_mat(dataset, dim, p_format='<f4', f_trans=False):
    ''' Read binary matrix as np.arrary
    data_all = read_raw_mat(dataset, p_format='<f4', f_trans=True)
    :input: 
        dataset:    binary data file
        dim:        dimension of the data
        p_format:   format of data (default: '<f4', little-endian 4-bytes float)
        f_trans:    transpose the data before output (default=False)
    :output:
        data_all:   np.array, one data vector per row
    '''
    assert len(p_format)>0, 'Unspecified p_format'
    data_all = np.fromfile(dataset, dtype=p_format)
    assert dim>0 and ((data_all.shape[0]) % dim==0), 'dim is invalid'
    row = (data_all.shape[0])/dim
    data_all = data_all.reshape(row,dim)
    if f_trans:        
        data_all = data_all.T
    return data_all

def write_raw_mat(outfile, data):
    if os.path.isfile(outfile):
        print '%s will be over-written' % (outfile)
    assert type(data) is np.ndarray, 'data is not numpy.ndarray'
    data.tofile(outfile)

def load_pickled_data(dataset, f_Gzip=True, f_Pickled=True):
    ''' Loads the dataset
    load_raw_data(dataset, f_Gzip=True, f_Pickled=True)
    :param dataset: raw data pickeld in dataset
    '''
    #############
    # LOAD DATA #
    #############
    assert os.path.isfile(dataset), "Can't find file %s" % (dataset)
    print '... loading data'

    # Load the dataset
    if f_Gzip is True:
        f = gzip.open(dataset, 'rb')
    else:
        f = open(dataset,'rb')
    
    if f_Pickled is True:
        data_all = cPickle.load(f)
    else:
        print "Please use read_raw_mat or other function to read data"
    f.close()    
    return data_all



if __name__ == "__main__":
    #print "data_all = load_data(dataset, f_Gzip=True, f_Pickled=True)\n"
    #print "data_all = read_raw_mat(dataset, dim, p_format='<f4', f_trans=False)\n"
    #write_raw_mat('outDataTmp',data)
    temp = read_scp2list('/home/wangx/Work/Project/Chinese/Theano_RBM_DBN/data/model_list.scp')