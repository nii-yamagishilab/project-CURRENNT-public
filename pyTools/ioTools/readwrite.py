#!/usr/bin/python

from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import os

#if __name__ == "__main__" and __package__ is None:
#    import os, sys
#    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyTools import strtools as sd_strtool
#from pyTools.strtools import strtool as sd_strtool

import pickle

##########################################################
# functions
def read_raw_mat(filename,col,format='f4',end='l'):
    """read_raw_mat(filename,col,format='float',end='l')
       Read the binary data from filename
       Return data, which is a (N, col) array
       
       filename: the name of the file, take care about '\\'
       col:      the number of column of the data
       format:   please use the Python protocal to write format
                 default: 'f4', float32
                 see for more format:
       end:      little endian 'l' or big endian 'b'?
                 default: 'l'
       
       dependency: numpy
       Note: to read the raw binary data in python, the question
             is how to interprete the binary data. We can use
             struct.unpack('f',read_data) to interprete the data
             as float, however, it is slow.
    """
    f = open(filename,'rb')
    if end=='l':
        format = '<'+format
    elif end=='b':
        format = '>'+format
    else:
        format = '='+format
    datatype = np.dtype((format,(col,)))
    data = np.fromfile(f,dtype=datatype)
    f.close()
    if data.ndim == 2 and data.shape[1] == 1:
        return data[:,0]
    else:
        return data

def read_mat_frame(filename,frame,col,nm=1,bias=0,format='f4',end='l'):
    """read_raw_mat(filename,col,format='float',end='l')
       Read the binary data from filename
       Return data, which is a (N, col) array
       
       filename: the name of the file, take care about '\\'
       frame:    the n-th frame to start
       col:      the number of column of the data
       nm:       number of frame to read
                 default: 1
       bias:     if there is any bias needed
                 default: 0
       format:   please use the Python protocal to write format
                 default: 'f4', float32
                 see for more format:
       end:      little endian 'l' or big endian 'b'?
                 default: 'l'
       
       dependency: numpy
       Note: f.seek(DataSize*col(frame-1)+bias,0)
    """
    f = open(filename,'rb')
    if end=='l':
        format = '<'+format
    elif end=='b':
        format = '>'+format
    else:
        format = '='+format
    datatype = np.dtype((format,(col,)))
    f.seek(datatype.itemsize*(frame-1)+bias,0)
    data = np.fromfile(f,dtype=datatype,count=nm)
    f.close()
    return data
    
def read_htk(filename, format='f4', end='l'):
    """read_htk(filename, format='f4', end='l')
        Read HTK File and return the data as numpy.array 
        filename:   input file name
        format:     the format of the data
                    default: 'f4' float32
        end:        little endian 'l' or big endian 'b'?
                    default: 'l'
    """
    if end=='l':
        format = '<'+format
        formatInt4 = '<i4'
        formatInt2 = '<i2'
    elif end=='b':
        format = '>'+format
        formatInt4 = '>i4'
        formatInt2 = '>i2'
    else:
        format = '='+format
        formatInt4 = '=i4'
        formatInt2 = '=i2'

    head_type = np.dtype([('nSample',formatInt4), ('Period',formatInt4),
                          ('SampleSize',formatInt2), ('kind',formatInt2)])
    f = open(filename,'rb')
    head_info = np.fromfile(f,dtype=head_type,count=1)
    
    """if end=='l':
        format = '<'+format
    elif end=='b':
        format = '>'+format
    else:
        format = '='+format
    """    
    if 'f' in format:
        sample_ize = head_info['SampleSize'][0]/4
    else:
        print("Error in read_htk: input should be float32")
        return False
        
    datatype = np.dtype((format,(sample_ize,)))
    data = np.fromfile(f,dtype=datatype)
    f.close()
    return data

    
def read_txt_mat(filename, format='f4', delimiter='\t'):
    """read_txt_mat(filename, format='f4', delimiter='\t')
        Read data as np.array from filename
        
        filename: the name of the file (data in text format)
        format:   please use the Python protocal to write format
                  default: 'f4', float32
        delimiter: the delimiter between numbers
                    default: '\\t'
        dependency: numpy
    """
    return np.loadtxt(filename, dtype=format, delimiter=delimiter)

def read_txt_list(filename, f_chop=True):
    """
    """
    f = open(filename,'r')
    data = []
    for line in f:
        if f_chop:
            line = sd_strtool.str_chop(line)
        data.append(line)
    f.close()
    return data

def write_raw_mat(data,filename,format='f4',end='l'):
    """write_raw_mat(data,filename,format='',end='l')
       Write the binary data from filename. 
       Return True
       
       data:     np.array
       filename: the name of the file, take care about '\\'
       format:   please use the Python protocal to write format
                 default: 'f4', float32
       end:      little endian 'l' or big endian 'b'?
                 default: '', only when format is specified, end
                 is effective
       
       dependency: numpy
       Note: we can also write two for loop to write the data using
             f.write(data[a][b]), but it is too slow
    """
    if not isinstance(data, np.ndarray):
        print("Error write_raw_mat: input shoul be np.array")
        return False
    f = open(filename,'wb')
    if len(format)>0:
        if end=='l':
            format = '<'+format
        elif end=='b':
            format = '>'+format
        else:
            format = '='+format
        datatype = np.dtype((format,1))
        temp_data = data.astype(datatype)
    else:
        temp_data = data
    temp_data.tofile(f,'')
    f.close()
    return True





def write_htk(data,targetfile,sampPeriod=50000,sampKind=9,format='f4',end='l'):
    """
    write_htk(data,targetfile,
    sampPeriod=50000,sampKind=9,format='f4',end='l')
    """
    if data.ndim==1:
        nSamples, vDim = data.shape[0], 1
    else:
        nSamples, vDim = data.shape
    if format=='f4':
        sampSize = vDim * 4;
    else:
        sampSize = vDim * 8;
    
    f = open(targetfile,'wb')

    if len(format)>0:
        if end=='l':
            format1 = '<i4'
            format2 = '<i2'
        elif end=='b':
            format1 = '>i4'
            format2 = '>i2'
        else:
            format1 = '=i4'
            format2 = '=i2'
    
    temp_data = np.array([nSamples, sampPeriod], 
                         dtype=np.dtype((format1,1)))
    temp_data.tofile(f, '')
    
    temp_data = np.array([sampSize, sampKind], dtype=np.dtype((format2,1)))
    temp_data.tofile(f, '')
    
    
    if len(format)>0:
        if end=='l':
            format = '<'+format
        elif end=='b':
            format = '>'+format
        else:
            format = '='+format
        datatype = np.dtype((format,1))
        temp_data = data.astype(datatype)
    else:
        temp_data = data
    temp_data.tofile(f, '')
    f.close()
    return True


def append_raw_mat(data,filename,format='f4',end='l'):
    """append_raw_mat(data,filename,format='',end='l')
       append the binary data from filename. 
       Return True
       
       data:     np.array
       filename: the name of the file, take care about '\\'
       format:   please use the Python protocal to write format
                 default: 'f4', float32
       end:      little endian 'l' or big endian 'b'?
                 default: '', only when format is specified, end
                 is effective
       
       dependency: numpy
       Note: we can also write two for loop to write the data using
             f.write(data[a][b]), but it is too slow
    """
    if not isinstance(data, np.ndarray):
        print("Error write_raw_mat: input shoul be np.array")
        return False
    f = open(filename,'ab')
    if len(format)>0:
        if end=='l':
            format = '<'+format
        elif end=='b':
            format = '>'+format
        else:
            format = '='+format
        datatype = np.dtype((format,1))
        temp_data = data.astype(datatype)
    else:
        temp_data = data
    temp_data.tofile(f,'')
    f.close()
    return True

def write_mat2csv(data, filename):
    """write_mat2csv(data, filename):
        Write the data array as csv file
        data: np.array, which can be 1 or 2 dimensions
        filename: the target file
        
        dependency: numpy
    """
    return sub_write_mat(data, filename, ',')

def write_mat2txt(data, filename):
    """write_mat2txt(data, filename):
        Write the data array as text format
        data: np.array, which can be 1 or 2 dimensions
        filename: the target file
        
        dependency: numpy
    """
    return sub_write_mat(data, filename, '\t')

def write_vec2txt(data, filename):
    """write_vec2txt(data, filename):
        Write the data array as text format(one data one row)
        data: np.array, which can be 1 dimensions
        filename: the target file
        
        dependency: numpy
    """
    if isinstance(data,np.ndarray) and data.ndim==1:
        return sub_write_mat(data[:,np.newaxis], filename, '\x0a')
    else:
        print("Error write_vec2txt: input shoul be 1 dim")
        return False

def write_log(name, dirOut):
    logName  = dirOut + os.path.sep + '__log'
    command  = 'cat %s > %s/__log' % (name, dirOut)
    os.system(command)
    

def read_raw_lf0(filename, col, format='f4',end='l', opt=1):
    """read_raw_lf0(filename, col, format='f4',end='l', opt=1):
    filename, col, format, end are the same as read_raw_mat
    opt = 1: read the linear F0 and convert -1.0+e10 to 0
    """
    data = read_raw_mat(filename, col, format=format, end=end)
    if opt == 1:
        data[data<0] = 0
    return data


def write_txt_list(list_data, file_path):
    """ write_txt_list(list_data, file_path)
        write list_data into file_path as a txt file
    """
    assert type(list_data) is list, "Input is not list"
    filePtr = open(file_path, 'w')           
    for fileName in list_data:               
        filePtr.write('%s\n' % (str(fileName)))   
    filePtr.close()

def list_file_name_in_dir(fileDir):
    """ list_file_name_in_dir(fileDir):
        return the list of file names without extension
    """                                                   
    return [os.path.splitext(x)[0] for x in os.listdir(fileDir) if not x.startswith('.')] 
    
    
def qdump(variable, file_name):
    with open(file_name, 'w') as file_ptr:
        pickle.dump(variable, file_ptr)

def qload(file_name):
    with open(file_name, 'r') as file_ptr:
        temp = pickle.load(file_ptr)
    return temp


        
##########################################################
# sub function 
def sub_write_mat(data, filename, separator):
    """write_mat2csv(data, filename):
        Write the data array as csv file
        data: np.array, which can be 1 or 2 dimensions
        filename: the target file
        
        dependency: numpy
    """
    if not isinstance(data, np.ndarray):
        print("Error sub_write_mat: input shoul be np.array")
        return False
    f = open(filename,'w')
    if data.ndim == 1:
        temp_data = np.expand_dims(data,axis=0)
    elif data.ndim == 2:
        temp_data = data
    else:
        print("Error write_mat2csv: input must be 1 or 2 dimension")
        return False
        
    row, col = temp_data.shape
    for x in range(row):
        temp = ''
        for y in range(col-1):
            temp += str(temp_data[x][y])+separator
        temp += str(temp_data[x][col-1])
        f.writelines(temp+'\x0a')
    f.close()
    return True

def Bytes(file_path, divide_factor = 1):
    """ return number_of_data_bytes / divide_factor
    """
    data = read_raw_mat(file_path, 1, format='c')
    return data.shape[0]/divide_factor


##########################################################
# main function
def debug_interface():
    """Debug interface
    """
    pass

if __name__ == "__main__":
    # for debugging
    #data = np.arange(10)
    #read_htk(r'f:\temp\temp\temp.cmp')
    #print(read_txt_list('/home/smg/wang/PROJ/DL/DNNAM/data/lists/mono.list'))
    pass
