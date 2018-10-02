from libc.stdio cimport *

cdef extern from "stdio.h":
     FILE *fopen(const char *fillename, const char *opentype)
     int fseek(FILE *stream, long int offset, int origin)
     
import numpy as np
cimport numpy as np


def FromFile(file):
    
    cdef int size
    
    cdef FILE* cfile
    cdef np.ndarray[np.float32_t, ndim=1] data
    cdef char* file_c = file
    cfile = fopen(file_c, 'rb') # attach the stream
    fseek(cfile, 0, SEEK_END)
    size = ftell(cfile)
    
    size = size/sizeof(np.float32_t)

    fseek(cfile, 0, SEEK_SET)

    data  = np.zeros(size).astype(np.float32)

    fread(np.PyArray_DATA(data), sizeof(np.float32_t), size, cfile)
    
    #return data,size,sizeof(np.float32_t),sizeof(np.float32)
    fclose(cfile)
    return data