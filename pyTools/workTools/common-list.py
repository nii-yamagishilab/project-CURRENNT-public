#!/usr/bin/python
import os
import sys
from ioTools import readwrite

def common_list(list1, list2):
    return list(set(list1).intersection(list2))

if __name__ == "__main__":
    len_arg = len(sys.argv) - 1
    for idx in range(len_arg):
        if idx == 0:
            file_common_list = readwrite.read_txt_list(sys.argv[idx+1])
        else:
            file_temp_list = readwrite.read_txt_list(sys.argv[idx+1])
            file_common_list = common_list(file_common_list, file_temp_list)
    for file_name in file_common_list:
        print(file_name)
