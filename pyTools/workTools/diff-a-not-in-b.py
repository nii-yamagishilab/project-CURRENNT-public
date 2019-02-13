#!/usr/bin/python
import os
import sys
from ioTools import readwrite

def diff_list(list1, list2):
    return list(set(list1).difference(list2))

if __name__ == "__main__":
    len_arg = len(sys.argv) - 1
    if len_arg == 2:
        list1 = readwrite.read_txt_list(sys.argv[1])
        list2 = readwrite.read_txt_list(sys.argv[2])
        diff_list = diff_list(list1, list2)
    elif len_arg >= 3:
        if sys.argv[3] == 'r':
            list1 = readwrite.read_txt_list(sys.argv[2])
            list2 = readwrite.read_txt_list(sys.argv[1])
            diff_list = diff_list(list1, list2)
        else:
            list1 = readwrite.read_txt_list(sys.argv[1])
            list2 = readwrite.read_txt_list(sys.argv[2])
            diff_list = diff_list(list1, list2)
    else:
        pass

    if len_arg == 4 and sys.argv[4] == 'part':
        idx = 0
        for file_name in diff_list:
            print(file_name)
            idx = idx + 1
            if idx > 10:
                break
        print("%d lines" % (len(diff_list)))
    else:
        diff_list.sort()
        for file_name in diff_list:
            print(file_name)    
