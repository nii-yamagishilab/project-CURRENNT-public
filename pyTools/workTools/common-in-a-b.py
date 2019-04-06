#!/usr/bin/python
import os
import sys
from ioTools import readwrite

def common_list(list1, list2):
    return list(set(list1).intersection(list2))

if __name__ == "__main__":
    
    list1 = readwrite.read_txt_list(sys.argv[1])
    list2 = readwrite.read_txt_list(sys.argv[2])
    common_list = common_list(list1, list2)

    if len(sys.argv) > 3 and sys.argv[3] == 'part':
        idx = 0
        for file_name in common_list:
            print(file_name)
            idx = idx + 1
            if idx > 10:
                break
        print("%d lines" % (len(common_list)))
    else:
        common_list.sort()
        for file_name in common_list:
            print(file_name)    
