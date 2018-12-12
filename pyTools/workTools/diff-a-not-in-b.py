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
    diff_list.sort()
    for file_name in diff_list:
        print(file_name)
