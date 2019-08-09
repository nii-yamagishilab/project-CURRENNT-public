#!/usr/bin/python
""" Make json file pretty
"""
import os
import sys
import json

if __name__ == "__main__":

    network_path = sys.argv[1]

    network_path_tmp = network_path + '.tmp'
    with open(network_path, 'r') as file_ptr:
        network_data = json.load(file_ptr)

    with open(network_path_tmp, 'w') as file_ptr:
        json.dump(network_data, file_ptr, indent=4, separators=(',', ': '), sort_keys=True)
    if os.path.isfile(network_path_tmp):
        os.system("mv %s %s" % (network_path_tmp, network_path))
        print('%s updated' % (network_path))

    
