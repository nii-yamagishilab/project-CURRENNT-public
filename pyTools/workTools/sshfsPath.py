#!/usr/bin/python

import os
import sys
import re

def local_path(network_path):
    home_str = '/home/smg/wang/WORK/WORK'
    local_str = '/Users/wang-local/WORK/REMO/WORK/WORK'

    return re.sub(home_str, local_str, network_path)
