#!/usr/bin/python
from __future__ import print_function
import datetime
import sys
class pyToolsDcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[91m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def self_print(message, opt='ok'):
    """ self_print(message, opt)
        opt: warning, highlight, ok, error
    """
    if opt == 'warning':
        print(pyToolsDcolors.WARNING + str(message) + pyToolsDcolors.ENDC)
    elif opt == 'highlight':
        print(pyToolsDcolors.OKGREEN + str(message) + pyToolsDcolors.ENDC)
    elif opt == 'ok':
        print(pyToolsDcolors.OKBLUE + str(message) + pyToolsDcolors.ENDC)
    elif opt == 'error':
        print(pyToolsDcolors.FAIL + str(message) + pyToolsDcolors.ENDC)
    else:
        print(message)

def self_print_with_date(message, level='h'):
    """ self_print_with_date(message, level)
        level: h, m, l
    """
    if level == 'h':
        message = '---  ' + str(message) + ' ' + str(datetime.datetime.now())  + ' ---'
        tmp = ''.join(['-' for x in range(len(message))])
        self_print(tmp)
        self_print(message)
        self_print(tmp)
    elif level == 'm':
        self_print('---' + str(message) + ' ' + str(datetime.datetime.now().time()) + '---')
    else:
        self_print(str(message) + ' ' + str(datetime.datetime.now().time()))
    sys.stdout.flush()
    
