#!/usr/bin/python

class pyToolsDcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def self_print(message, opt):
    """ self_print(message, opt)
        
    """
    if opt == 'warning':
        print pyToolsDcolors.WARNING + message + pyToolsDcolors.ENDC
    elif opt == 'highlight':
        print pyToolsDcolors.OKGREEN + message + pyToolsDcolors.ENDC
    else:
        print message
