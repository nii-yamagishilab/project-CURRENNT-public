#!/usr/bin/python

import os
import sys

class figConfig(object):
    def __init__(self, opt=1):
        if opt == 1:
            self.left   = 0.06
            self.right  = 0.98
            self.bottom = 0.15
            self.top    = 0.95
            self.wspace = 0.2
            self.hspace = 0.2
            self.fontSize = 9
            self.figureSize=(9.44, 2.75)
        else:
            self.left   = 0.125
            self.right  = 0.9
            self.bottom = 0.1
            self.top    = 0.9
            self.wspace = 0.2
            self.hspace = 0.2
            self.fontSize = 9
            self.figureSize=(9.44, 2.75)

    def l(self, adjustpara=0):
        return self.left + adjustpara
    def r(self, adjustpara=0):
        return self.right + adjustpara
    def b(self, adjustpara=0):
        return self.bottom + adjustpara
    def t(self, adjustpara=0):
        return self.top + adjustpara
    def w(self, adjustpara=0):
        return self.wspace + adjustpara
    def h(self, adjustpara=0):
        return self.hspace + adjustpara
    def fontS(self, adjustpara=0):
        return self.fontSize + adjustpara
    def figureS(self, add1=0,add2=0,mul1=1,mul2=1):
        return tuple([self.figureSize[0]*mul1 + add1,
                      self.figureSize[1]*mul2 + add2])

def cm2inch(*tupl):
    """
    cm 2 inch conversion
    """
    inch = 2.5423728813559325
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


if __name__ == "__main__":
    print("cm2inch")
