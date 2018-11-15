# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 23:09:17 2015

@author: wangx
"""
#import theano
from __future__ import print_function

def g_inspect_inputs(i, node, fn):
    print(i, node, "input(s) value(s):", [input[0] for input in fn.inputs], end=' ')

def g_inspect_outputs(i, node, fn):
    print("output(s) value(s):", [output[0] for output in fn.outputs])

def createNoneLayer(layer):
    w_pre = []
    for i in range(layer):
        w_pre.append(None)
    return w_pre
    
def dataBatchNum(x,y):
    """
    """
    return (x/y)+(x%y>0)
 

def getDict(rawData):
    """ 
    """
    data = {}
    for index, stackType in enumerate(rawData):
        if type(stackType) is list:
            for tmp in stackType:
                data[tmp] = index
        elif type(stackType) is str:
            data[stackType] = index
    return data
   
if __name__ == '__main__':
    print('createNoneLayer(layer)')
