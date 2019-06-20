#!/usr/bin/python

from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import json


import numpy as np
from   ioTools import readwrite as py_rw


def distParaNum(featDim, mdnConfig, tieVariance, ARDynamicOrder):
    """ Calcualte the number of paramteres for a specific type of 
        distribution
        
        mdnConfig > 0: GMM with the number of mixtures equal to mdnConfig
        mdnConfig = 0: binomial distribution. The featDim should be 1
        mdnConfig < 0: multicategorical distribution. 
    """
    if mdnConfig > 0:
        # mixture model
        
        mixtureNum = mdnConfig
        if tieVariance:
            paraNum = mixtureNum * ( featDim + 2 )
        else:
            paraNum = mixtureNum * ( featDim + featDim + 1 )
        
        if ARDynamicOrder > 0:
            return paraNum + (ARDynamicOrder + 1) * featDim
        else:
            return paraNum
    elif mdnConfig < 0:
        # categorical distribution
        # return the number of softmax bins
        classNum = -1 * mdnConfig
        return classNum
    else:
        # binomial distribution (sigmoid output)
        assert featDim == 1, "feature dimension must be 1 for binomial dist"
        return featDim


def createMdnConfig(mdnConfigFile, MDNType, MDNTargetDim, ARDynamic=None, tieVariance=0):
    """Create the mdn.config for MDN CURRENNT
    """
    if ARDynamic is None:
        # default, no AR dynamic
        ARDynamic = np.ones([len(MDNType)]) * -1.0
    
    bias = 0
    MDNNNOutDim = []
    for idx, mdnConfig in enumerate(MDNType):
        temp = distParaNum(MDNTargetDim[idx][1] - MDNTargetDim[idx][0], mdnConfig,
                           tieVariance, ARDynamic[idx])
        MDNNNOutDim.append([bias, bias+temp])
        bias = temp+bias
    #print MDNNNOutDim

    # check and generating the MDN configuration
    mdnconfigdata = np.zeros([1+len(MDNType)*5], dtype = np.float32)
    mdnconfigdata[0] = len(MDNType)

    tmp = 0
    for idx, mdnConfig in enumerate(MDNType):
        mdntarDim = MDNTargetDim[idx]
        mdnoutDim = MDNNNOutDim[idx]
        tmp1 = distParaNum(mdntarDim[1]-mdntarDim[0], mdnConfig, tieVariance, ARDynamic[idx])
        tmp2 = (mdnoutDim[1]-mdnoutDim[0])

        if mdnConfig > 0:
            assert tmp1 == tmp2, "Error in MDN mixture configuraiton"
            tmp = tmp + tmp2

        elif mdnConfig < 0:
            assert mdntarDim[1]-mdntarDim[0]==1, "Softmax to 1 dimension targert"
            #tmp = tmp + 1
            tmp = tmp + tmp2
            mdnConfig = -1 # change it back to -1
        else:
            tmp = tmp + tmp2
        mdnconfigdata[(idx*5+1):((idx+1)*5+1)] = [mdnoutDim[0],mdnoutDim[1],
                                                  mdntarDim[0],mdntarDim[1],
                                                  mdnConfig]

    #print "Dimension of output of NN should be %d" % (tmp)
    py_rw.write_raw_mat(mdnconfigdata, mdnConfigFile)
    return tmp



def modifyNetworkFile(inputNetwork, inputSize, outputSize, mdnconfig, outputPath, mdnPath):

    with open(inputNetwork, 'r') as filePtr:
        networkRaw = json.load(filePtr)

    if networkRaw['layers'][1]['type'] != 'externalloader':
        # add externalloader if necessary
        networkRaw['layers'].insert(1, {'bias': 1.0, 'type': 'externalloader',
                                        'name': 'external_loader', 'size': inputSize})
        networkRaw['layers'][0]['size'] = 1
        
    networkRaw['layers'][1]['size'] = inputSize
    networkRaw['layers'][-1]['size'] = outputSize
    if 'useExternalOutput' in networkRaw['layers'][-1]:
        networkRaw['layers'][-1]['useExternalOutput'] = 1
    else:
        networkRaw['layers'][-1]['useExternalOutput'] = 1
    
    if len(mdnconfig) > 0:
        assert networkRaw['layers'][-1]['type'] == 'mdn', \
            "%s should use mdn as last layer" % (inputNetwork)
        MDNType      = []
        MDNTargetDim = []
        for config in mdnconfig:
            assert len(config) == 3, 'Wront format in mdnConfig in globalconfig'
            MDNType.append(config[2])
            MDNTargetDim.append([config[0], config[1]])
        parameterDim = createMdnConfig(mdnPath, MDNType, MDNTargetDim)
        networkRaw['layers'][-2]['size'] = parameterDim
    else:
        networkRaw['layers'][-2]['size'] = outputSize
        
    with open(outputPath, 'w') as filePtr:
        json.dump(networkRaw, filePtr, indent=4, separators=(',', ': '))
    
    
if __name__ == "__main__":
    
    # 
    mdn_output_path = sys.argv[1]
    template = sys.argv[2]
    
    if template == 'wavenet-mu-law':
        # generate a template file wavenet-mu-law mdn.config
        bit_num = int(sys.argv[3])
        createMdnConfig(mdn_output_path, [-1 * np.power(2, bit_num)], [[0, 1]])
    else:
        print("template option is unknown: %s" % (template))
    
    
