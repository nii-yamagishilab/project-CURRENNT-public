#!/usr/bin/python
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from ioTools import readwrite as py_rw

############
# This script create the weight mask 
# 1. Configure in  __main__
# 2. python this script
############

def weightNumber(i, o, layertype):
    if layertype==1:
        return o*(1+i)                      # feedforward, skippara layer
    elif layertype==2:         
        return o*(4*(i+1)+2*o+3)        # blstm
    elif layertype==3:
        return o*(4*(i+1)+4*o+3)        # lstm
    elif layertype==4:
        return 0                        # skipadd layer
    else:
        print("Not implemented")
        return 0

def getNetStruct(netStruc, subnet, layeType):
    layerNM= []
    idx = 0
    num = 0
    for dat in subnet:
        num = num + weightNumber(netStruc[idx], subnet[idx], layeType[idx+1])
        layerNM.append(weightNumber(netStruc[idx], subnet[idx], layeType[idx+1]))
        idx = idx + 1;
        
    print("Weight Numer: %s\nTotal:%d" % (str(layerNM), num))
    return layerNM, num

def genWeightMast(TotalWNum, netconfig, layerNM, netTopoNoInput, netTopo,
                  layerType, flag_verbose=0):
    weight = np.ones(TotalWNum)
    offset = 0
    for idx, layer in enumerate(netconfig):
    
        if len(layer) == 0:
            elements = np.ones(layerNM[idx], dtype=np.float32)
        else:
            if layerType[idx+1]==1: 
                # feed-forward or gate layer
                elements = np.zeros(netTopoNoInput[idx]*netTopo[idx], dtype=np.float32)
                # with row to the next layer
                # with col to the previous layer
                elements = elements.reshape([netTopoNoInput[idx], netTopo[idx]]);

                for pat in layer:
                    assert len(pat)==4, "netconfig for feed-forward has 4 elements"
                    cols, cole = pat[0], pat[1]
                    rows, rowe = pat[2], pat[3]
                    if rows<0:
                        rows = 0
                    if cols<0:
                        cols = 0
                    if rowe<0:
                        rowe = netTopoNoInput[idx]
                    if cole<0:
                        cole = netTopo[idx]
                    elements[rows:rowe, cols:cole] = 1.0
            elif layerType[idx+1]==2 or layerType[idx+1]==3:
                # blstm layer
                # currently, we contrain the specification to this case:
                #    all transformation from preceding layer to current layer
                #    share same topo structure (P-L matrix)
                #    all transformation from previous step to currennt step
                #    share same topo structure, and be identical to L of P-L matrix (L-L matrix)
                # 
                # specify the transformation matrix from pre-layer to this layer

                # 4 transformation P-L matrices 
                elements1 = np.zeros(netTopoNoInput[idx]*netTopo[idx], dtype=np.float32) 
                elements1 = elements1.reshape([netTopoNoInput[idx], netTopo[idx]]);
                for pat in layer:
                    assert len(pat)==4, "netconfig for feed-forward has 4 elements"
                    cols, cole = pat[0], pat[1]
                    rows, rowe = pat[2], pat[3]
                    if rows<0:
                        rows = 0
                    if cols<0:
                        cols = 0
                    if rowe<0:
                        rowe = netTopoNoInput[idx]
                    if cole<0:
                        cole = netTopo[idx]
                    elements1[rows:rowe, cols:cole] = 1.0
                elements1 = elements1.flatten()
                elements1 = np.concatenate((elements1, elements1, elements1, elements1))
                
                # bias
                bias = np.ones([4*netTopoNoInput[idx]])
                # 4 transformation L-L matrices
                elements2 = np.zeros(netTopoNoInput[idx]*netTopoNoInput[idx], dtype=np.float32)
                elements2 = elements2.reshape([netTopoNoInput[idx], netTopoNoInput[idx]]);
                for pat in layer:
                    assert len(pat)==4, "netconfig for feed-forward has 4 elements"
                    cols, cole = pat[2], pat[3] # make it symmtrical topo
                    rows, rowe = pat[2], pat[3]
                    if rows<0:
                        rows = 0
                    if cols<0:
                        cols = 0
                    if rowe<0:
                        rowe = netTopoNoInput[idx]
                    if cole<0:
                        cole = netTopo[idx]
                    elements2[rows:rowe, cols:cole] = 1.0
                elements2 = elements2.flatten()
                # 2/4 transformation matrices
                if layerType[idx+1]==3:
                    elements2 = np.concatenate((elements2, elements2, elements2, elements2))
                elif layerType[idx+1]==2:
                    elements2 = np.concatenate((elements2, elements2))
                    
                elements = np.concatenate((elements1, bias, elements2))
                
        # layerNM = elements + bias number
        if flag_verbose:
            print("Transform matrix of layer %d" % (idx+1))
            if layerType[idx+1]==2 or layerType[idx+1]==3:
                print("P-L")
                if len(layer) is not 0:
                    print(elements1[0:len(elements1)/4].reshape([netTopoNoInput[idx],netTopo[idx]]))
                else:
                    print("All ones (full connection)")
                print("L-L")
                if len(layer) is not 0:
                    print(elements2[0:netTopoNoInput[idx]*netTopoNoInput[idx]].reshape([netTopoNoInput[idx],netTopoNoInput[idx]]))
                else:
                    print("All ones (full connection)")
            else:
                if len(layer) is not 0:
                    print(elements)
                else:
                    print("All ones (full connection)")

        # only specify the necessary part
        weight[offset:offset+len(elements.flatten())] = elements.flatten()
        offset = offset + layerNM[idx]
    return weight

def dupVec(data, time):
    out = []
    i = 0
    while i<time:
        out.append(data)
        i = i+1
    return out


if __name__ == "__main__":
    
    # layer size, from input layer to output layer (postoutput layer is not needed)
    netTopo   = [382, 10, 8, 259]

    # layer type
    #    0: input
    #    1: feedforward (also for highway gate layer)
    #    2: blstm
    #    3: lstm
    layerType = [0, 1, 2, 1]
    
    # link between layers (start from the link between input & 1st layer, then 1st & 2nd)
    #    for one link between two layers, use this format [row_start, row_end, col_start, col_end] 
    #    note, starts from zero, python Numpy style [start, end)
    #    For example, 
    netconfig= [[],                            # full link between input & 1st layer
                [[0,  5, 0, 4],                # link 1-5 of 1st with 1-4 of 2nd layer
                 [5, 10, 4, 8]],               # link 6-10 of 1st with 5-8 of 2nd layer
                []]                            # full link between 2nd and 3rd layers
    
    
    ### Processing
    netTopo = np.array(netTopo)
    netTopoNoInput = netTopo[1:]
    layerNM, TotalWNum = getNetStruct(netTopo, netTopoNoInput, layerType)
    weight = genWeightMast(TotalWNum, netconfig, layerNM, netTopoNoInput, 
                           netTopo, layerType, 1)
    py_rw.write_raw_mat(weight, './weightMask')
