#!/usr/bin/python
"""
This script create the weight mask for CURRENNT
Please add PYTHONPATH to pyTools
Please configure in the __main__ section

Intrustruction: 
    # layer size from input to output (the post-sse layer after the output layer is not necessary)
    # 
    # if the network.jsn is 
    #    input:       382
    #    feedforward: 10
    #    blstm:       8
    #    output:      259
    #    sse:         259
    # the netTopo should be
    netTopo = [382, 512, 256, 259]
    
    # the layerType should be
    # [0]: input layer
    # [1]: feedforward layer
    # [2]: blstm layer
    layerType = [0, 1, 1, 1]
    
    # Now, specify the connection between the 1st and 2nd, 2nd and 3rd, 3rd and 4th layer
    # [input_start_dimension, input_end_dimension, output_start_dimension, output_end_dimension]
    #
    # I want to connect the 1-256th components of 2nd layer to the 1-128th elements of 3rd layer
    #     it's [0, 256, 0, 128]
    # Then 257-512th components of 2nd layer to the 129-256th of 3rd layer
    #     [256, 512, 128, 256] 
    # NOTE: python style of index [0, 5] => [0, 1, 2, 3, 4] 
    netconfig= [[],  # void for the 1st-2nd layer
                [[0, 256, 0, 128], [256, 512, 128, 256]], # 2nd-3rd layer
                []]  # void for the 3rd-4th layer

"""
import sys, os
import numpy as np
from ioTools import readwrite as py_rw

def weightNumber(i, o, layertype):
    if layertype==1:
        return o*(1+i) # feedforward, skippara layer
    elif layertype==2: 
        
        return o*(4*(i+1)+2*o+3)        # blstm
    elif layertype==3:
        return o*(4*(i+1)+4*o+3)        # lstm
    else:
        print "Not implemented"
        return 0

def getNetStruct(netStruc, subnet, layeType):
    layerNM= []
    idx = 0
    num = 0
    for dat in subnet:
        num = num + weightNumber(netStruc[idx], subnet[idx], layeType[idx+1])
        layerNM.append(weightNumber(netStruc[idx], subnet[idx], layeType[idx+1]))
        idx = idx + 1;
        
    print "Weight Numer: %s\nTotal:%d" % (str(layerNM), num)
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
            print "Transform matrix of layer %d" % (idx+1)
            if (layerType[idx+1]==2 or layerType[idx+1]==3) and (len(layer)>0):
                print "P-L: input x_t to current hidden h_t (show one matrix)"
                print elements1[0:len(elements1)/4].reshape([netTopoNoInput[idx],netTopo[idx]])
                print "L-L: previous hidden h_t-1 to current hidden h_t (show one matrix)"
                print elements2[0:netTopoNoInput[idx]*netTopoNoInput[idx]].reshape([netTopoNoInput[idx],netTopoNoInput[idx]])
            else:
                print elements

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

    # layer size from input to output (the post-sse layer after the output layer is not necessary)
    # 
    # if the network.jsn is 
    #    input:       382
    #    feedforward: 10
    #    blstm:       8
    #    output:      259
    #    sse:         259
    # the netTopo should be
    netTopo = [382, 512, 256, 259]
    
    # the layerType should be
    # [0]: input layer
    # [1]: feedforward layer
    # [2]: blstm layer
    layerType = [0, 1, 1, 1]
    
    # Now, specify the connection between the 1st and 2nd, 2nd and 3rd, 3rd and 4th layer
    # [input_start_dimension, input_end_dimension, output_start_dimension, output_end_dimension]
    #
    # I want to connect the 1-256th components of 2nd layer to the 1-128th elements of 3rd layer
    #     it's [0, 256, 0, 128]
    # Then 257-512th components of 2nd layer to the 129-256th of 3rd layer
    #     [256, 512, 128, 256] 
    # NOTE: python style of index [0, 5] => [0, 1, 2, 3, 4] 
    netconfig= [[],  # void for the 1st-2nd layer
                [[0, 256, 0, 128], [256, 512, 128, 256]], # 2nd-3rd layer
                []]  # void for the 3rd-4th layer

    # 
    netTopo = np.array(netTopo)
    netTopoNoInput = netTopo[1:]
    layerNM, TotalWNum = getNetStruct(netTopo, netTopoNoInput, layerType)
    weight = genWeightMast(TotalWNum, netconfig, layerNM, netTopoNoInput, 
                           netTopo, layerType, 1)
    py_rw.write_raw_mat(weight, './weightMask')
    
    #os.system("cat ./" + __file__ + " > ./log")
