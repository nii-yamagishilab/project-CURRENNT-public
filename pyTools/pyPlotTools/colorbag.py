#!/usr/bin/python

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import matplotlib

def color_list(N, colorMap = 'jet'):
    """ color_list =  colorlist(N, colorMap = 'jet'):
    N: number of colors 
    colorMap: which colormap to use?

    color_list: [N, RGV tuple]
    """
    colorBuf = np.zeros([N, 4])
    tmp = np.linspace(0, 1, N)#np.sort(np.random.rand(N))
    norm = matplotlib.colors.Normalize(vmin=tmp.min(),
                                       vmax=tmp.max())
    cmap = matplotlib.cm.get_cmap(colorMap)
    color_list = cmap(norm(tmp))
    return color_list
    

def colormap_self(N, opt = 1, order = []):
    """ color_bag = colormap_self(N, opt, order)
    input
    N:      number of colors
    opt:    which color map ?
    oroder: order of each color of the output N colors

    output
    color_bag: [N, RGB tuple]
    """
    if opt == 1:
        color_bag = np.array([[78, 78, 78],
                              [132,145,252],
                              [253,134,141],
                              [110,143,82],
                              [229,173,69],
                              [139,139,139]], dtype=np.float32)/255.0
        assert N<=color_bag.shape[0], "Not enough colors from opt==1"
    elif opt==2:
        color_bag = np.array([[178, 180, 253],
                              [253, 178, 179]])/255.0
    
    elif opt==3:
        color_bag = np.array([[1.0, 1.0, 1.0],
                              [0.95, 0.95, 0.95],
                              [0.90, 0.90, 0.90],
                              [0.85, 0.85, 0.85],
                              [0.80, 0.80, 0.80]])
    
    elif opt==4:
        color_bag = np.array([[174,174,174],
                              [13,13,13],
                              [11,36,251],
                              [252,13,27],
                              [55,183,164],
                              [189,27,189],
                              [26,120,148],
                              [110,143,82]], dtype = np.float32)/255.0;
        assert N<=color_bag.shape[0], "Not enough colors from opt==4"
    elif opt==5:
        color_bag = np.array([[132,145,252],
                              [253,134,141],
                              [110,143,82], 
                              [229,173,69],
                              [139,139,139],
                              [200,200,200]], dtype = np.float32)/255.0;
        assert N<=color_bag.shape[0], "Not enough colors from opt==5"
    elif opt==6:
        color_bag = np.array([[243,243,243],[202,202,202],[160,160,160]], 
                             dtype=np.float32)/255.0
        assert N<=color_bag.shape[0], "Not enough colors from opt==5"

    if N > 0:
        color_bag = color_bag[0:N, :]
        
    if len(order) > 0:
        color_bagOut = np.zeros([len(order), color_bag.shape[1]])
        color_bagOut[:, :] = color_bag[order,:].copy()
        #assert len(order) == color_bag.shape[0], "N != len(order)"
        #color_temp = color_bag[order, :].copy()
        #color_bag  = color_temp
        color_bag = color_bagOut
    return color_bag
        

if __name__ == "__main__":
    print "colorbag"
    fig = plt.figure(figsize=(10, 10))
    num = 6
    for x in np.arange(num):
        plt.subplot(num, 1, x+1)
        colors = colormap_self(-1, opt = x+1)
        if colors is None:
            continue
        rows = colors.shape[0]
        data = np.random.randn(100, rows)
        bp = plt.boxplot(data, positions=np.arange(0, data.shape[1]), patch_artist=True)
        plt.xticks(np.arange(data.shape[1]), [str(x) for x in np.arange(data.shape[1])])
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        for patch, color in zip(bp['medians'], colors):
            patch.set(color=color)
            
    plt.savefig('color.pdf')
    
    
