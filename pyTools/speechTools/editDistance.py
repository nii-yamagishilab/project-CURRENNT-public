# Natural Language Toolkit: Distance Metrics
#
# Copyright (C) 2001-2017 NLTK Project
# Author: Edward Loper <edloper@gmail.com>
#         Steven Bird <stevenbird1@gmail.com>
#         Tom Lippincott <tom@cs.columbia.edu>
# URL: <http://nltk.org/>
# For license information, see LICENSE.TXT
#

"""
Distance Metrics.

Compute the distance between two items (usually strings).
As metrics, they must satisfy the following three requirements:

1. d(a, a) = 0
2. d(a, b) >= 0
3. d(a, c) <= d(a, b) + d(b, c)
"""


from __future__ import print_function
from __future__ import division

import numpy as np

def defaultDis(c1, c2):
    if c1 != c2:
        return 1
    else:
        return 0

def _edit_dist_init(len1, len2):
    lev = []
    for i in range(len1):
        lev.append([0] * len2)  # initialize 2D array to zero
    for i in range(len1):
        lev[i][0] = i           # column 0: 0,1,2,3,4,...
    for j in range(len2):
        lev[0][j] = j           # row 0: 0,1,2,3,4,...
    return lev


def _edit_dist_step(lev, i, j, s1, s2, disFunc):
    c1 = s1[i - 1]
    c2 = s2[j - 1]

    # skipping a character in s1
    a = lev[i - 1][j] + 1
    # skipping a character in s2
    b = lev[i][j - 1] + 1
    # substitution
    c = lev[i - 1][j - 1] + disFunc(c1, c2)

    # transposition
    d = c + 1  # never picked by default

    # pick the cheapest
    lev[i][j] = min(a, b, c, d)
    return np.argmin([a, b, c, d])

def traceBack(ope):
    len1, len2 = ope.shape
    x = len1 - 1
    y = len2 - 1
    opeStr = []
    while y > 0 or x > 0:
        if ope[x, y] == 1:
            opeStr.append('i')
            y = y - 1
        elif ope[x,y] == 2:
            opeStr.append('r')
            y = y - 1
            x = x -1
        elif ope[x,y] == 0:
            opeStr.append('d')
            x = x - 1
        else:
            return []

    if y == 0 and x == 0:
        pass

    elif x > 0:
        while x > 0:
            if ope[x, y]==0:
                x = x - 1
                opeStr.append('d')
            else:
                assert 1==0, "Fail to alignment"
                return []
    elif y > 0:
        while y > 0:
            if ope[x, y] == 1:
                y = y - 1
                opeStr.append('i')
            else:
                assert 1==0, "Fail to alignment"
                return []
    else:
        assert 1==0, "Fail to alignment"
        return []
    
    return opeStr[::-1]

def getAlignment(len1, len2, opeStr, direc='p'):
    cnt1 = 0
    cnt2 = 0
    aligna = []
    if direc == 'p':
        for val in opeStr:
            if val is 'r':
                aligna.append(cnt2)
                cnt1 = cnt1 + 1
                cnt2 = cnt2 + 1
            elif val is 'i':
                cnt2 = cnt2 + 1
            elif val is 'd':
                aligna.append(-1)
                cnt1 = cnt1 + 1
        assert cnt1 == len1, 'Fail to get alignment'
    else:
        for val in opeStr:
            if val is 'r':
                aligna.append(cnt2)
                cnt1 = cnt1 + 1
                cnt2 = cnt2 + 1
            elif val is 'i':
                aligna.append(-1)
                cnt2 = cnt2 + 1
            elif val is 'd':
                cnt1 = cnt1 + 1
        assert cnt2 == len2, 'Fail to get alignment'
    return aligna
    

def edit_distance(s1, s2, disFunc=None):
    """
    """
    if disFunc is None:
        tempFunc = defaultDis
    else:
        tempFunc = disFunc

    # set up a 2-D array
    len1 = len(s1)
    len2 = len(s2)
    lev = _edit_dist_init(len1 + 1, len2 + 1)
    ope = -1*np.ones([len1+1, len2+1])

    # iterate over the array
    for i in range(len1):
        for j in range(len2):
            ope[i+1, j+1]= _edit_dist_step(lev, i + 1, j + 1, s1, s2,
                                           tempFunc)

    opeStr = traceBack(ope)
    #print(opeStr)
    #print()
    #print()
    return getAlignment(len1, len2, opeStr, 'p'), getAlignment(len1, len2, opeStr, 'r')


def demo():
    edit_distance_examples = [
        (['b', 'a', 'c'], ['b','d','c','b'])]
    for s1, s2 in edit_distance_examples:
        print("Edit distance between '%s' and '%s':" % (s1, s2), edit_distance(s1, s2))



if __name__ == '__main__':
    demo()
