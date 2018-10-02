#!/usr/bin/python

###################################
# 
#
#
#
###################################

import numpy as np
from ioTools import readwrite as py_rw


"""
\begin{table}[h]
\centering
\caption{Boundary Tone}
\label{T20151219-3}
\begin{tabular}{|l|l|l|l|}
\hline
 the same & Rec    & Pre    & f-score \\ \hline
 & 0.7631 & 0.762  & \textbf{0.7626}  \\ \hline
 & 0.7536 & 0.756  & \textbf{0.7548}  \\ \hline
 & 0.6837 & 0.729  & 0.7056  \\ \hline
 & 0.7371 & 0.7575 & 0.7472  \\ \hline
 & 0.7259 & 0.7459 & 0.7358  \\ \hline
 & 0.643  & 0.7103 & 0.675   \\ \hline
 & 0.7374 & 0.7535 & 0.7454  \\ \hline
 & 0.7175 & 0.7395 & 0.7283  \\ \hline
 & 0.5858 & 0.6792 & 0.629   \\ \hline
\end{tabular}
\end{table}
"""

def data2latexTable(dataFile, dPformat='%3.3f', xlabel=[], ylabel=[], 
                    dformat='f4', delimiter='\t', xJump=[0,1],yJump=[0,1]):
    data = py_rw.read_txt_mat(dataFile, dformat, delimiter)
    data = data[:,xJump[0]::xJump[1]]
    data = data[yJump[0]::yJump[1],:]
    n,m  = data.shape
    
    if len(xlabel)>0 and len(xlabel) != m:
        print "make sure the xlabel is typed in correctly"

    if len(ylabel)>0 and len(ylabel) != n:
        print "make sure the ylabel is typed in correctly"
    
    if len(xlabel)==0:
        for x in xrange(m):
            xlabel.append('')
    if len(ylabel)==0:
        for y in xrange(n):
            ylabel.append('')

    print "\\begin{table}[h]"
    print "\\centering"
    print "\\caption{CHANGE_THE_CAPTION}"
    print "\\label{CHANGE_THE_LABEL}"
    tmpStr = ''
    tmpStr2 = '|'
    for x in xrange(len(xlabel)+1):
        tmpStr += 'c'
        tmpStr2+= 'c|'
    print "%%Choose one tabular"
    print "%%\\begin{tabular}{%s}" % (tmpStr)
    print "%%\\begin{tabular}{%s}" % (tmpStr2)
    print "\\hline"
    for y in xrange(len(ylabel)+1):
        if y==0:
            tmpStr = '&'
            for x in xrange(len(xlabel)):
                tmpStr += xlabel[x]
                tmpStr += '&'
            #tmpStr += '\\\\ \\hline'
        else:
            tmpStr = '%s &' % (ylabel[y-1])
            for x in xrange(len(xlabel)):
                tmpStr += dPformat % (data[y-1][x])
                tmpStr += '&'
        tmpStr = tmpStr.rstrip('&')
        tmpStr += '\\\\ \\hline'
        print tmpStr

    print "\\end{tabular}"
    print "\\end{table}"


def printToString(data):
    if len(data.shape) == 1:
        print '[',
        for ele in data[0:-1]:
            print "%s," % str(ele),
        
        print "%s]" % str(data[-1]),
        
if __name__ == "__main__":
    data2latexTable('/home/smg/wang/TEMP/temp_data.txt', xlabel=['pre','rec','acc'], yJump=[0,2])
