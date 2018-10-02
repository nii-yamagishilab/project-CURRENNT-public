#!/usr/bin/python

import os, sys
import numpy as np

# edit distance


def editD(a,b, change_f = lambda x,y: x!=y, del_c =1, add_c=1, fgetJump=0):
    """Calculates the EditD distance between a and b
       a, b can be list of string or int or other data structure, however, please
       specify change_f so that the difference between elements of a, b can be calculated
       add: 1
       delete: 2
       change: 3
    """
    
    n, m = len(a), len(b)
    #if n > m:
    #    # Make sure n <= m, to use O(min(n,m)) space
    #    a,b = b,a
    #    n,m = m,n
        
    current = range(n+1)
    jumpMap = np.zeros([m+1,n+1])

    for i in range(1,m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1,n+1):
            add, delete = previous[j]+add_c, current[j-1]+del_c
            change = previous[j-1]
            # we can use more complex function here
            change = change + change_f(a[j-1],b[i-1])
            #if a[j-1] != b[i-1]
            #    change = change + 1
            current[j] = min(add, delete, change)
            if fgetJump:
                if add == current[j]:
                    jumpMap[i, j] = 1
                elif delete == current[j]:
                    jumpMap[i, j] = 2
                else:
                    jumpMap[i, j] = 3
    
    if fgetJump==1:
        # trace back
        path = []
        i,j = m,n
        while j!=0 or i!=0:
            if j!=0 and i!=0:
                if jumpMap[i,j]==1:
                    # from add
                    i -= 1
                    path.append(1)
                elif jumpMap[i,j]==2:
                    j -= 1
                    path.append(2)
                else:
                    i -= 1
                    j -= 1
                    path.append(3)
            elif j!=0 and i==0:
                j -= 1
                path.append(2) # only delete
            elif i!=0 and j==0:
                i -= 1
                path.append(1)
            else:
                #end it
                pass
        path.reverse()
        return current[n], path
            
    else:
        return current[n]


def getAlignment(alignPath):
    """ for editD(a,b, fgetJump=1) return the position of elements in a in b
        The returned list is the same length as a
        1: the word in b should be deleted (it is added to the alignment)
        2: the word in a should be deleted (delete)
        3: change or match 
    """
    temp = []
    pos  = 0
    for indx, tmpJump in enumerate(alignPath):
        if tmpJump == 1:
            # add, the element of b is not in a
            # but position needs to be shifted to skip over this element in b
            pos += 1
        elif tmpJump == 2:
            # delete, the element of a is not in b
            temp.append(-1)
        elif tmpJump == 3:
            temp.append(pos)
            pos += 1
        else:
            print "*** unknow jump %s" % (str(alignPath))
            return []
    return temp


def printAlignment(alignPath, seq_a, seq_b):
    """ Print the alignment of seq_a and seq_b
        alignPath is the output from editD (not getAlignment)
    """
    str1 = 'Alignment between source and target:\n'
    str2 = ''
    ctr1 = 0
    ctr2 = 0
    maxLength = 0
    for word in seq_a:
        if len(word)>maxLength:
            maxLength = len(word)
    for word in seq_b:
        if len(word)>maxLength:
            maxLength = len(word)
    control = '%%s%%%ds %%%ds\n' % (maxLength, maxLength)
           
    for indx, align in enumerate(alignPath):
        if align  == 3:
            # print both seq_a and seq_b
            #maxLength = max([len(seq_a[ctr1]), len(seq_b[ctr2])])+1
            str1 = control % (str1, seq_a[ctr1], seq_b[ctr2])
            #str2 = control % (str2, seq_b[ctr2])
            ctr1 += 1
            ctr2 += 1
        elif align == 1:
            # the word in seq_a is additional
            #control = '%%s%%%ds (%%%ds)' % (len(seq_b[ctr2])+1, len(seq_b[ctr2])+1)
            str1 = control % (str1, '_', seq_b[ctr2])
            #str2 = control % (str2, seq_b[ctr2])
            ctr2 += 1
        elif align == 2:
            #control = '%%s%%%ds (%%%ds)' % (len(seq_a[ctr1])+1, len(seq_a[ctr1])+1)
            str1 = control % (str1, seq_a[ctr1], '_')
            #str2 = control % (str2, ' ')
            ctr1 += 1
        else:
            print "Unknown align type "
    print str1
    #print str2

if __name__ == "__main__":
    # edit between number lists
    dis, jump = editD([1,2,3,5,6,8], [2,3,5,6,8], fgetJump=1)
    # edit between strings
    dis, jump = editD('eiHght', 'Height', fgetJump=1)
    # even edit between list of strings, where two words are further calculated by editD(word1, word2)
    a = ['a', 'b', 'c', 'sd', 'msdq']
    b = ['tmp', 'a', 'c', 'sd', 'msd']
    dis, jump = editD(a,b,change_f = lambda x,y: 2.0*editD(x,y)/(len(x)+len(y)), fgetJump=1)
    print a
    print jump
    print getAlignment(jump)
    printAlignment(jump, a, b)
    print b
    a= ['endsil', 'Wanted', 'sil', 'Chief', 'Justice', 'of', 'the', 'Massachusetts', 'Supreme', 'Court', 'sil', 'In', 'April', 'the', "S.J.C.'s", 'current', 'leader', 'sil', 'Edward', 'Hennessy', 'sil', 'sil', 'reaches', 'the', 'mandatory', 'retirement', 'age', 'of', 'seventy', 'sil', 'and', 'a', 'successor', 'is', 'expected', 'to', 'be', 'named', 'In', 'March', 'sil', 'sil', 'It', 'may', 'be', 'the', 'most', 'important', 'appointment', 'sil', 'Governor', 'Michael', 'Dukakis', 'makes', 'sil', 'during', 'the', 'remainder', 'of', 'his', 'administration', 'sil', 'and', 'one', 'of', 'the', 'toughest', 'sil', 'sil', 'as', "WBUR's", 'Margo', 'Melnicove', 'reports', 'sil', 'Hennessy', 'will', 'be', 'a', 'sil', 'hard', 'sil', 'act', 'to', 'sil', 'follow', 'endsil']
    b = ['Wanted', ':', 'Chief', 'Justice', 'of', 'the', 'Massachusetts', 'Supreme', 'Court.', 'In', 'April', ',', 'the', 'S.J.C.', "'s", 'current', 'leader', 'Edward', 'Hennessy', 'brth', 'reaches', 'the', 'mandatory', 'retirement', 'age', 'of', 'seventy', ',', 'and', 'a', 'successor', 'is', 'expected', 'to', 'be', 'named', 'in', 'March.', 'brth', 'It', 'may', 'be', 'the', 'most', 'important', 'appointment', 'Governor', 'Michael', 'Dukakis', 'makes', 'brth', 'during', 'the', 'remainder', 'of', 'his', 'administration', 'and', 'one', 'of', 'the', 'toughest.', 'brth', 'As', 'WBUR', "'s", 'Margo', 'Melnicove', 'reports', ',', 'Hennessy', 'will', 'be', 'a', 'hard', 'act', 'to', 'follow', '.']
    dis, jump = editD(a, b, change_f = lambda x,y: 2.0*editD(x,y)/(len(x)+len(y)), fgetJump=1)
    print jump
    printAlignment(jump, a, b)
