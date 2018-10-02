
import re

##########################################################
# functions

def str_pathgen(InStrList):
    """
    """
    tempStrList = str_pathgen_sub(InStrList)
    if type(tempStrList[0]) is list:
        outList = tempStrList[0]
    else:
        outList = [tempStrList[0]]
    for tmpStr in tempStrList[1:]:
        tmpBuffer = []
        for m1 in outList:
            if type(tmpStr) is list:
                for m2 in tmpStr:
                    tmpBuffer.append(str(m1)+str(m2))
            else:
                tmpBuffer.append(str(m1)+str(tmpStr))
        outList = tmpBuffer
    return outList
    
    
    
    
def str_pathgen_sub(InStrList):
    """str_pathgen_sub(InStr)
        Return list of str seperated by r'\-' in InStr
        Used by str_pathgen()
    """
    for ind,subList in enumerate(InStrList[:]):
        if type(subList) is list:
            InStrList[ind] = str_pathgen_sub(subList)
        elif type(subList) is str:
            if subList[0] == '*':
                tempStr = subList[1:]
                pat1 = re.compile(r'\-')
                se   = map(int, pat1.split(tempStr))
                if len(se) != 2:
                    print "Possible Error str_pathgen_sub:"+str(se)
                    return InStrList
                se[1] +=  1
                InStrList[ind] = range(se[0],se[1])
    return InStrList

def str_chop(InStr, FChopMore=True):
    """str_chop(InStr):
        Chop the input string
        InStr: the input string
        FChopMore: True: both '0x0d' and '0x0a' at the 
                         end will be chopped
                   False: only chop 0x0a
                   default: True
    """
    if ord(InStr[-1]) == 10 and ord(InStr[-2]) == 13 and True:
        return InStr[:-2]
    elif ord(InStr[-1]) == 10:
        return InStr[:-1]
    else:
        return InStr

def str_in_ord(InStr):
    """str_in_ord(InStr):
        Return the string in ordinary number
        
    """
    return map(ord,InStr)
    
def str_in_hex(InStr):
    """str2hex(InStr):
        Return the string in string hex number
    """
    return map(hex,str_in_ord(InStr))


if __name__ == "__main__":
    pass