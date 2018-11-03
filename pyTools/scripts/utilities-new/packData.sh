#!/bin/sh

#####
## This is a simplified script to run the example for MODEL_DNN
## Note:
##    1. please make sure pyTools (from my github) is in PYTHONPATH
#####

## Configuration
# where is the utilities directory?
utiDir=$PWD/utilities/dataProcess

# training data directory (where data_config.py locates)
trainDataDir=$1
# test data directory (where data_config.py locates)
testDataDir=0

# switch
# whether data.mv is needed?
getMV=0
# whether normalize the data (if yes, getMV must be 1)
normData=0

# step1. prepare the training data
step1=1



#############################
##
GRE='\033[0;32m'
NC='\033[0m' # No Color


## Step1 
echo "------------------------------------------------"
echo "------ Step1. Packaing the training data -------"
echo "------------------------------------------------"
if [ ${step1} != 0 ]
then
    if [ -e ${trainDataDir}/data_config.py ]
    then
	rm ${trainDataDir}/*.info
	echo "\n${GRE}COMMANDLINE${NC}: python ${utiDir}/PrePareData.py \c"
	echo "${trainDataDir}/data_config.py ${trainDataDir}\n"
	python ${utiDir}/PrePareData.py ${trainDataDir}/data_config.py ${trainDataDir}

	if [ -e ${trainDataDir}/mask.txt ]
	then
	    maskOpt=${trainDataDir}/mask.txt
	else
	    maskOpt=None
	fi

	if [ -e ${trainDataDir}/normMask ]
	then
	    normMaskOpt=${trainDataDir}/normMask
	else
	    normMaskOpt=None
	fi

	if [ -e ${trainDataDir}/normMethod ]
	then
	    normMethOpt=${trainDataDir}/normMethod
	else
	    normMethOpt=None
	fi

	
	step1Cmd="${utiDir}/PackData.py ${trainDataDir}/all.scp ${trainDataDir}/data.mv"
	step1Cmd="${step1Cmd} ${trainDataDir} 1 ${getMV} ${normData} ${maskOpt} 0 ${normMaskOpt} ${normMethOpt}"
	echo "\n${GRE}COMMANDLINE${NC}: python ${step1Cmd}\n"
	python ${step1Cmd}
    
    else
	echo "File not found: ${trainDataDir}/data_config.py"
	echo "Terminate process"
	exit
    fi
else
    echo "Skip step1"
fi


