#!/bin/sh

#####
## This is a simplified script to run the example for MODEL_DNN
## Note:
##    1. please make sure pyTools (from my github) is in PYTHONPATH
#####

## Configuration
# where is the CURRENNT_DIS (absolute path)?
prjDir=/work/smg/wang/TEMP/code/CURRENNT/examples/TEMP/CURRENNT_DIS

# path to CURRENNT
currenntBin=currennt_exp

# training data directory (where data_config.py locates)
trainDataDir=${prjDir}/EXAMPLE/DATA

# test data directory (where data_config.py locates)
testDataDir=${prjDir}/EXAMPLE/TESTDATA

# network directory (where network.jsn and config.cfg locate)
networkDir=${prjDir}/EXAMPLE/MODEL_DNN

# trained network that will be created (by default trained_network.jsn)
networkPath=${networkDir}/trained_network.jsn

# directory for generated features
outputDir=${networkDir}/output

# mlpg FLAG (If mlpgFlag is 0 and
# the target feature contains [static, delta, delta-delta] components, the output
# will only contain [static]. If mlpgFlag is 1, all the components will be generated)
mlpgFlag=0

# Additional options for different kinds of network
# for DNN network, no additional options
addOptions=None

# switch
# step1. prepare the training data
step1=1
# step2. train network
step2=1
# step3. prepare the test data
step3=1
# step4. generate from the network
step4=1
# step5. get the outut features from the raw data 
step5=1



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
	echo "\n${GRE}COMMANDLINE${NC}: python ${prjDir}/utilities/PrePareData.py \c"
	echo "${trainDataDir}/data_config.py ${trainDataDir}\n"
	python ${prjDir}/utilities/PrePareData.py ${trainDataDir}/data_config.py ${trainDataDir}

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

    
	step1Cmd="${prjDir}/utilities/PackData.py ${trainDataDir}/all.scp ${trainDataDir}/data.mv"
	step1Cmd="${step1Cmd} ${trainDataDir} 1 1 1 ${maskOpt} 0 ${normMaskOpt} ${normMethOpt}"
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

## Step2
echo "------------------------------------------------"
echo "------ Step2. Traing the network  -------"
echo "------------------------------------------------"
if [ ${step2} != 0 ]
then
    if [ -e ${networkDir}/config.cfg ] && [ -e ${networkDir}/network.jsn ]
    then
	tempDir=$PWD
	cd ${networkDir}
	cat ./config.cfg
	
	echo "\n${GRE}COMMANDLINE${NC}: ${currenntBin} --options_file config.cfg > log_train 2>&1\n"
	${currenntBin} --options_file config.cfg > log_train 2>&1
	
	cat ./log_train
	cd ${tempDir}
    
    else
	echo "File not found: ${networkDir}/config.cfg or ${networkDir}/network.jsn"
	echo "Terminate process"
	exit
    fi
else
    echo "Skip step2"
fi



## Step3
echo "------------------------------------------------"
echo "------ Step3. Packagin the test data  ----------"
echo "------------------------------------------------"
if [ ${step3} != 0 ]
then
    if [ -e ${testDataDir}/data_config.py ]
    then
	echo "\n${GRE}COMMANDLINE${NC}: python ${prjDir}/utilities/PrePareData.py \c"
	echo "${testDataDir}/data_config.py ${testDataDir}\n"
	python ${prjDir}/utilities/PrePareData.py ${testDataDir}/data_config.py ${testDataDir}

	if [ -e ${testDataDir}/mask.txt ]
	then
	    maskOpt=${testDataDir}/mask.txt
	else
	    maskOpt=None
	fi

	if [ -e ${testDataDir}/normMask ]
	then
	    normMaskOpt=${testDataDir}/normMask
	else
	    normMaskOpt=None
	fi

	if [ -e ${testDataDir}/normMethod ]
	then
	    normMethOpt=${testDataDir}/normMethod
	else
	    normMethOpt=None
	fi

    
	step1Cmd="${prjDir}/utilities/PackData.py ${testDataDir}/all.scp ${trainDataDir}/data.mv"
	step1Cmd="${step1Cmd} ${testDataDir} 1 0 1 ${maskOpt} 1 ${normMaskOpt} ${normMethOpt}"
	echo "\n${GRE}COMMANDLINE${NC}: python ${step1Cmd}\n"
	python ${step1Cmd}
    
    else
	echo "File not found: ${testDataDir}/data_config.py"
	echo "Terminate process"
	exit
    fi
else
    echo "Skip step3"
fi



## Step4
echo "------------------------------------------------"
echo "------ Step4. Generate from the network  -------"
echo "------------------------------------------------"
if [ ${step4} != 0 ]
then
    if [ -e ${networkPath} ] && [ -e ${networkDir}/config_syn.cfg ]
    then
	mkdir ${outputDir}
	
	tempDir=$PWD
	cd ${networkDir}
	cat ./config_syn.cfg

	for dataNcFile in `ls ${testDataDir}/data.nc*`
	do
	    genCommand="--options_file config_syn.cfg --ff_input_file ${dataNcFile}"
	    genCommand="${genCommand} --ff_output_file ${outputDir} --network ${networkPath}"
	    genCommand="${genCommand}"
	    if [ ${addOptions} != "None" ]
	    then
		genCommand="${genCommand} ${addOptions}"
	    fi
	    echo "\n${GRE}COMMANDLINE${NC}: ${currenntBin} ${genCommand}\n"
	    ${currenntBin} ${genCommand}
	done
	cd ${tempDir}	
	
    else
	echo "File not found: ${networkPath} or ${networkDir}/config_syn.cfg"
	echo "Terminate process"
	exit
    fi
    
else
    echo "Skip step4"
fi


## Step5
echo "------------------------------------------------"
echo "------ Step5. Get output features from raw data-"
echo "------------------------------------------------"
if [ ${step5} != 0 ]
then
    if [ -e ${testDataDir}/data_config.py ]
    then
	genCommand="${testDataDir}/data_config.py ${outputDir} ${mlpgFlag} ${outputDir} None"
	echo "\n${GRE}COMMANDLINE${NC}: python ${prjDir}/utilities/GenSynData.py ${genCommand}\n"
	python ${prjDir}/utilities/GenSynData.py ${genCommand}
	
    else
	echo "File not found: ${testDataDir}/data_config.py"
	echo "Terminate process"
	exit
    fi
else
    echo "Skip step5"
fi
