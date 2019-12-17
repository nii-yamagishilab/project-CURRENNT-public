#!/bin/sh

### usage:
#   sh batch_normRMSE.sh PATH_TO_DIRECTORY_OF_WAVEFORM
# normalized waveforms will be in the input directory
# 
# note: 
# 1. make sure sox and sv56demo is in your path
# 2. make sure that SCRIPT_DIR points to the directory that contains batch_normRMSE.sh
###

# level of amplitude normalization, 26 by default
#LEV=26

# input directory
DATA_DIR=$1

# argument, which can be delete_origin
TEMP=$2


SCRIPT_DIR=`readlink -f $0 | xargs -I{} dirname {} `
if [ -e ${SCRIPT_DIR}/sub_normRMSE.sh ];
then
    cd ${DATA_DIR}
    ls ./ | grep wav | grep -v "norm.wav" | parallel bash ${SCRIPT_DIR}/sub_normRMSE.sh {} ${SCRIPT_DIR} ${TEMP}
else
    echo "${SCRIPT_DIR}/sub_normRMSE.sh not found"
fi

exit









for file_name in `ls ./ | grep wav`
do
    echo ${file_name}
    basename=`basename ${file_name} .wav`
    INPUT=${file_name}
    OUTPUT=${basename}_norm.wav

    SOX=sox
    SV56=sv56demo
    SCALE=${SCRIPT_DIR}/03_scale.py
    
    RAWORIG=${OUTPUT}.raw
    RAWNORM=${OUTPUT}.raw.norm
    BITS16=${OUTPUT}.16bit.wav

    SAMP=`${SOX} --i -r ${INPUT}`
    BITS=`${SOX} --i -b ${INPUT}`

    # make sure input is 16bits int
    if [ ${BITS} -ne 16 ];
    then
	python ${SCALE} ${INPUT} ${INPUT}_tmp.wav
	${SOX} ${INPUT}_tmp.wav -b 16 ${BITS16}
	${SOX} ${BITS16} ${RAWORIG}
	rm ${BITS16}
	rm ${INPUT}_tmp.wav
    else
	${SOX} ${INPUT} ${RAWORIG}
    fi

    if [ ! -z "${SV56}" ] && [ -e "${SV56}" ];
    then
    
	# norm the waveform
	${SV56} -q -sf ${SAMP} -lev -${LEV} ${RAWORIG} ${RAWNORM} > log_sv56 2>log_sv56

	# convert
	${SOX} -t raw -b 16 -e signed -c 1 -r ${SAMP} ${RAWNORM} ${OUTPUT}

	rm ${RAWNORM}
	rm ${RAWORIG}
    else
	# convert but not normed
	${SOX} -t raw -b 16 -e signed -c 1 -r ${SAMP} ${RAWORIG} ${OUTPUT}
	rm ${RAWORIG}
    fi

    if [ "${TEMP}" = "delete_origin" ];
    then
	rm ${INPUT}
    fi
done

