#!/bin/sh

LEV=26

INPUT=$1
OUTPUT=$2

SOX=$3
SV56=$4

RAWORIG=${OUTPUT}.raw
RAWNORM=${OUTPUT}.raw.norm
BITS16=${OUTPUT}.16bit.wav

SAMP=`${SOX} --i -r ${INPUT}`
BITS=`${SOX} --i -b ${INPUT}`

# make sure input is 16bits int
if [ ${BITS} -ne 16 ];
then
    ${SOX} ${INPUT} -b 16 ${BITS16}
    ${SOX} ${BITS16} ${RAWORIG}
    rm ${BITS16}
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
