#!/bin/sh

LEV=26

INPUT=$1
OUTPUT=$2

SOX=$3
SV56=$4

RAWORIG=${INPUT}.raw
RAWNORM=${INPUT}.raw.norm
BITS16=${INPUT}.16bit.wav

SAMP=`sox --i -r ${INPUT}`
BITS=`sox --i -b ${INPUT}`

# make sure input is 16bits int
if [ ${BITS} -ne 16 ];
then
    sox ${INPUT} -b 16 ${BITS16}
    sox ${BITS16} ${RAWORIG}
    rm ${BITS16}
else
    sox ${INPUT} ${RAWORIG}
fi

if [ -e ${SV56} ];
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
