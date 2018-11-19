#!/bin/sh

INPUT=$1
OUTPUT=$2
SAMP=$3
SOX=$4
${SOX} ${INPUT} -r ${SAMP} -b 16 ${OUTPUT}
