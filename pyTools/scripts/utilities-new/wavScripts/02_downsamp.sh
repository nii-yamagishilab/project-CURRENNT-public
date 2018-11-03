#!/bin/sh

INPUT=$1
OUTPUT=$2
SOX=$3
${SOX} ${INPUT} -r 16000 -b 16 ${OUTPUT}
