#!/bin/sh
# usage:
#  1. configure NETWORK, OUTPUT, CURRENNT
#  2. sh plot_network_graph.sh
# requirement:
#  1. CURRENNT
#  2. dot tools from https://www.graphviz.org/
#
# note:
#   NETWORK can be *.jsn or *.autosave

# path to input *.autosave
NETWORK=./network.jsn
OUTPUT=./network.pdf

# path to CURRENNT
CURRENNT=currennt

currennt --network_graph ${OUTPUT}.gv --network ${NETWORK} --cuda off
dot -Tpdf ${OUTPUT}.gv -o ${OUTPUT}
rm ${OUTPUT}.gv
