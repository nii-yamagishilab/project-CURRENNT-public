#!/bin/sh
# usage:
#  1. configure AUTOSAVEMODEL, OUTPUTJSN, CURRENNT
#  2. sh convert_autosave_to_jsn.sh
# note:
#   *.autosave contains network weights, gradients and other statics to continue training
#   *.jsn only contains network weights
#   You can currennt --continue epoch***.autosave to continue training
#   You cannot currennt --continue *.jsn to continue training

# path to input *.autosave
AUTOSAVEMODEL=./epoch020.autosave
OUTPUTJSN=./epoch020.jsn

# path to CURRENNT
CURRENNT=currennt

currennt --print_weight_to ${OUTPUTJSN} --print_weight_opt 2 --network ${AUTOSAVEMODEL} --cuda off
