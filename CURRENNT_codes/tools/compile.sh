#!/bin/sh

g++ -ohtk2nc htk2nc.cpp -lnetcdf
g++ -onc-standardize nc-standardize.cpp -lnetcdf -lm
ln -s nc-standardize nc-standardize-input
