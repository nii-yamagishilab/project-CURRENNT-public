#!/bin/sh

echo "Compiling the binaryTools"
cd binaryTools
python setup.py build_ext --inplace
echo "Done"
