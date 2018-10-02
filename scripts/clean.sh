#!/bin/sh

echo "Clean the EXAMPLE directory"
rm EXAMPLE/*/*.*~
rm EXAMPLE/*/data.nc*
rm EXAMPLE/*/data.mv
rm EXAMPLE/*/data.scp
rm EXAMPLE/*/all.scp*

rm EXAMPLE/MODEL_*/epo*
rm EXAMPLE/MODEL_*/trained_network*
rm -r EXAMPLE/MODEL_*/out*

rm CONFIGPOOLS/*.*~
rm utilities/*.*~

