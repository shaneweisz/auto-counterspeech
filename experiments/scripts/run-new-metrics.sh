#!/bin/bash

# PRED_TXTS=`find experiments -name *predictions.txt`
PRED_TXTS=`find experiments -name *predictions.txt`
for f in $PRED_TXTS; do
    echo "python evaluate.py -p $f -v"
    python evaluate.py -p $f -v
done
