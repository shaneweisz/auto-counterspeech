#!/bin/bash

# Run python evaluate.py on all files ending in predictions.txt in the experiments directory
for f in experiments/**/*predictions.txt; do
    python evaluate.py -p $f -m min-length max-length mean-length median-length -v
    # echo "python evaluate.py -p $f -m min-length max-length mean-length median-length -v"
done
