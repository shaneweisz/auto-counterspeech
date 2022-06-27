#!bin/bash

python eval.py --ref_file Human/Human.txt --hyp_file refs.txt 2>&1 | tee Human/Human.scores.txt

bash scrape-results-to-csv.sh

python reformat-results-in-csv.py
