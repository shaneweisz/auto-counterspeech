#!bin/bash
EXPERIMENT_DIR="DialoGPT-vs-Finetuned-vs-Human"

python eval.py --ref_file experiments/$EXPERIMENT_DIR/Human/Human.txt --hyp_file refs.txt 2>&1 | tee experiments/$EXPERIMENT_DIR/Human/Human.scores.txt

bash experiments/scrape-results-to-csv.sh $EXPERIMENT_DIR

python experiments/reformat-results-in-csv.py $EXPERIMENT_DIR
