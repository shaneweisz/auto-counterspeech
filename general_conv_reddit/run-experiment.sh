#!bin/bash

EXPERIMENT_DIR=experiments/DialoGPT-vs-Finetuned-vs-Human
DATA_DIR=data

echo "Evaluating human predictions"
python evaluate.py --refs_dir $DATA_DIR/refs --hyp_file $EXPERIMENT_DIR/Human/predictions.txt > $EXPERIMENT_DIR/Human/scores.txt

echo "Evaluating DialoGPT-outofthebox predictions"
python util/clean-str.py $EXPERIMENT_DIR/DialoGPT-outofthebox/predictions.txt
python evaluate.py --refs_dir $DATA_DIR/refs --hyp_file  $EXPERIMENT_DIR/DialoGPT-outofthebox/predictions.cleaned.txt > $EXPERIMENT_DIR/DialoGPT-outofthebox/scores.txt

echo "Evaluating DialoGPT-finetuned predictions"
python util/clean-str.py $EXPERIMENT_DIR/DialoGPT-finetuned/predictions.txt
python evaluate.py --refs_dir $DATA_DIR/refs --hyp_file $EXPERIMENT_DIR/DialoGPT-finetuned/predictions.cleaned.txt > $EXPERIMENT_DIR/DialoGPT-finetuned/scores.txt

echo "Scraping results to csv"
bash experiments/scrape-results-to-csv.sh $EXPERIMENT_DIR

echo "Prettifying csv results"
python experiments/reformat-results-in-csv.py $EXPERIMENT_DIR
