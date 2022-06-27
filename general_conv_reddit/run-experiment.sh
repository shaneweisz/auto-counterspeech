#!bin/bash
experiment_dir=$1

echo "Evaluating human predictions"
python eval.py --ref_file refs.txt --hyp_file $experiment_dir/Human/predictions.txt > $experiment_dir/Human/scores.txt
echo "Evaluating DialoGPT-outofthebox predictions"
python eval.py --ref_file refs.txt --hyp_file  $experiment_dir/DialoGPT-outofthebox/predictions.txt > $experiment_dir/DialoGPT-outofthebox/scores.txt
echo "Evaluating DialoGPT-finetuned predictions"
python eval.py --ref_file refs.txt --hyp_file $experiment_dir/DialoGPT-finetuned/predictions.txt > $experiment_dir/DialoGPT-finetuned/scores.txt

echo "Scraping results to csv"
bash experiments/scrape-results-to-csv.sh $experiment_dir

echo "Prettifying csv results"
python experiments/reformat-results-in-csv.py $experiment_dir
