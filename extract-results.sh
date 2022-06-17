# First scrape all results from `experiments` subdirectories to `experiments/results.csv`
bash experiments/scripts/scrape-results-to-csv.sh

# Then reformat the results CSV nicely
python experiments/scripts/reformat-results-in-csv.py
