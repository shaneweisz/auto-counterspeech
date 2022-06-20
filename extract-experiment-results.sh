# Script that extracts the results stored in an `experiments_dir` folder in the experiments directory,
# and writes them to a CSV file.
# The desired `experiments_dir` must contain evaluation results files of the format `*predictions.scores.txt`.

# 1. Extract the desired experiments directory from the first argument
experiments_dir=$1

# 2. Scrape all results from this directory to `experiments/$experiments_dir/results.csv`
bash experiments/scrape-results-to-csv.sh $experiments_dir

# 3. Then reformat the results CSV nicely
python experiments/reformat-results-in-csv.py $experiments_dir
