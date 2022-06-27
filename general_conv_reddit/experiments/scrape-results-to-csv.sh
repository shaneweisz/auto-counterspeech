experiments_dir=$1
RESULTS_CSV="$experiments_dir/results.csv"

# Extract all metric names from all words left of ":" in the scores file
EVAL_FILES=`find . -name scores.txt`
for f in $EVAL_FILES; do
    metric_names=$(cat $f | sed 's/:.*//g')
done

# Print metric names separated by commas to the results file
echo -n "Model," > $RESULTS_CSV # since first column will be the model name
echo $metric_names | sed 's/ /,/g' >> $RESULTS_CSV

# Then extract the scores from each file and print them separated by commas
for f in $EVAL_FILES; do
    model_name=$(basename $(dirname $f))
    echo -n $model_name >> $RESULTS_CSV
    echo -n "," >> $RESULTS_CSV

    scores=$(grep -oE '\-?[0-9]+\.[0-9]+' $f)

    echo $scores | sed 's/ /,/g' >> $RESULTS_CSV
done
