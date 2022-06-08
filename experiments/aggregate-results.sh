# NB: Run from base directory

RESULTS_CSV=experiments/results.csv

# extract metrics from all words left of ":" using grep
EVAL_FILES=`find experiments -name *predictions.scores.txt -not -path *archive*`
for f in $EVAL_FILES; do
    metric_names=$(cat $f | sed 's/:.*//g')
done

# print metric names separated by commas
echo -n "Model," > $RESULTS_CSV # since first column will be the model name
echo $metric_names | sed 's/ /,/g' >> $RESULTS_CSV

# extract scores from each file and print them separated by commas
for f in $EVAL_FILES; do
    model_name=$(basename $(dirname $f))
    echo -n $model_name >> $RESULTS_CSV
    echo -n "," >> $RESULTS_CSV

    scores=$(grep -oE '[0-9]+\.[0-9]+' $f)
    echo $scores | sed 's/ /,/g' >> $RESULTS_CSV
done

python experiments/format-results.py
