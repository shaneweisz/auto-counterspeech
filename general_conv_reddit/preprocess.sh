# This script
# 1) extracts inputs.6k.txt from refs.6k.txt
# 2) separates refs.6k.txt to individual files in refs directory

DATA_DIR=data

# 1)
cat $DATA_DIR/test.refs.txt | cut -f 1 > $DATA_DIR/inputs.6k.txt

# 2)
mkdir -p $DATA_DIR/refs
for (( i=2; i<=7; i++ ))
do
    # exclude the last column since this is used for the human response
    cat $DATA_DIR/test.refs.txt | rev | cut -f 2- | rev | cut -f $i > $DATA_DIR/refs/ref_$((i-1)).txt
done
