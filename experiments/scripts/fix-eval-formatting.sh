

# remove all lines with "len", "ent" etc. in all files in experiments/**/*predictions.eval.txt
PRED_TXTS=`find experiments -name *predictions.eval.txt`
for f in $PRED_TXTS; do
    sed -i -e '/Len/d' $f
    sed -i -e '/Ent/d' $f
    sed -i -e '/Dist/d' $f
    # sed -i -e '/BLEU/d' $f
    # sed -i -e '/ROUGE/d' $f
done

# replace "MinResponseLength" with "MinLen" in all files in experiments/**/*predictions.eval.txt
# for f in experiments/**/*predictions.eval.txt; do
#     sed -i -e 's/minResponseLength/MinLen/g' $f
#     sed -i -e 's/maxResponseLength/MaxLen/g' $f
#     sed -i -e 's/meanResponseLength/MeanLen/g' $f
#     sed -i -e 's/medianResponseLength/MedianLen/g' $f
# done
