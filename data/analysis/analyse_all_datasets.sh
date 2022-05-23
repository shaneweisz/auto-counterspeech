python analysis/analyse_dataset.py -f gab.csv | tee analysis/summary-statistics.txt
python analysis/analyse_dataset.py -f reddit.csv | tee -a analysis/summary-statistics.txt
python analysis/analyse_dataset.py -f CONAN.csv | tee -a analysis/summary-statistics.txt
python analysis/analyse_dataset.py -f Multitarget-CONAN.csv | tee -a analysis/summary-statistics.txt

echo "Results written to analysis/summary-statistics.txt"
