import pandas as pd
from krippendorff import alpha

results_file = "human_evaluation/analysis/results_all_raters.tsv"
reliability_data = pd.read_csv(results_file, sep="\t", header=None).to_numpy()
score = alpha(reliability_data=reliability_data, level_of_measurement="ordinal", value_domain=[1, 2, 3, 4, 5])
print(f"Krippendorff's alpha: {score:.2f}")
