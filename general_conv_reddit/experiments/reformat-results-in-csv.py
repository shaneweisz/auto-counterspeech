import pandas as pd
import sys

experiments_dir = sys.argv[1]
results_csv = f"experiments/{experiments_dir}/results.csv"

# Read in CSV
df = pd.read_csv(results_csv)

# Extract only desired metrics
PERCENTAGE_METRICS = ["BLEU-2", "BLEU-4", "METEOR", "Dist-1", "Dist-2"]
OTHER_METRICS = ["Ent-4", "AvgLen"]
df = df[["Model", "BLEU-2", "BLEU-4", "METEOR", "Ent-4", "Dist-1", "Dist-2", "AvgLen"]]

# Format the % metrics as percentages
for col in PERCENTAGE_METRICS:
    df[col] = df[col] * 100
    df[col] = round(df[col], 2)
    df.rename(columns={col: col + "(%)"}, inplace=True)

# Round off remaining metrics to 2 dp
for col in OTHER_METRICS:
    df[col] = round(df[col], 2)

# Save the processed results to CSV
df.to_csv(results_csv, index=False)
