import pandas as pd
import sys

experiments_dir = sys.argv[1]
results_csv = f"{experiments_dir}/results.csv"

# Read in CSV
df = pd.read_csv(results_csv)

# Extract only desired metrics
PERCENTAGE_METRICS = ["Fluency", "Toxicity", "BERTScore", "BLEU-4", "Dist-2"]
OTHER_METRICS = ["Ent-4", "AvgLen"]
df = df[["Model"] + PERCENTAGE_METRICS + OTHER_METRICS]

# Format the % metrics as percentages
for col in PERCENTAGE_METRICS:
    df[col] = df[col] * 100
    df[col] = round(df[col], 1)
    df.rename(columns={col: col + "(%)"}, inplace=True)

# Round off remaining metrics to 1 dp
for col in OTHER_METRICS:
    df[col] = round(df[col], 1)

# Replace row with model name == "Gold-standard" with N/A for BERTScore, since it would just be 100%
df.loc[df["Model"] == "Gold-standard", "BERTScore(%)"] = "N/A"
df.loc[df["Model"] == "Gold-standard", "BLEU-4(%)"] = "N/A"

# Save the processed results to CSV
df.to_csv(results_csv, index=False)
