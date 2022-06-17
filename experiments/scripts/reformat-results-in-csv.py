import pandas as pd

results_csv = "experiments/results.csv"

# Read in CSV
df = pd.read_csv(results_csv)

# Extract only desired metrics
METRICS = ["Fluency", "Toxicity", "BERTScore", "Dist-2"]
df = df[["Model"] + METRICS]

# Format the % metrics as percentages
for col in METRICS:
    df[col] = df[col] * 100
    df[col] = round(df[col], 2)
    df.rename(columns={col: col + "(%)"}, inplace=True)

# Replace row with model name == "Gold-standard" with N/A for BERTScore, since it would just be 100%
df.loc[df["Model"] == "Gold-standard", "BERTScore(%)"] = "N/A"

# Save the processed results to CSV
df.to_csv(results_csv, index=False)
