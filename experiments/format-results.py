import pandas as pd

results_csv = "/home/sw984/rds/hpc-work/mphil-project/auto-counterspeech/experiments/results.csv"

df = pd.read_csv(results_csv)

# Format columns as percentages
COLS = ["GRUEN", "BERTScore", "BLEU-2", "Dist-1", "Dist-2"]
for col in COLS:
    df[col] = df[col] * 100
    df.rename(columns={col: col + "(%)"}, inplace=True)

non_modelname_cols = [col for col in df.columns if col != "Model"]
for col in non_modelname_cols:
    df[col] = round(df[col], 2)

df.to_csv(results_csv, index=False)
