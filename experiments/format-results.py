import pandas as pd

results_csv = "/home/sw984/rds/hpc-work/mphil-project/auto-counterspeech/experiments/results.csv"

df = pd.read_csv(results_csv)

# Reorder columns
df = df[["Model", "Fluency", "BLEU-2", "BERTScore", "Toxicity", "Dist-1", "Dist-2", "Ent-4", "AvgLen"]]

# Format columns as percentages
COLS = ["Fluency", "BERTScore", "Toxicity", "BLEU-2", "Dist-1", "Dist-2"]
for col in COLS:
    df[col] = df[col] * 100
    df.rename(columns={col: col + "(%)"}, inplace=True)

non_modelname_cols = [col for col in df.columns if col != "Model"]
for col in non_modelname_cols:
    df[col] = round(df[col], 2)

# replace row with model name == "oracle" with N/A for BERTScore and BLEU-2
df.loc[df["Model"] == "oracle", "BERTScore(%)"] = "N/A"
df.loc[df["Model"] == "oracle", "BLEU-2(%)"] = "N/A"

# change model names to be more readable
df.loc[df["Model"] == "oracle", "Model"] = "Oracle"
df.loc[df["Model"] == "dialoGPT-finetuned-mtconan-beam10", "Model"] = "DialoGPT-finetuned"
df.loc[df["Model"] == "dialoGPT-outofthebox-beam10", "Model"] = "DialoGPT-outofthebox"
df.loc[df["Model"] == "GPS-3000epochs", "Model"] = "GPS"

df["Model"] = pd.Categorical(df["Model"], ["DialoGPT-outofthebox", "GPS", "DialoGPT-finetuned", "Oracle"])
df = df.sort_values(by=["Model"])

df.to_csv(results_csv, index=False)
