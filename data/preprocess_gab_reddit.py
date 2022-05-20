import argparse
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import ast
import re


def preprocess(file_path: Path) -> None:
    print(f"Loading {file_path}")
    df = load_dataset(file_path)

    print(f"Extracting HS-CS pairs from {file_path}")
    df = remove_conversations_with_no_hate_speech(df)
    hs_cs_pairs = extract_hs_cs_pairs(df)

    new_file_path = write_to_new_csv(hs_cs_pairs, file_path)
    print(f"Processed file written to {new_file_path}")


def load_dataset(file_path: Path) -> pd.DataFrame:
    return pd.read_csv(file_path)


def remove_conversations_with_no_hate_speech(df: pd.DataFrame) -> pd.DataFrame:
    return df[pd.notnull(df["hate_speech_idx"])]


def extract_hs_cs_pairs(df: pd.DataFrame) -> List[Tuple[str, str]]:
    hs_cs_pairs = []
    for i in range(len(df)):
        row = df.iloc[i]

        text = row["text"]
        hs_idxes = ast.literal_eval(row["hate_speech_idx"])
        cs_responses = ast.literal_eval(row["response"])

        hss = []
        for hs_idx in hs_idxes:
            r = rf"{hs_idx}\..*"
            try:
                hs = re.findall(r, text)[0]
                hs = hs.split(".", maxsplit=1)[1].strip()  # get HS text after the index
                hss.append(hs)
            except IndexError:
                print(f"Error: HS idx {hs_idx} not a valid index in the conversation.")
        for hs in hss:
            for cs in cs_responses:
                hs_cs_pairs.append((hs, cs))

    return hs_cs_pairs


def write_to_new_csv(hs_cs_pairs: List[Tuple[str, str]], orig_file_path: Path) -> Path:
    HEADER = "HateSpeech,CounterSpeech"
    new_file_path = orig_file_path.parent / (orig_file_path.stem + "_processed.csv")
    with open(new_file_path, "w") as f:
        f.write(HEADER + "\n")
        for hs, cs in hs_cs_pairs:
            f.write(f'"{hs}","{cs}"\n')
    return new_file_path


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-f",
        "--file_path",
        type=Path,
        required=True,
        help="Path to gab-orig.csv or reddit-orig.csv",
    )
    args = argparser.parse_args()
    assert args.file_path.name in ["gab.csv", "reddit.csv"]
    preprocess(args.file_path)
