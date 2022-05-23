import argparse
from pathlib import Path
import pandas as pd


def main(args):
    df = pd.read_csv(args.file_path)

    print(f"Analysing {args.file_path}:")
    print("-" * 50)

    print(f"Number of rows: {len(df)}")

    idx = 1000
    print(f"Random HS comment: {df['hate_speech'][idx]}")
    print(f"Random CS response: {df['counter_speech'][idx]}")

    print(f"Number of unique HS comments: {len(df['hate_speech'].unique())}")
    print(f"Number of unique CS responses: {len(df['counter_speech'].unique())}")

    print(
        f"Median length of HS in words: {df['hate_speech'].apply(lambda x: len(str(x).split())).median()}"
    )
    print(
        f"Median length of CS in words: {df['counter_speech'].apply(lambda x: len(str(x).split())).median()}"
    )

    print()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-f", "--file_path", type=Path, help="csv file to analyse")
    args = argparser.parse_args()
    assert args.file_path.name.endswith(".csv")
    main(args)
