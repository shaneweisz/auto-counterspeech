import argparse
from pathlib import Path
import pandas as pd
import logging
import sys


def main(args):
    logging.basicConfig(
        filename=args.output_path,
        level=logging.INFO,
        format="%(message)s",
        filemode="w",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    for file_path in args.file_paths:
        analyse_dataset(file_path)


def analyse_dataset(file_path):
    df = pd.read_csv(file_path)

    logging.info("-" * 100)
    logging.info(f"Analysing {file_path}:")
    logging.info("-" * 100)

    logging.info(f"Number of rows: {len(df)}")

    idx = 1000
    logging.info(f"Random HS comment: {df['hate_speech'][idx]}")
    logging.info(f"Random CS response: {df['counter_speech'][idx]}")

    logging.info(f"Number of unique HS comments: {len(df['hate_speech'].unique())}")
    logging.info(f"Number of unique CS responses: {len(df['counter_speech'].unique())}")

    logging.info(
        "Median length of HS in words:"
        f" {df['hate_speech'].apply(lambda x: len(str(x).split())).median()}"
    )
    logging.info(
        "Median length of CS in words:"
        f" {df['counter_speech'].apply(lambda x: len(str(x).split())).median()}"
    )

    logging.info("Distribution of number of repeats of CS responses:")
    logging.info(
        df["counter_speech"]
        .value_counts()
        .groupby(df["counter_speech"].value_counts())
        .count()
        .to_frame()
        .T.to_string(index=False)
    )

    logging.info("Distribution of number of repeats of HS comments:")
    logging.info(
        df["hate_speech"]
        .value_counts()
        .groupby(df["hate_speech"].value_counts())
        .count()
        .to_frame()
        .T.to_string(index=False)
    )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    datasets = ["gab", "reddit", "CONAN", "Multitarget-CONAN"]
    argparser.add_argument(
        "-f",
        "--file_paths",
        type=Path,
        nargs="+",
        default=[f"data/processed/{dataset}.csv" for dataset in datasets],
    )
    argparser.add_argument(
        "-o",
        "--output_path",
        type=Path,
        default="data/analysis/summary_statistics.txt",
    )
    args = argparser.parse_args()
    main(args)
