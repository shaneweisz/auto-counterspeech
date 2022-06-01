import argparse
from pathlib import Path
import pandas as pd
import logging
import sys


def main(args):
    setup_logging(args.output_path)
    for file_path in args.file_paths:
        analyse_dataset(file_path)


def setup_logging(output_path):
    config = get_logging_config(output_path)
    logging.basicConfig(**config)
    if output_path:
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def get_logging_config(output_path):
    config = {}
    config["level"] = logging.INFO
    config["format"] = "%(message)s"
    if output_path:
        config["filename"] = output_path
        config["filemode"] = "w"
    return config


def analyse_dataset(file_path):
    df = pd.read_csv(file_path)

    log_header(file_path)

    log_num_rows(df)

    log_sample_row(df)

    log_num_unique(df, "hate_speech")
    log_num_unique(df, "counter_speech")

    log_median_length(df, "hate_speech")
    log_median_length(df, "counter_speech")

    log_distribution_of_num_repeats(df, "hate_speech")
    log_distribution_of_num_repeats(df, "counter_speech")


def log_header(file_path):
    DASHED_LINE = "-" * 100
    logging.info(DASHED_LINE)
    logging.info(f"Analysing {file_path}")
    logging.info(DASHED_LINE)


def log_num_rows(df):
    logging.info(f"Number of rows: {len(df)}")


def log_sample_row(df, idx=42):
    logging.info(f"Random HS comment: {df['hate_speech'][idx]}")
    logging.info(f"Random CS response: {df['counter_speech'][idx]}")


def log_num_unique(df, col="hate_speech"):
    logging.info(f"Number of unique {col} comments: {len(df[col].unique())}")


def log_median_length(df, col="hate_speech"):
    logging.info(
        f"Median length of {col} in words:"
        f" {df[col].apply(lambda x: len(str(x).split())).median()}"
    )


def log_distribution_of_num_repeats(df, col="hate_speech"):
    logging.info(f"Distribution of number of repeats of {col} responses:")
    logging.info(
        df[col]
        .value_counts()
        .groupby(df[col].value_counts())
        .count()
        .to_frame()
        .T.to_string(index=False)
    )


ALL_DATASETS = ["gab", "reddit", "CONAN", "Multitarget-CONAN"]
ALL_PROCESSED_DATASET_FILES = [
    f"data/preprocessed/{dataset}.csv" for dataset in ALL_DATASETS
]

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-f",
        "--file_paths",
        type=Path,
        nargs="+",
        default=ALL_PROCESSED_DATASET_FILES,
    )
    argparser.add_argument("-o", "--output_path", type=Path)
    args = argparser.parse_args()
    main(args)
