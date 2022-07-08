import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import csv
from util.file_io import write_list_to_file


def main(args):
    """
    This script is used to create the `gab` and `reddit` splits, and enforces that there is no overlap in inputs
    between the training and validation/test sets.
    """
    print(f"Seed: {args.seed}, test_size: {args.test_size}, val_size: {args.val_size}")

    print(f"Reading csv file from {args.input_file_path}")
    df = pd.read_csv(args.input_file_path)

    print("Splitting data into train/val/test sets")
    n_val = int(args.val_size * len(df))
    n_test = int(args.test_size * len(df))
    df, df_test = train_test_split(df, test_size=n_test, random_state=args.seed, shuffle=False)
    df, df_val = train_test_split(df, test_size=n_val, random_state=args.seed, shuffle=False)
    df_train = df

    print(f"Writing train, val, and test sets to csv's in {args.output_dir}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)  # create the output directories if they don't exist
    df_train.to_csv(args.output_dir / "train.csv", index=False, quoting=csv.QUOTE_ALL)
    df_val.to_csv(args.output_dir / "val.csv", index=False, quoting=csv.QUOTE_ALL)
    df_test.to_csv(args.output_dir / "test.csv", index=False, quoting=csv.QUOTE_ALL)

    print(f"Writing the test inputs and references to txt files in {args.output_dir}")
    inputs = [str(text).replace("\n", " ") for text in list(df_test["hate_speech"])]
    references = [str(text).replace("\n", " ") for text in list(df_test["counter_speech"])]
    print(len(inputs), len(references))
    write_list_to_file(args.output_dir / "test.inputs.txt", inputs)
    write_list_to_file(args.output_dir / "test.references.txt", references)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-i", "--input_file_path", type=Path, required=True)
    argparser.add_argument(
        "-t",
        "--test_size",
        type=float,
        default=0.1,
    )
    argparser.add_argument(
        "-v",
        "--val_size",
        type=float,
        default=0.1,
    )
    argparser.add_argument("-o", "--output_dir", type=Path, default="data")
    argparser.add_argument("--seed", default=42)
    args = argparser.parse_args()
    main(args)
