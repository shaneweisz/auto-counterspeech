import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import csv


def main(args):
    """
    This script is used to create the `multitarget-conan` splits, by enforcing there is no overlap in inputs between
    the training and validation/test sets.
    """
    print(f"Seed: {args.seed}, test_size: {args.test_size}, val_size: {args.val_size}")

    print(f"Reading csv file from {args.input_file_path}")
    df = pd.read_csv(args.input_file_path)

    df_no_hs_duplicates = df[df["hate_speech"].duplicated(keep=False) == False]
    df_hs_duplicates = df[df["hate_speech"].duplicated(keep=False) == True]

    print("Creating test set from rows with unique hate speech")
    df_no_hs_duplicates, df_test = train_test_split(
        df_no_hs_duplicates, test_size=args.test_size, random_state=args.seed
    )

    print("Creating validation set from remaining rows with unique hate speech")
    df_no_hs_duplicates, df_val = train_test_split(
        df_no_hs_duplicates, test_size=args.val_size, random_state=args.seed
    )

    print("Creating train set from all remaining rows")
    df_train = pd.concat([df_no_hs_duplicates, df_hs_duplicates])

    print(f"Writing train, val, and test sets to csv's in {args.output_dir}")
    args.output_dir.mkdir(exist_ok=True, parents=True)
    df_train.to_csv(args.output_dir / "train.csv", index=False, quoting=csv.QUOTE_ALL)
    df_val.to_csv(args.output_dir / "val.csv", index=False, quoting=csv.QUOTE_ALL)
    df_test.to_csv(args.output_dir / "test.csv", index=False, quoting=csv.QUOTE_ALL)

    print(f"Writing the test inputs and references to txt files in {args.output_dir}")
    with open(args.output_dir / "test.inputs.txt", "w") as f:
        for row in df_test.itertuples():
            f.write(row.hate_speech + "\n")
    with open(args.output_dir / "test.references.txt", "w") as f:
        for row in df_test.itertuples():
            f.write(row.counter_speech + "\n")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-i", "--input_file_path", type=Path, required=True)
    argparser.add_argument(
        "-t",
        "--test_size",
        type=int,
        default=500,
    )
    argparser.add_argument(
        "-v",
        "--val_size",
        type=int,
        default=500,
    )
    argparser.add_argument("-o", "--output_dir", type=Path, default="data")
    argparser.add_argument("--seed", default=42)
    args = argparser.parse_args()
    main(args)
