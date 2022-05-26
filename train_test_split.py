import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import csv


def main(args):
    df = pd.read_csv(args.input_file_path)

    df_no_hs_duplicates = df[df["hate_speech"].duplicated(keep=False) == False]
    df_hs_duplicates = df[df["hate_speech"].duplicated(keep=False) == True]

    df_no_hs_duplicates, df_test = train_test_split(
        df_no_hs_duplicates, test_size=args.test_size, random_state=args.seed
    )

    df_no_hs_duplicates, df_val = train_test_split(
        df_no_hs_duplicates, test_size=args.val_size
    )

    # the rest of df_no_hs_duplicates plus the rows with duplicates form the train set
    df_train = pd.concat([df_no_hs_duplicates, df_hs_duplicates])

    df_train.to_csv(args.output_dir / "train.csv", index=False, quoting=csv.QUOTE_ALL)
    df_val.to_csv(args.output_dir / "val.csv", index=False, quoting=csv.QUOTE_ALL)
    df_test.to_csv(args.output_dir / "test.csv", index=False, quoting=csv.QUOTE_ALL)


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
