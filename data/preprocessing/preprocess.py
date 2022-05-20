from pathlib import Path
import argparse
from gab_preprocessor import GabPreprocessor


def main(args):
    assert args.file_path.name in ["gab.csv", "reddit.csv"]

    if args.file_path.name in ["gab.csv", "reddit.csv"]:
        preprocessor = GabPreprocessor
    elif args.file_path.name in ["CONAN.csv"]:
        pass

    preprocessor(args.file_path).preprocess()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-f",
        "--file_path",
        type=Path,
        required=True,
        help="Path to unprocessed gab.csv, reddit.csv or CONAN.csv",
    )
    args = argparser.parse_args()
    main(args)
