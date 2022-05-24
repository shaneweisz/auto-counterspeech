from pathlib import Path
import argparse
from preprocessing import (
    GabPreprocessor,
    RedditPreprocessor,
    ConanPreprocessor,
    MultiTargetConanPreprocessor,
)


def main(args):
    args = validate(args)

    if args.input_file_path.name == "gab.csv":
        preprocessor = GabPreprocessor
    elif args.input_file_path.name == "reddit.csv":
        preprocessor = RedditPreprocessor
    elif args.input_file_path.name == "CONAN.csv":
        preprocessor = ConanPreprocessor
    elif args.input_file_path.name == "Multitarget-CONAN.csv":
        preprocessor = MultiTargetConanPreprocessor

    preprocessor(args.input_file_path, args.output_file_path).preprocess()


def validate(args):
    supported_input_files = [
        "gab.csv",
        "reddit.csv",
        "CONAN.csv",
        "Multitarget-CONAN.csv",
    ]
    if args.input_file_path.name not in supported_input_files:
        err_msg = f"Input file name must be one of: {supported_input_files}"
        raise ValueError(err_msg)

    if args.output_file_path is None:
        default_output_file_path = (
            Path(__file__).parent / "data" / (args.input_file_path.name)
        )
        args.output_file_path = default_output_file_path

    if args.output_file_path == args.input_file_path:
        err_msg = (
            "Input and output file paths must be different, otherwise you'll overwrite"
            " the input file."
        )
        raise ValueError(err_msg)

    if not args.output_file_path.name.endswith(".csv"):
        err_msg = "Output file name must end with .csv"
        raise ValueError(err_msg)

    return args


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-i", "--input_file_path", type=Path, required=True)
    argparser.add_argument("-o", "--output_file_path", type=Path)
    args = argparser.parse_args()
    main(args)
