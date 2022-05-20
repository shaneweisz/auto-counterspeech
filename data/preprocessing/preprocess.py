from pathlib import Path
import argparse
from gab_preprocessor import GabPreprocessor, RedditPreprocessor
from conan_preprocessor import ConanPreprocessor, MultiTargetConanPreprocessor


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
    assert args.input_file_path.name in [
        "gab.csv",
        "reddit.csv",
        "CONAN.csv",
        "Multitarget-CONAN.csv",
    ]

    if args.output_file_path is None:
        default_output_file_path = Path(__file__).parent.parent / (
            args.input_file_path.name
        )
        args.output_file_path = default_output_file_path

    assert args.output_file_path != args.input_file_path
    assert args.output_file_path.name.endswith(".csv")

    return args


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-i", "--input_file_path", type=Path, required=True)
    argparser.add_argument("-o", "--output_file_path", type=Path)
    args = argparser.parse_args()
    main(args)
