import argparse
from pathlib import Path
from evaluation.evaluator import Evaluator


def main(args):
    evaluator = evaluator_from_file_path(args.file_path)
    evaluator.evaluate(args.metrics)


def evaluator_from_file_path(file_path):
    if file_path.suffix == ".csv":
        return Evaluator.from_csv(file_path)
    elif file_path.suffix == ".json":
        return Evaluator.from_json(file_path)
    else:
        raise ValueError("File must be .csv or .json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_path", type=Path, required=True)
    parser.add_argument(
        "-m",
        "--metrics",
        nargs="+",
        default=["sacrebleu"],
    )
    args = parser.parse_args()
    main(args)
