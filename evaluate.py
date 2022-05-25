import argparse
from pathlib import Path
from evaluation.evaluator import Evaluator
from evaluation.metrics import MetricFactory


def main(args):
    if args.verbose:
        print(f"Loading metrics: {args.metrics}")
    metrics = [MetricFactory.from_metric_name(name) for name in args.metrics]
    evaluator = Evaluator.from_csv(args.file_path)
    scores = evaluator.evaluate(metrics, args.verbose)
    print_scores(scores)


def print_scores(scores):
    for metric_name, score in scores.items():
        print(f"{metric_name}: {score:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_path", type=Path, required=True)
    parser.add_argument(
        "-m",
        "--metrics",
        nargs="+",
        default=["bleu-2"],
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    main(args)
