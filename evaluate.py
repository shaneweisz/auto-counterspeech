import argparse
from pathlib import Path
from evaluation.evaluator import Evaluator
from evaluation.metrics.metric_factory import MetricFactory


def main(args):
    metrics = [MetricFactory.from_metric_name(name) for name in args.metrics]
    evaluator = Evaluator.from_csv(args.file_path)
    scores = evaluator.evaluate(metrics)
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
    args = parser.parse_args()
    main(args)
