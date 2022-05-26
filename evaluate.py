import argparse
from pathlib import Path
from evaluation.evaluator import Evaluator
from evaluation.metrics import MetricFactory


def main(args):
    metrics = load_metrics(args.metrics)
    evaluator = setup_evaluator(args)
    scores = evaluator.evaluate(metrics, args.verbose)
    output_path = output_scores(scores)
    print(f"Saved scores to {output_path.absolute()}")


def load_metrics(metric_names):
    return [MetricFactory.from_metric_name(name) for name in metric_names]


def setup_evaluator(args):
    if args.from_csv:
        evaluator = Evaluator.from_csv(args.from_csv)
    else:
        evaluator = Evaluator.from_files(args.predictions, args.references, args.inputs)
    return evaluator


def output_scores(scores):
    print("=" * 20 + "\nScores:\n" + "=" * 20)
    file_with_predictions = args.from_csv if args.from_csv else args.predictions
    output_path = Path(file_with_predictions).with_suffix(".eval.txt")
    with output_path.open("w") as f:
        for metric_name, score in scores.items():
            output_line = f"{metric_name+':':<12} {score:.3f}"
            f.write(output_line + "\n")
            print(output_line)
    print("=" * 20)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--predictions", type=Path)
    parser.add_argument("-r", "--references", type=Path)
    parser.add_argument("-i", "--inputs", type=Path)
    parser.add_argument("--from_csv", type=Path)
    parser.add_argument(
        "-m",
        "--metrics",
        nargs="+",
        default=["bleu-2"],
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    main(args)
