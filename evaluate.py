import argparse
from pathlib import Path
from evaluation.evaluator import Evaluator
from evaluation.metrics import MetricFactory


def main(args):
    metrics = load_metrics(args.metrics)
    evaluator = setup_evaluator(args)
    scores = evaluator.evaluate(metrics, args.verbose)
    output_scores(scores, args)


def load_metrics(metric_names):
    return [MetricFactory.from_metric_name(name) for name in metric_names]


def setup_evaluator(args):
    if args.from_csv:
        evaluator = Evaluator.from_csv(args.from_csv)
    else:
        evaluator = Evaluator.from_files(args.predictions, args.references, args.inputs)
    return evaluator


def output_scores(scores, args):
    print_scores(scores)
    if not args.no_save_to_file:
        save_scores_to_file(scores, args)


def print_scores(scores):
    print("=" * 20 + "\nScores:\n" + "=" * 20)
    for metric_name, score in scores.items():
        output_line = f"{metric_name+':':<12} {score:.3f}"
        print(output_line)
    print("=" * 20)


def save_scores_to_file(scores, args):
    file_with_predictions = args.from_csv if args.from_csv else args.predictions
    output_path = Path(file_with_predictions).with_suffix(".eval.txt")
    with output_path.open("w") as f:
        for metric_name, score in scores.items():
            output_line = f"{metric_name+':':<12} {score:.3f}"
            f.write(output_line + "\n")
    print(f"Saved scores to {output_path.absolute()}")
    return output_path


DEFAULT_METRICS = [
    "gruen",
    "dist1",
    "dist2",
    "ent1",
    "ent2",
    "bert-score",
    "bleu1",
    "bleu2",
    "rouge1",
    "rouge2",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--predictions", type=Path)
    parser.add_argument("-r", "--references", type=Path)
    parser.add_argument("-i", "--inputs", type=Path)
    parser.add_argument("--from_csv", type=Path)
    parser.add_argument("-o", "--no_save_to_file", action="store_true")
    parser.add_argument(
        "-m",
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    main(args)
