import argparse
from pathlib import Path
from evaluation.metrics import MetricFactory


def main(args):
    # Load metrics, predictions and references
    metrics = load_metrics(args.metrics)
    preds, refs, inputs = read_files_into_lists(args.predictions, args.references, args.inputs)

    # Compute metrics
    for metric in metrics:
        print("Evaluating metric:", metric.name)
        metric.compute_score(preds, references=refs, verbose=args.verbose)

    # Output results
    print_scores(metrics)
    if not args.no_save:
        save_scores_to_file(metrics, args)
        save_detailed_scores_to_file(metrics, args, preds)


def load_metrics(metric_names):
    return [MetricFactory.from_metric_name(name) for name in metric_names]


def read_files_into_lists(predictions_path: Path, references_path: Path = None, inputs_path: Path = None):
    predictions = [line.strip() for line in predictions_path.open()]
    if references_path:
        references = [line.strip() for line in references_path.open()]
        assert len(predictions) == len(references)
    else:
        references = None
    if inputs_path:
        inputs = [line.strip() for line in inputs_path.open()]
        assert len(predictions) == len(inputs)
    else:
        inputs = None
    return predictions, references, inputs


def print_scores(computed_metrics):
    print("=" * 20 + "\nScores:\n" + "=" * 20)
    for metric in computed_metrics:
        output_line = f"{metric.name+':':<12} {metric.score:.3f}"
        print(output_line)
    print("=" * 20)


def save_scores_to_file(computed_metrics, args):
    output_path = Path(args.predictions).with_suffix(".eval.txt")
    with output_path.open("a") as f:
        for metric in computed_metrics:
            output_line = f"{metric.name +':':<12} {metric.score:.3f}"
            f.write(output_line + "\n")
    print(f"Saved scores to {output_path.absolute()}")


def save_detailed_scores_to_file(computed_metrics, args, preds):
    output_path = Path(args.predictions).with_suffix(".eval.detailed.txt")

    metrics_dict = {metric.name: metric for metric in computed_metrics}
    output = ""

    if "Toxicity" in metrics_dict:
        metric = metrics_dict["Toxicity"]
        TOXICITY_CUTOFF = 0.5
        header = f"Potentially Toxic Responses (score >= {TOXICITY_CUTOFF})\n"
        output += "=" * len(header) + "\n" + header + "=" * len(header) + "\n"
        for idx, pred in enumerate(preds):
            score = metric.individual_scores[idx]
            if score >= TOXICITY_CUTOFF:
                output += f"{idx},{score:.3f},{pred}\n"

    if "GRUEN" in metrics_dict:
        metric = metrics_dict["GRUEN"]
        LQ_CUTOFF = 0.52
        header = f"Potentially Influent Responses (score <= {LQ_CUTOFF})\n"
        output += "=" * len(header) + "\n" + header + "=" * len(header) + "\n"
        for idx, pred in enumerate(preds):
            score = metric.individual_scores[idx]
            if score <= LQ_CUTOFF:
                output += f"{idx},{score:.3f},{pred}\n"

    if "BERTScore" in metrics_dict:
        metric = metrics_dict["BERTScore"]
        header = f"BERTScores\n"
        output += "=" * len(header) + "\n" + header + "=" * len(header) + "\n"
        for idx, pred in enumerate(preds):
            output += f"{idx+1},{metric.individual_scores[idx]:.3f},{pred}\n"

    with output_path.open("a") as f:
        f.write(output)

    print(f"Saved detailed scores to {output_path.absolute()}")


DEFAULT_METRICS = ["gruen", "bleu2", "bert-score", "toxicity", "dist1", "dist2", "ent4", "avg-len"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--predictions", type=Path)
    parser.add_argument("-r", "--references", type=Path, default="data/test.refs.txt")
    parser.add_argument("-i", "--inputs", type=Path, default="data/test.inputs.txt")
    parser.add_argument("-n", "--no_save", action="store_true")
    parser.add_argument(
        "-m",
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    main(args)
