import argparse
from pathlib import Path
from typing import List
from metrics.cli import load_metrics
from metrics import Metric
from util.file_io import read_list_from_file, write_list_to_file, write_text_to_file
from util.print_utils import horizontal_line


def main(args):
    print(f"Loading metrics: {args.metric_names}")
    metrics = load_metrics(args.metric_names)

    print(f"Extracting predictions from: {args.predictions_path}")
    predictions = read_list_from_file(args.predictions_path)

    print(f"Extracting references from: {args.references_path}")
    references = read_list_from_file(args.references_path)
    assert len(references) == len(predictions)

    print(f"Extracting inputs from: {args.inputs_path}")
    inputs = read_list_from_file(args.inputs_path)
    assert len(inputs) == len(predictions)

    print("Computing scores:")
    for metric in metrics:
        print(metric.name + "...")
        score = metric.compute_score(predictions, references=references, inputs=inputs, verbose=args.verbose)
        print(f"Score: {score:.3f}")

    if not args.no_save:
        print("Saving scores:")

        scores_output_path = Path(args.predictions_path).with_suffix(".scores.txt")
        write_list_to_file(scores_output_path, [str(metric) for metric in metrics], append=True)
        print(f"Saved aggregate scores to: {scores_output_path.absolute()}")

        if any(metric.individual_scores is not None for metric in metrics):
            detailed_scores_output_path = Path(args.predictions_path).with_suffix(".scores.detailed.csv")
            content = individual_scores_output(metrics, predictions)
            write_text_to_file(detailed_scores_output_path, content, append=True)
            print(f"Saved detailed scores to: {detailed_scores_output_path.absolute()}")

            flagged_responses_output_path = Path(args.predictions_path).with_suffix(".scores.flagged.txt")
            content = responses_to_flag(metrics, predictions)
            write_text_to_file(flagged_responses_output_path, content, append=True)
            print(f"Saved flagged responses to: {flagged_responses_output_path.absolute()}")


def responses_to_flag(computed_metrics: List[Metric], predictions: List[str]):
    flaggable_metrics = [
        ("Toxicity", 0.75, ">="),
        ("GRUEN", 0.5, "<="),
        ("Grammaticality", 0.75, "<="),
        ("BERTScore", 0.5, ">="),
        ("BERTScore", -0.05, "<="),
    ]

    output = ""
    for metric_name, cutoff, ineq_sign in flaggable_metrics:
        if metric_name in [metric.name for metric in computed_metrics]:
            metric = [metric for metric in computed_metrics if metric.name == metric_name][0]
            output += filter_responses_beyond_cutoff(metric, predictions, cutoff, ineq_sign)

    return output


def filter_responses_beyond_cutoff(metric: Metric, predictions: List[str], cutoff: float, ineq_sign=">="):
    header = f"Responses with {metric.name} {ineq_sign} {cutoff}:"
    output = horizontal_line(len(header)) + "\n"
    output += header + "\n"
    output += horizontal_line(len(header)) + "\n"

    no_filtered_responses = True
    for i, prediction in enumerate(predictions):
        score = metric.individual_scores[i]
        if eval(f"score{ineq_sign}{cutoff}"):
            output += f"{i+1},{score:.3f},{prediction}\n"
            no_filtered_responses = False

    if no_filtered_responses:
        output += "None\n"

    return output


def individual_scores_output(computed_metrics: List[Metric], predictions: List[str]) -> str:
    metrics_with_individual_scores = [metric for metric in computed_metrics if metric.individual_scores is not None]

    header = "Id," + ",".join([f"{metric.name}" for metric in metrics_with_individual_scores]) + ",Response"
    output = header + "\n"

    for idx, prediction in enumerate(predictions):
        line = f"{idx+1}"
        for metric in metrics_with_individual_scores:
            line += f",{metric.individual_scores[idx]:.3f}"
        line += f',"{prediction}"'

        output += line + "\n"

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-p",
        "--predictions",
        type=Path,
        dest="predictions_path",
        required=True,
        help="Path to file containing predicted responses for each input that should be scored",
    )
    parser.add_argument(
        "-r",
        "--references",
        type=Path,
        dest="references_path",
        default="evaluation/test.refs.txt",
        help="Path to file containing gold-standard reference responses for each input",
    )
    parser.add_argument(
        "-i",
        "--inputs",
        type=Path,
        dest="inputs_path",
        default="evaluation/test.inputs.txt",
        help="Path to file containing hate speech inputs that the predictions and references correspond to",
    )

    DEFAULT_METRICS = [
        "grammaticality",
        "toxicity",
        "bert-score",
        "bleu1",
        "bleu2",
        "bleu4",
        "rouge1",
        "rouge2",
        "rougeL",
        "dist1",
        "dist2",
        "ent4",
        "avg-len",
    ]
    parser.add_argument(
        "-m",
        "--metrics",
        dest="metric_names",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Metrics to compute scores for. Defaults to: {}".format(", ".join(DEFAULT_METRICS)),
    )

    parser.add_argument(
        "-n", "--no_save", action="store_true", help="Do not save scores to file. Print to console only."
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="More verbose output when computing metrics")

    args = parser.parse_args()
    main(args)
