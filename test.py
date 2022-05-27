from pathlib import Path
from typing import List
from evaluation.evaluator import Evaluator
from evaluate import load_metrics
from models import DialoGPT
import argparse
import wandb
import json

wandb.init(project="AutoCounterspeech", group="Multitarget-CONAN", job_type="dialoGPT")


def main(args):
    inputs = get_list_from_file(args.inputs_path)
    references = get_list_from_file(args.references_path)

    # config from config.json
    config = json.load(open(args.config_path))
    wandb.config.update(config)

    if args.model == "dialoGPT":
        model = DialoGPT(config)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    predictions = model.generate_responses(inputs)

    pred_table = wandb.Table(columns=["Input", "Model Response", "Gold Response"])
    for input, prediction, reference in zip(inputs, predictions, references):
        pred_table.add_data(input, prediction, reference)

    scores = evaluate(predictions, references, inputs, args.metrics)
    print(scores)
    scores_table = wandb.Table(columns=["Metric", "Score"])
    for metric, score in scores.items():
        scores_table.add_data(metric, score)

    wandb.log(
        {
            "test_predictions": pred_table,
            "scores": scores_table,
        }
    )


def get_list_from_file(file_path: Path) -> List[str]:
    return [line.rstrip("\n") for line in open(file_path)]


def evaluate(predictions, references, inputs, metrics):
    evaluator = Evaluator(predictions, references, inputs)
    metrics = load_metrics(metrics)
    scores = evaluator.evaluate(metrics, verbose=True)
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--inputs_path", type=Path, default="data/test_inputs.txt"
    )
    parser.add_argument(
        "-r", "--references_path", type=Path, default="data/test_references.txt"
    )
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("-c", "--config", dest="config_path", type=Path)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=[
            "gruen",
            "bert-score",
            "bleu1",
            "bleu2",
            "rouge1",
            "rouge2",
            "dist1",
            "dist2",
            "ent1",
            "ent2",
        ],
    )

    args = parser.parse_args()
    main(args)
