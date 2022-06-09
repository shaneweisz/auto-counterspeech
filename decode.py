from pathlib import Path
from typing import Any, Dict, List
from response_generation import ResponseGenerator
from util import update_config_from_string
import argparse
import json
import datetime


def main(args):
    print(f"Reading hate speech inputs from {args.inputs_path}")
    inputs = read_hate_speech_inputs(args.inputs_path)

    decoding_config = get_decoding_config(args.decoding_config_path, args.config_overrides)

    print(f"Loading pretrained model from {args.model_name_or_path}")
    model = ResponseGenerator(args.model_name_or_path, decoding_config)

    print(f"Decoding config: {decoding_config}")
    print("Generating responses to hate speech inputs")
    predictions = model.generate_responses(inputs, args.batch_size)

    print("Saving experiment")
    exp_path = save_experiment(predictions, decoding_config, args)
    print(f"Experiment saved to {exp_path}")


def read_hate_speech_inputs(file_path: Path) -> List[str]:
    return [line.rstrip("\n") for line in open(file_path)]


def get_decoding_config(decoding_config_path: Path, config_overrides: str) -> Dict[str, Any]:
    decoding_config = json.load(open(decoding_config_path))
    if "exponential_decay_length_penalty" in decoding_config:
        # the tuple had to be stored as a string since tuples can't be stored in JSON
        decoding_config["exponential_decay_length_penalty"] = eval(decoding_config["exponential_decay_length_penalty"])
    decoding_config = update_config_from_string(decoding_config, config_overrides)
    return decoding_config


def save_experiment(predictions, decoding_config, args, base_dir=Path("experiments")) -> Path:
    """Returns the path to the saved experiment"""
    if not args.experiment_name:
        experiment_name = generate_experiment_name(args.model_name_or_path)
    else:
        experiment_name = args.experiment_name

    experiment_dir = base_dir / experiment_name
    experiment_dir.mkdir(exist_ok=True)

    # Save predictions
    preds_output_path = experiment_dir / "predictions.txt"
    with open(preds_output_path, "w") as f:
        for pred in predictions:
            pred = pred.replace("\n", " ")  # don't allow newlines in predictions (only seems to arise from sampling)
            f.write(f"{pred}\n")

    # Save decoding config
    decoding_config_output_path = experiment_dir / "decode.config.json"
    with open(decoding_config_output_path, "w") as f:
        json.dump(decoding_config, f)

    # Save experiment metadata
    args_output_path = experiment_dir / "metadata.yaml"
    with open(args_output_path, "w") as f:
        NOW = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        f.write(f"Date: {NOW}\n")
        f.write("Args:\n")
        for key, value in vars(args).items():
            f.write(f"\t{key}: {value}\n")

    return experiment_dir


def generate_experiment_name(model_name_or_path: str) -> str:
    NOW_AS_STR = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_name = model_name_or_path.strip("/")
    model_name = model_name.split("/")[-1]
    return NOW_AS_STR + "_" + model_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", dest="model_name_or_path", type=str)
    parser.add_argument("-c", "--config", dest="decoding_config_path", type=Path)
    parser.add_argument("-i", "--inputs_path", type=Path, default="evaluation/test.inputs.txt")
    parser.add_argument("-b", "--batch_size", type=int, default=16)
    parser.add_argument("-e", "--experiment_name", type=str)
    parser.add_argument(
        "-o",
        "--config_overrides",
        type=str,
        default="",
        help="Override some default config settings. Example: num_beams=5,no_repeat_ngram_size=3",
    )

    args = parser.parse_args()
    main(args)
