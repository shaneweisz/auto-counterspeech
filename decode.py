from pathlib import Path
from typing import List
from response_generation import ResponseGenerator
from util import update_config_from_string
import argparse
import json
import datetime


def main(args):
    print(f"Reading hate speech inputs from {args.inputs_path}")
    inputs = read_hate_speech_inputs(args.inputs_path)

    decoding_config = json.load(open(args.decoding_config_path))
    decoding_config = update_config_from_string(decoding_config, args.config_overrides)

    print(f"Loading pretrained model from {args.pretrained_model_name_or_path}")
    model = ResponseGenerator(args.pretrained_model_name_or_path, decoding_config)

    print(f"Decoding config: {decoding_config}")
    print("Generating responses to hate speech inputs")
    predictions = model.generate_responses(inputs)

    print("Saving experiment")
    exp_path = save_experiment(predictions, decoding_config, args)
    print(f"Experiment saved to {exp_path}")


def read_hate_speech_inputs(file_path: Path) -> List[str]:
    return [line.rstrip("\n") for line in open(file_path)]


def save_experiment(predictions, decoding_config, args, base_dir=Path("experiments")) -> Path:
    """Returns the path to the saved experiment"""
    experiment_name = generate_experiment_name(args.pretrained_model_name_or_path)

    experiment_dir = base_dir / experiment_name
    experiment_dir.mkdir(exist_ok=True)

    # Save predictions
    preds_output_path = experiment_dir / "predictions.txt"
    with open(preds_output_path, "w") as f:
        for pred in predictions:
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
    parser.add_argument("-m", "--pretrained_model_name_or_path", type=str)
    parser.add_argument("-c", "--config", dest="decoding_config_path", type=Path)
    parser.add_argument("-i", "--inputs_path", type=Path)
    parser.add_argument(
        "--config_overrides",
        type=str,
        default="",
        help="Override some default config settings. Example: num_beams=5,no_repeat_ngram_size=3",
    )

    args = parser.parse_args()
    main(args)
