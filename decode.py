from pathlib import Path
from typing import List
from response_generation import ResponseGenerator
import argparse
import json
import datetime


def main(args):
    print(f"Loading inputs from {args.inputs_path}")
    inputs = read_inputs_from_file(args.inputs_path)
    print(f"Loading model from {args.pretrained_model_name_or_path}")
    model = load_response_generator(
        args.pretrained_model_name_or_path, args.decoding_config_path
    )

    print("Generating predictions")
    predictions = model.generate_responses(inputs)

    print(f"Saving predictions to {args.output_path}")
    save_experiment(predictions, args)


def read_inputs_from_file(file_path: Path) -> List[str]:
    return [line.rstrip("\n") for line in open(file_path)]


def load_response_generator(pretrained_model_name_or_path, config_path):
    decoding_config = json.load(open(config_path))
    model = ResponseGenerator(pretrained_model_name_or_path, decoding_config)
    return model


def save_experiment(predictions, args, base_dir=Path("experiments")):
    experiment_name = generate_experiment_name(args)

    experiment_dir = base_dir / experiment_name
    experiment_dir.mkdir(exist_ok=True)

    preds_output_path = experiment_dir / "predictions.txt"
    save_predictions(predictions, preds_output_path)

    decoding_config_output_path = experiment_dir / "decode.config.json"
    save_decoding_config(args.decoding_config_path, decoding_config_output_path)

    args_output_path = experiment_dir / "args.yaml"
    save_args(args, args_output_path)


def generate_experiment_name(args):
    NOW_AS_STR = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_name = args.pretrained_model_name_or_path.strip("/")
    model_name = model_name.replace("models/", "")
    model_name = model_name.replace("/", " ")
    return NOW_AS_STR + " " + model_name


def save_predictions(predictions, output_path):
    with open(output_path, "w") as f:
        for pred in predictions:
            f.write(f"{pred}\n")


def save_decoding_config(decoding_config_path, output_path):
    decoding_config = json.load(open(decoding_config_path))
    with open(output_path, "w") as f:
        json.dump(decoding_config, f)


def save_args(args, output_path):
    with open(output_path, "w") as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--pretrained_model_name_or_path", type=str)
    parser.add_argument("-c", "--config", dest="decoding_config_path", type=Path)
    parser.add_argument("-i", "--inputs_path", type=Path)
    parser.add_argument("-o", "--output_path", type=Path)
    args = parser.parse_args()
    main(args)
