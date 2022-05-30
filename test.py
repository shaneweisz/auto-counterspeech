from pathlib import Path
from typing import List
from models import DialoGPT
import argparse
import json


def main(args):
    inputs = get_lines_from_file(args.inputs_path)
    model = get_model(args.model_name, args.config_path)
    predictions = model.generate_responses(inputs)
    write_lines_to_file(predictions, args.output_path)


def get_lines_from_file(file_path: Path) -> List[str]:
    return [line.rstrip("\n") for line in open(file_path)]


def write_lines_to_file(lines: List[str], path):
    with open(path, "w") as f:
        for line in lines:
            f.write(f"{line}\n")


def get_model(model_name, config_path):
    config = json.load(open(config_path))
    if model_name == "dialoGPT":
        return DialoGPT(config)
    else:
        raise ValueError(f"Unknown model: {model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--inputs_path", type=Path, default="data/test_inputs.txt"
    )
    parser.add_argument(
        "-o", "--output_path", type=Path, default="predictions/predictions.txt"
    )
    parser.add_argument("-m", "--model", dest="model_name", type=str)
    parser.add_argument("-c", "--config", dest="config_path", type=Path)

    args = parser.parse_args()
    main(args)
