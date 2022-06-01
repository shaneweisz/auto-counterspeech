from pathlib import Path
from typing import List
from response_generation import ResponseGenerator
import argparse
import json


def main(args):
    inputs = get_lines_from_file(args.inputs_path)
    response_generator = load_response_generator(
        args.pretrained_model_name_or_path, args.decoding_config_path
    )
    predictions = response_generator.generate_responses(inputs)
    write_lines_to_file(predictions, args.output_path)


def get_lines_from_file(file_path: Path) -> List[str]:
    return [line.rstrip("\n") for line in open(file_path)]


def write_lines_to_file(lines: List[str], path):
    with open(path, "w") as f:
        for line in lines:
            f.write(f"{line}\n")


def load_response_generator(pretrained_model_name_or_path, config_path):
    decoding_config = json.load(open(config_path))
    response_generator = ResponseGenerator(
        pretrained_model_name_or_path, decoding_config
    )
    return response_generator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--pretrained_model_name_or_path", type=str)
    parser.add_argument("-c", "--config", dest="decoding_config_path", type=Path)
    parser.add_argument("-i", "--inputs_path", type=Path)
    parser.add_argument("-o", "--output_path", type=Path)
    args = parser.parse_args()
    main(args)
