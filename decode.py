from pathlib import Path
from typing import Any, Dict
from response_generation import ResponseGenerator
from util import update_config_from_string
from util.file_io import read_list_from_file
import argparse
import json
import datetime


def main(args):
    print(f"Reading inputs from {args.inputs_path}")
    inputs = read_list_from_file(args.inputs_path)

    decoding_config = get_decoding_config(args.decoding_config_path, args.config_overrides)
    print(f"Decoding config: {decoding_config}")

    print(f"Loading pretrained model from {args.model_name_or_path}")
    model = ResponseGenerator(args.model_name_or_path, decoding_config.copy())

    print("Generating responses to inputs")
    predictions = model.generate_responses(inputs, args.batch_size)

    print("Saving decoding output")
    output_dir = save_decoding_output(predictions, decoding_config, args.output_dir)
    print(f"Decoding output saved to {output_dir}")


def get_decoding_config(decoding_config_path: Path, config_overrides: str) -> Dict[str, Any]:
    decoding_config = json.load(open(decoding_config_path))
    decoding_config = update_config_from_string(decoding_config, config_overrides)
    return decoding_config


def save_decoding_output(predictions, decoding_config, output_dir) -> Path:
    """Returns the path to the saved decoding output"""
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save predictions
    preds_output_path = output_dir / "predictions.txt"
    with open(preds_output_path, "w") as f:
        for pred in predictions:
            pred = pred.replace("\n", " ")  # don't allow newlines in predictions (only seems to arise from sampling)
            f.write(f"{pred}\n")

    # Save decoding config
    decoding_config_output_path = output_dir / "decode.config.json"
    with open(decoding_config_output_path, "w") as f:
        json.dump(decoding_config, f)

    # Save script metadata
    args_output_path = output_dir / "metadata.yaml"
    with open(args_output_path, "w") as f:
        NOW = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        f.write(f"Date: {NOW}\n")
        f.write("Args:\n")
        for key, value in vars(args).items():
            f.write(f"\t{key}: {value}\n")

    return output_dir


def generate_unique_name(model_name_or_path: str) -> str:
    NOW_AS_STR = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_name = model_name_or_path.strip("/")
    model_name = model_name.split("/")[-1]
    return NOW_AS_STR + "_" + model_name


DEFAULT_MODEL = "models/DialoGPT-finetuned-multiCONAN"
DEFAULT_CONFIG = "config/decode.config.json"
DEFAULT_INPUTS = "evaluation/test.inputs.txt"
DEFAULT_BATCH_SIZE = 16

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model", dest="model_name_or_path", default=DEFAULT_MODEL, type=str)
    parser.add_argument("-c", "--config", dest="decoding_config_path", default=DEFAULT_CONFIG, type=Path)
    parser.add_argument("-i", "--inputs_path", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("-b", "--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "-o", "--output_dir", type=Path, help="Path to directory where the decoding output should be saved."
    )
    parser.add_argument(
        "-co",
        "--config_overrides",
        type=str,
        default="",
        help="Override some default config settings. Example: 'num_beams=5,no_repeat_ngram_size=3'",
    )

    args = parser.parse_args()
    main(args)
