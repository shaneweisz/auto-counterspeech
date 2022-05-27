from pathlib import Path
from models import DialoGPTmGenerator
import argparse


def main(args):
    inputs = get_inputs_from_file(args.input_path)
    if args.model == "dialoGPTm":
        model = DialoGPTmGenerator()
    else:
        raise ValueError(f"Unknown model: {args.model}")
    predictions = model.generate(inputs, verbose=True)
    write_to_file(predictions, model, args)


def get_inputs_from_file(file_path):
    return [line.rstrip("\n") for line in open(file_path)]


def write_to_file(predictions, model, args):
    output_path = Path(args.output_dir) / (
        model.__class__.__name__ + "_predictions.txt"
    )
    with open(output_path, "w") as f:
        for prediction in predictions:
            f.write(prediction + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=Path, default="data/test_inputs.txt")
    parser.add_argument("-o", "--output_dir", type=Path, default="predictions")
    parser.add_argument("-m", "--model", type=str, default="dialoGPTm")
    args = parser.parse_args()
    main(args)
