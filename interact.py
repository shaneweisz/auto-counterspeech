import argparse
from pathlib import Path
from response_generation import ResponseGenerator
import json


def main(args):
    decoding_config = json.load(open(args.decoding_config_path))

    print(f"Model: {args.pretrained_model_name_or_path}")
    print(f"Decoding parameters: {decoding_config}")

    print("Loading counterspeech model...")
    model = ResponseGenerator(args.pretrained_model_name_or_path, decoding_config)

    print("Model ready. Throw your worst at it, and see how it responds.")
    model.interact()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--pretrained_model_name_or_path", type=str)
    parser.add_argument("-c", "--config", dest="decoding_config_path", type=Path)
    args = parser.parse_args()
    main(args)
