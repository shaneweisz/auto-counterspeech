import argparse
from pathlib import Path
from response_generation import ResponseGenerator
from util import update_config_from_string
import json


def main(args):
    print(f"Model: {args.pretrained_model_name_or_path}")

    decoding_config = json.load(open(args.decoding_config_path))
    decoding_config = update_config_from_string(decoding_config, args.config_overrides)
    print(f"Decoding parameters: {decoding_config}")

    print("Loading counterspeech model...")
    model = ResponseGenerator(args.pretrained_model_name_or_path, decoding_config)

    print("Model ready. Throw your worst at it, and see how it responds.")
    model.interact()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--pretrained_model_name_or_path", type=str)
    parser.add_argument("-c", "--config", dest="decoding_config_path", type=Path)
    parser.add_argument(
        "-o",
        "--config_overrides",
        type=str,
        default="",
        help="Override some default config settings. Example: num_beams=5;no_repeat_ngram_size=3",
    )
    args = parser.parse_args()
    main(args)
