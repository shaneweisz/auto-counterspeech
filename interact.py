import argparse
from pathlib import Path
from response_generation import ResponseGenerator
from decode import get_decoding_config, DEFAULT_CONFIG, DEFAULT_MODEL


def main(args):
    print(f"Model: {args.model_name_or_path}")

    decoding_config = get_decoding_config(args.decoding_config_path, args.config_overrides)
    print(f"Decoding parameters: {decoding_config}")

    print("Loading counterspeech model...")
    model = ResponseGenerator(args.model_name_or_path, decoding_config)

    print("Model ready. Throw your worst at it, and see how it responds.")
    model.interact()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name_or_path", type=str, default=DEFAULT_MODEL)
    parser.add_argument("-c", "--config", dest="decoding_config_path", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "-o",
        "--config_overrides",
        type=str,
        default="",
        help="Override some default config settings. Example: num_beams=5;no_repeat_ngram_size=3",
    )
    args = parser.parse_args()
    main(args)
