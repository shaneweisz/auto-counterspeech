from pathlib import Path
from typing import List, Tuple
from datasets import DatasetDict, load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
import argparse
import json
import wandb

wandb.init(project="AutoCounterspeech")


def main():
    print(f"Loading model: {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    print(f"Loading tokenizer: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))

    print(f"Setting up training params using: {args.data_dir}")
    training_params = json.load(open((args.config_path)))
    training_args = TrainingArguments(**training_params)
    print(f"Training params: {training_params}")

    print(f"Loading dataset from: {args.data_dir}")
    dataset = load_dataset_dict(args.data_dir, splits=["train", "val"])

    print("Tokenizing dataset")
    tokenized_dataset = tokenize(dataset, tokenizer)

    print("Setup Trainer")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["val"],
        tokenizer=tokenizer,
    )

    print("Start training")
    trainer.train()

    print("Training finished. Saving best model.")
    trainer.save_model()


def load_dataset_dict(data_dir: Path, splits: List[str] = ["train", "val"]) -> DatasetDict:
    data_files = {split: str(data_dir / f"{split}.csv") for split in splits}
    dataset = load_dataset("csv", data_files=data_files)
    return dataset


def tokenize(dataset: DatasetDict, tokenizer: AutoTokenizer) -> DatasetDict:
    def tokenize_row(row):
        hs = row["hate_speech"]
        cs = row["counter_speech"]
        EOS = tokenizer.eos_token

        text = f"{hs}{EOS}{cs}{EOS}"
        tokenized_text = tokenizer(text)

        return tokenized_text

    return dataset.map(tokenize_row, remove_columns=dataset["train"].column_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    DIALO_GPT_MEDIUM = "microsoft/DialoGPT-medium"
    parser.add_argument(
        "-m",
        "--model_name_or_path",
        type=str,
        default=DIALO_GPT_MEDIUM,
    )

    parser.add_argument("-c", "--config", dest="config_path", default="config/train.config.json", type=Path)
    parser.add_argument("-d", "--data_dir", type=Path, default="data/splits/multitarget-conan")

    args = parser.parse_args()

    main()
