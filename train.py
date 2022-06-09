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
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Setting up training params using: {args.data_dir}")
    training_params = json.load(open((args.config_path)))
    training_args = TrainingArguments(**training_params)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    print(f"Loading and tokenizing train and val datasets stored in: {args.data_dir}")
    train_dataset, val_dataset = prepare_data_for_training(args.data_dir, tokenizer)

    print("Setup Trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    print("Start training")
    trainer.train()

    print("Training finished. Saving best model.")
    trainer.save_model()


def prepare_data_for_training(data_dir: Path, tokenizer: AutoTokenizer) -> Tuple[Dataset, Dataset]:
    raw_datasets = load_raw_datasets(data_dir)
    tokenized_datasets = tokenize(raw_datasets, tokenizer)
    train_dataset, val_dataset = tokenized_datasets["train"], tokenized_datasets["val"]
    return train_dataset, val_dataset


def load_raw_datasets(data_dir: Path, splits: List[str] = ["train", "val"]) -> DatasetDict:
    data_files = {split: str(data_dir / f"{split}.csv") for split in splits}
    raw_datasets = load_dataset("csv", data_files=data_files)
    return raw_datasets


def tokenize(raw_datasets: DatasetDict, tokenizer: AutoTokenizer) -> DatasetDict:
    def tokenize_row(row):
        hs = row["hate_speech"]
        cs = row["counter_speech"]
        EOS = tokenizer.eos_token

        text = f"{hs}{EOS}{cs}{EOS}"
        tokenized_text = tokenizer(text)

        return tokenized_text

    return raw_datasets.map(tokenize_row, remove_columns=raw_datasets["train"].column_names)


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
