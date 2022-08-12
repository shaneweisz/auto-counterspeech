from pathlib import Path
from typing import Any, Dict, List, Union
from datasets import DatasetDict, load_dataset
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


wandb.init(project="AutoCounterspeech")  # for visualizing training, see: https://wandb.ai/quickstart/hugging-face


def main():
    print(f"Loading model: {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model = freeze_first_n_layers(model, args.freeze_first_n)

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
    data_collator = DataCollatorForDialog(tokenizer=tokenizer, mlm=False)
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


def freeze_first_n_layers(model, n):
    for i in range(n):
        gpt2_block = model.transformer.h[i]
        for param in gpt2_block.parameters():
            param.requires_grad = False
    return model


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


class DataCollatorForDialog(DataCollatorForLanguageModeling):
    """Extend the DataCollatorForLanguageModeling to overwrite the `torch_call` function, so that language modelling
    training is based on next-token prediction of the response only (not the context too).
    This is done by setting the training label to -100 for all tokens from the context."""

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)
        labels = batch["labels"]

        # Set training label to -100 for all tokens left of the 2nd last EOS token (marking the end of the context)
        for i in range(len(labels)):
            index_of_eos_at_end_of_context = (labels[i] == self.tokenizer.eos_token_id).nonzero()[-2].item()
            labels[i, : index_of_eos_at_end_of_context + 1] = -100

        return batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    DIALO_GPT_MEDIUM = "microsoft/DialoGPT-medium"
    parser.add_argument("-m", "--model_name_or_path", type=str, default=DIALO_GPT_MEDIUM)
    parser.add_argument("-c", "--config", dest="config_path", default="config/train.config.mc.json", type=Path)
    parser.add_argument("-d", "--data_dir", type=Path, default="data/splits/multitarget-conan")
    parser.add_argument("-f", "--freeze_first_n", type=int, default=0)

    args = parser.parse_args()

    main()
