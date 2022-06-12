from typing import Any, Dict, List
from tqdm import tqdm
from colorama import Fore, Style
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
import torch
from .min_new_tokens import MinNewTokensLogitsProcessor


class ResponseGenerator:
    def __init__(self, pretrained_model_name_or_path: str, decoding_config: Dict[str, Any], seed=42, verbose=True):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        if "min_new_tokens" in decoding_config:  # add `min_new_tokens` functionality using our custom logits processor
            min_new_tokens_logits_processor = MinNewTokensLogitsProcessor(
                decoding_config["min_new_tokens"], self.tokenizer.eos_token_id
            )
            decoding_config["logits_processor"] = LogitsProcessorList([min_new_tokens_logits_processor])
            decoding_config.pop("min_new_tokens")
        self.decoding_config = decoding_config
        self.verbose = verbose

        torch.manual_seed(seed)

    def generate_responses(self, inputs: List[str], batch_size=1) -> List[str]:
        responses = []
        for i in tqdm(range(0, len(inputs), batch_size), disable=not self.verbose):
            batch_inputs = inputs[i : i + batch_size]
            batch_responses = self.generate_responses_for_batch(batch_inputs)
            responses.extend(batch_responses)
        return responses

    def generate_responses_for_batch(self, inputs: List[str]) -> str:
        inputs = [input_text + self.tokenizer.eos_token for input_text in inputs]

        self.tokenizer.padding_side = "left"
        tokenized_inputs = self.tokenizer(inputs, return_tensors="pt", padding=True).to(self.device)

        output_ids = self.model.generate(
            **tokenized_inputs, **self.decoding_config, pad_token_id=self.tokenizer.pad_token_id
        )
        input_len = tokenized_inputs["input_ids"].shape[-1]
        response_ids = output_ids[:, input_len:]
        responses = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        return responses

    def respond(self, input_text: str) -> str:
        """Respond to a single hate speech input."""
        return self.generate_responses([input_text])[0]

    def interact(self):
        prompt = Fore.RED + "Hate speech: " + Style.RESET_ALL
        input_text = input(prompt)
        while input_text != "":
            print(Fore.GREEN + "Response: " + Style.RESET_ALL, end="")
            response = self.respond(input_text)
            print(response)
            input_text = input(prompt)
