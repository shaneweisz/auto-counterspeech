from typing import Any, Dict, List
from tqdm import tqdm
from colorama import Fore, Style
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
import torch
from .min_new_tokens import MinNewTokensLogitsProcessor


class ResponseGenerator:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, pretrained_model_name_or_path: str, decoding_config: Dict[str, Any], seed=42, verbose=True):
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        if "pad_token" not in self.tokenizer.special_tokens_map:
            self.tokenizer.pad_token = self.tokenizer.eos_token  # A pad token needs to be set for batch decoding
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
        input_len = tokenized_inputs["input_ids"].shape[-1]

        params_for_generate = self._params_for_generate(input_len)
        output_ids = self.model.generate(
            **tokenized_inputs, **params_for_generate, pad_token_id=self.tokenizer.pad_token_id
        )

        response_ids = output_ids[:, input_len:]
        responses = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)

        return responses

    def _params_for_generate(self, input_length: int) -> Dict[str, Any]:
        params_for_generate = self.decoding_config.copy()

        if "min_new_tokens" in params_for_generate and params_for_generate["min_new_tokens"] is not None:
            # the HuggingFace `generate` function accepts a `logits_processor` argument, not a `min_new_tokens`,
            # so we replace `min_new_tokens` from the `decoding_config` with our custom logits processor
            # that enforces a minimum response length
            min_new_tokens = params_for_generate["min_new_tokens"]
            min_new_tokens_logits_processor = MinNewTokensLogitsProcessor(
                min_new_tokens, self.tokenizer.eos_token_id, input_length
            )
            params_for_generate["logits_processor"] = LogitsProcessorList([min_new_tokens_logits_processor])
            params_for_generate.pop("min_new_tokens")

        return params_for_generate

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
