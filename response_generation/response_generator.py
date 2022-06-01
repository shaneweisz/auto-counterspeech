from typing import Any, Dict, List
from tqdm import tqdm
from colorama import Fore, Style
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResponseGenerator:
    def __init__(
        self, pretrained_model_name_or_path: str, decoding_config: Dict[str, Any]
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path)
        self.model = self.model.to(device)
        self.decoding_config = decoding_config

    def generate_responses(self, inputs: List[str]) -> List[str]:
        responses = [self.generate_response(hs_input) for hs_input in tqdm(inputs)]
        return responses

    def generate_response(self, hs_input: str) -> str:
        input_with_eos_token = hs_input + self.tokenizer.eos_token
        input_ids = self.tokenizer.encode(input_with_eos_token, return_tensors="pt")
        input_ids = input_ids.to(device)

        input_ids_then_response_ids = self.model.generate(
            input_ids,
            pad_token_id=self.tokenizer.eos_token_id,
            **self.decoding_config,
        )

        response_ids = input_ids_then_response_ids[:, input_ids.shape[-1] :].squeeze()
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        return response

    def interact(self):
        prompt = Fore.RED + "Hate speech: " + Style.RESET_ALL
        hs_input = input(prompt)
        while hs_input != "":
            response = self.generate_response(hs_input)
            print(Fore.GREEN + "Response:" + Style.RESET_ALL, response)
            hs_input = input(prompt)
