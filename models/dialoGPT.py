from typing import List
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DialoGPT:
    def __init__(self, config):
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        self.model = AutoModelForCausalLM.from_pretrained(config["model_name"]).to(
            device
        )
        self.num_beams = config["num_beams"]
        self.max_length = config["max_length"]

    def generate_responses(self, hs_inputs: List[str]) -> List[str]:
        responses = [self.generate_response(hs_input) for hs_input in tqdm(hs_inputs)]
        return responses

    def generate_response(self, hs_input: str) -> str:
        input_with_eos_token = hs_input + self.tokenizer.eos_token
        input_ids = self.tokenizer.encode(input_with_eos_token, return_tensors="pt").to(
            device
        )

        input_ids_then_response_ids = self.model.generate(
            input_ids,
            max_length=self.max_length,
            num_beams=self.num_beams,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        response_ids = input_ids_then_response_ids[:, input_ids.shape[-1] :].squeeze()
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        return response

    def interact_with_model(self):
        class col:
            BLUE = "\033[94m"
            GREEN = "\033[92m"
            END = "\033[0m"

        def blue(s):
            return col.BLUE + s + col.END

        def green(s):
            return col.GREEN + s + col.END

        prompt = blue("Hate speech: ")
        hs_input = input(prompt)
        while hs_input != "":
            response = self.generate_response(hs_input)
            print(green("Response: "), response)
            hs_input = input(prompt)


if __name__ == "__main__":
    config = {
        "model_name": "checkpoints/finetuned-dialogpt/checkpoint-2000",
        "num_beams": 1,
        "max_length": 100,
    }
    DialoGPT(config).interact_with_model()
