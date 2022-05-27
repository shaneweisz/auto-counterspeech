from typing import List
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


class DialoGPT:
    def __init__(self, config):
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        self.model = AutoModelForCausalLM.from_pretrained(config["model_name"])
        self.num_beams = config["num_beams"]
        self.max_length = config["max_length"]

    def generate_responses(self, hs_inputs: List[str]) -> List[str]:
        responses = [self.generate_response(hs_input) for hs_input in tqdm(hs_inputs)]
        return responses

    def generate_response(self, hs_input: str) -> str:
        input_with_eos_token = hs_input + self.tokenizer.eos_token
        input_ids = self.tokenizer.encode(input_with_eos_token, return_tensors="pt")

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
        hs_input = input("Hate speech: ")
        while hs_input != "":
            response = self.generate_response(hs_input)
            print("Response: ", response)
            hs_input = input("Hate speech: ")


if __name__ == "__main__":
    DialoGPT
    ().interact_with_model()
