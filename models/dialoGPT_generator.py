from typing import List
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


class DialoGPTmGenerator:
    def __init__(self, num_beams=1):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
        self.num_beams = num_beams

    def generate(self, hs_inputs: List[str], verbose: bool) -> List[str]:
        cs_predictions = []
        for hs_input in tqdm(hs_inputs):
            input_ids = self.tokenizer.encode(
                hs_input + self.tokenizer.eos_token, return_tensors="pt"
            )
            input_and_response_ids = self.model.generate(
                input_ids,
                max_length=100,
                num_beams=self.num_beams,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            response_ids = input_and_response_ids[:, input_ids.shape[-1] :].squeeze()
            response_text = self.tokenizer.decode(
                response_ids, skip_special_tokens=True
            )
            cs_predictions.append(response_text)
            if verbose:
                print(f"Hate speech: {hs_input}")
                print(f"Response: {response_text}")
        return cs_predictions

    def interact_with_model(self):
        hs_input = input("Hate speech: ")
        while hs_input != "":
            input_ids = self.tokenizer.encode(
                hs_input + self.tokenizer.eos_token, return_tensors="pt"
            )
            num_beams = 5
            input_and_response_ids = self.model.generate(
                input_ids,
                max_length=100,
                # do_sample=True,
                # top_p=0.92,
                # top_k=50,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                no_repeat_ngram_size=5,  # ensure no 5-grams are repeated
                early_stopping=True,  # finish generation when all beams have reached an <eos> token
                pad_token_id=self.tokenizer.eos_token_id,
            )
            output_ids_all_beams = input_and_response_ids[:, input_ids.shape[-1] :]

            print("Response(s): ")
            for output_ids in output_ids_all_beams:
                response = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                print(response)
            hs_input = input("Hate speech: ")


if __name__ == "__main__":
    DialoGPTmGenerator().interact_with_model()
