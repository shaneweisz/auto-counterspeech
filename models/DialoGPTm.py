from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")


def interact_with_model():
    hs_input = input("Hate speech: ")
    while hs_input != "":
        input_ids = tokenizer.encode(
            hs_input + tokenizer.eos_token, return_tensors="pt"
        )
        input_and_response_ids = model.generate(
            input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id
        )
        output_ids = input_and_response_ids[:, input_ids.shape[-1] :][0]
        response = tokenizer.decode(output_ids, skip_special_tokens=True)
        print("Response: " + response)
        hs_input = input("Hate speech: ")


if __name__ == "__main__":
    interact_with_model()
