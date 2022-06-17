import torch
from transformers import LogitsProcessor


# HuggingFace's generate function does not yet support a `min_new_tokens`, so we need to add the functionality
# ourselves by adding a custom logits processor. Adapted from:
# https://huggingface.co/transformers/v4.1.1/_modules/transformers/generation_logits_process.html#MinLengthLogitsProcessor
class MinNewTokensLogitsProcessor(LogitsProcessor):
    r"""
    A [`LogitsProcessor`] enforcing a minimum response length by setting the `EOS` probability to 0 until
    `min_new_tokens` new tokens have been generated since `input_length`.
    """
    def __init__(self, min_new_tokens: int, eos_token_id: int, input_length: int):
        if not isinstance(min_new_tokens, int) or min_new_tokens < 0:
            raise ValueError(f"`min_new_tokens` has to be a positive integer, but is {min_new_tokens}")

        if not isinstance(eos_token_id, int) or eos_token_id < 0:
            raise ValueError(f"`eos_token_id` has to be a positive integer, but is {eos_token_id}")

        if not isinstance(input_length, int) or input_length < 0:
            raise ValueError(f"`input_length` has to be a positive integer, but is {input_length}")

        self.min_new_tokens = min_new_tokens
        self.eos_token_id = eos_token_id
        self.input_length = input_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if not hasattr(self, "input_length"):
            raise ValueError("`save_input_length` has to be called before `__call__`")

        total_length = input_ids.shape[-1]
        response_len = total_length - self.input_length

        if response_len < self.min_new_tokens:
            scores[:, self.eos_token_id] = -float("inf")

        return scores
