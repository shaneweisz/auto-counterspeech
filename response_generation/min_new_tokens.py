import torch
from transformers import LogitsProcessor


# HuggingFace's generate function does not yet support a `min_new_tokens`, so we need to add the functionality
# ourselves by adding a custom logits processor. Adapted from:
# https://huggingface.co/transformers/v4.1.1/_modules/transformers/generation_logits_process.html#MinLengthLogitsProcessor
class MinNewTokensLogitsProcessor(LogitsProcessor):
    r"""
    A [`LogitsProcessor`] enforcing a minimum response length by setting the EOS probability to 0 until
    `min_new_tokens` new tokens have been generated.

    Args:
        min_new_tokens (`int`):
            The minimum number of new tokens for which the score of `eos_token_id` is set to `-float("Inf")`.
        eos_token_id (`int`):
            The id of the *end-of-sequence* token.
    """

    def __init__(self, min_new_tokens: int, eos_token_id: int):
        if not isinstance(min_new_tokens, int) or min_new_tokens < 0:
            raise ValueError(f"`min_new_tokens` has to be a positive integer, but is {min_new_tokens}")

        if not isinstance(eos_token_id, int) or eos_token_id < 0:
            raise ValueError(f"`eos_token_id` has to be a positive integer, but is {eos_token_id}")

        self.min_new_tokens = min_new_tokens
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # input length is the index of the first EOS token of the first input in the batch (assuming left padding)
        input_length = input_ids[0].eq(self.eos_token_id).nonzero()[0].item()
        total_length = input_ids.shape[-1]
        response_len = total_length - input_length

        if response_len < self.min_new_tokens:
            scores[:, self.eos_token_id] = -float("inf")

        return scores
