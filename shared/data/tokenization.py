"""
Verified tokenization utilities that handle space-prefix edge cases.

GPT-2 and many HuggingFace tokenizers prepend a space (Ġ) to the first
sub-token of a word when it follows a space in the original text.  The
answer token " 68" is different from "68".  This module ensures the
caller always gets the right token id.
"""

from __future__ import annotations

import torch
from transformer_lens import HookedTransformer


def verified_tokenize(
    model: HookedTransformer,
    text: str,
    prepend_bos: bool = True,
) -> torch.Tensor:
    """Tokenize *text* and return the token tensor."""
    return model.to_tokens(text, prepend_bos=prepend_bos)


def answer_token_id(
    model: HookedTransformer,
    answer_str: str,
) -> int:
    """Return the single-token id for an answer string like " 68".

    Raises ``ValueError`` if the answer tokenizes to more than one token
    (indicating the model's vocabulary cannot represent it atomically).
    """
    ids = model.to_tokens(answer_str, prepend_bos=False).squeeze()
    if ids.dim() == 0:
        return ids.item()
    if ids.shape[0] == 1:
        return ids[0].item()
    raise ValueError(
        f"Answer '{answer_str}' tokenizes to {ids.shape[0]} tokens "
        f"({model.to_str_tokens(answer_str)}); expected 1."
    )
