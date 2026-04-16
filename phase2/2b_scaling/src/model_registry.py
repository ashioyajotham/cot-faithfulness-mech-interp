"""
HookedTransformer model loader for Qwen2.5-Math-7B and Gemma 3 12B IT.

Centralises model loading, dtype handling, and configuration so that
the patching and probing code remains model-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from transformer_lens import HookedTransformer


@dataclass
class ModelSpec:
    """Configuration for a supported model."""

    hf_name: str
    n_layers: int
    n_heads: int
    d_model: int
    dtype: str = "float16"
    fold_ln: bool = False
    center_writing_weights: bool = False
    center_unembed: bool = False


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "gpt2": ModelSpec(
        hf_name="gpt2",
        n_layers=12,
        n_heads=12,
        d_model=768,
        dtype="float32",
    ),
    "qwen25-math-7b": ModelSpec(
        hf_name="Qwen/Qwen2.5-Math-7B-Instruct",
        n_layers=32,
        n_heads=28,
        d_model=3584,
    ),
    "gemma3-12b-it": ModelSpec(
        hf_name="google/gemma-3-12b-it",
        n_layers=48,
        n_heads=8,
        d_model=3072,
    ),
}


def load_model(
    model_key: str,
    device: str = "auto",
    dtype_override: Optional[str] = None,
) -> HookedTransformer:
    """Load a HookedTransformer by registry key or HuggingFace name.

    If *model_key* is not in the registry it is treated as a raw HF name.
    """
    spec = MODEL_REGISTRY.get(model_key)

    if spec is None:
        hf_name = model_key
        dtype = dtype_override or "float16"
        fold_ln = False
        center_writing_weights = False
        center_unembed = False
    else:
        hf_name = spec.hf_name
        dtype = dtype_override or spec.dtype
        fold_ln = spec.fold_ln
        center_writing_weights = spec.center_writing_weights
        center_unembed = spec.center_unembed

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch_dtype = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]

    model = HookedTransformer.from_pretrained(
        hf_name,
        device=device,
        dtype=torch_dtype,
        fold_ln=fold_ln,
        center_writing_weights=center_writing_weights,
        center_unembed=center_unembed,
    )
    model.eval()
    return model
