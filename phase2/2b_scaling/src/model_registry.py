"""
HookedTransformer model loader for Phase 2B models.

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
    "qwen25-math-1.5b": ModelSpec(
        hf_name="Qwen/Qwen2.5-1.5B-Instruct",
        n_layers=28,
        n_heads=12,
        d_model=1536,
    ),
    "qwen25-math-7b": ModelSpec(
        hf_name="Qwen/Qwen2.5-7B-Instruct",
        n_layers=28,
        n_heads=28,
        d_model=3584,
    ),
    "gemma2-2b": ModelSpec(
        hf_name="google/gemma-2-2b-it",
        n_layers=26,
        n_heads=8,
        d_model=2304,
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

    # For large models (d_model > 2048 ~= 7B+), load and process weights
    # on CPU first, then move to GPU.  TransformerLens's fold_value_biases
    # clones the entire state dict, which doubles VRAM and OOMs on A100-40GB.
    is_large = spec is not None and spec.d_model > 2048
    load_device = "cpu" if (is_large and device == "cuda") else device

    extra_kwargs = {}
    if not is_large and (device == "cuda"):
        extra_kwargs["device_map"] = "auto"

    model = HookedTransformer.from_pretrained(
        hf_name,
        device=load_device,
        dtype=torch_dtype,
        fold_ln=fold_ln,
        center_writing_weights=center_writing_weights,
        center_unembed=center_unembed,
        low_cpu_mem_usage=True,
        **extra_kwargs
    )

    # Move to GPU after weight processing is complete
    if is_large and device == "cuda":
        print(f"  Moving model to {device}...")
        model = model.to(device)

    model.eval()
    return model
