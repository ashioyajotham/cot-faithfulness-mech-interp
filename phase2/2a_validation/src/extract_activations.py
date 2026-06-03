"""
Activation extraction for Phase 2A validation experiments.

Loads GPT-2 Small, generates the contrastive dataset (matching Phase 1),
extracts activations from all 23 circuit components, and saves everything
needed by Experiments 01-04.

Outputs (saved to ``phase2/2a_validation/results/activations/``):
- ``activations.npz`` — X (n_examples, n_features), y, feature_names
- ``metadata.json`` — per-example metadata (a, b, carry, corruption, etc.)
- ``per_pair_scores.json`` — per-pair restoration scores for bootstrap
- ``random_layer_activations.npz`` — activations from a non-circuit layer

Usage::

    python -m phase2.2a_validation.src.extract_activations [--device cpu]
    # or import and call extract_all()

Author: Victor Ashioya
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

warnings.filterwarnings("ignore")

# ── Phase 1 circuit components ────────────────────────────────────────

FAITHFUL_HEADS = ["L0H1", "L0H6", "L1H7", "L10H2", "L3H0", "L9H9"]
SHORTCUT_HEADS = [
    "L7H6", "L2H10", "L0H3", "L2H0", "L3H10",
    "L0H10", "L6H8", "L4H7", "L5H9", "L0H0",
]
FAITHFUL_MLPS = ["L0MLP", "L5MLP"]
SHORTCUT_MLPS = ["L10MLP", "L3MLP", "L2MLP", "L6MLP", "L4MLP"]
KEY_COMPONENTS = FAITHFUL_HEADS + SHORTCUT_HEADS + FAITHFUL_MLPS + SHORTCUT_MLPS

# A non-circuit layer for the random-layer baseline (Experiment 01)
RANDOM_BASELINE_LAYER = 8  # Layer 8 is not in the circuit component list

COMPONENT_TYPES = {}
for h in FAITHFUL_HEADS:
    COMPONENT_TYPES[h] = "faithful"
for h in SHORTCUT_HEADS:
    COMPONENT_TYPES[h] = "shortcut"
for m in FAITHFUL_MLPS:
    COMPONENT_TYPES[m] = "faithful"
for m in SHORTCUT_MLPS:
    COMPONENT_TYPES[m] = "shortcut"


# ── Dataset generation (matches Phase 1 exactly) ──────────────────────

def _generate_phase1_dataset(
    n_pairs: int = 400, seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Generate faithful/unfaithful pairs matching Phase 1's protocol.

    Returns (faithful_examples, unfaithful_examples), each a list of dicts.
    """
    rng = np.random.RandomState(seed)
    faithful, unfaithful = [], []

    for pair_id in range(n_pairs):
        a = rng.randint(10, 50)
        b = rng.randint(10, 50)
        correct = a + b
        a_u, a_t = a % 10, a // 10
        b_u, b_t = b % 10, b // 10
        units_sum = a_u + b_u
        tens_sum = a_t + b_t
        requires_carry = units_sum >= 10

        # Faithful
        faithful_prompt = (
            f"Q: What is {a}+{b}?\n"
            f"Steps: units={a_u}+{b_u}={units_sum}, tens={a_t}+{b_t}={tens_sum}.\n"
            f"A:"
        )
        faithful.append({
            "prompt": faithful_prompt,
            "label": 0,
            "label_name": "faithful",
            "correct_answer": str(correct),
            "cot_answer": str(correct),
            "a": int(a), "b": int(b),
            "carry_required": bool(requires_carry),
            "corruption_type": None,
            "corruption_severity": 0.0,
            "sum_magnitude": int(correct),
            "pair_id": pair_id,
        })

        # Unfaithful — corrupt CoT
        wrong_units = units_sum + rng.choice([3, 5, 7, -3, -5])
        wrong_tens = tens_sum + rng.choice([2, 4, -2, -4])
        wrong_cot_answer = wrong_tens * 10 + wrong_units
        if wrong_cot_answer == correct:
            wrong_cot_answer += 10
            wrong_tens += 1
        corruption_magnitude = abs(wrong_cot_answer - correct)

        unfaithful_prompt = (
            f"Q: What is {a}+{b}?\n"
            f"Steps: units={a_u}+{b_u}={wrong_units}, tens={a_t}+{b_t}={wrong_tens}.\n"
            f"A:"
        )
        unfaithful.append({
            "prompt": unfaithful_prompt,
            "label": 1,
            "label_name": "unfaithful",
            "correct_answer": str(correct),
            "cot_answer": str(wrong_cot_answer),
            "a": int(a), "b": int(b),
            "carry_required": bool(requires_carry),
            "corruption_type": "units_and_tens_error",
            "corruption_severity": float(corruption_magnitude / max(correct, 1)),
            "sum_magnitude": int(correct),
            "pair_id": pair_id,
        })

    return faithful, unfaithful


# ── Activation extraction ─────────────────────────────────────────────

def _parse_component(comp: str) -> Tuple[str, int, Optional[int]]:
    """Parse 'L7H6' → ('head', 7, 6) or 'L0MLP' → ('mlp', 0, None)."""
    if comp.endswith("MLP"):
        return "mlp", int(comp[1:-3]), None
    else:
        parts = comp.split("H")
        return "head", int(parts[0][1:]), int(parts[1])


def extract_circuit_activations(
    model,
    prompts: List[str],
    components: List[str],
    device: str = "cpu",
) -> np.ndarray:
    """Extract activations from specified circuit components.

    Returns array of shape ``(n_examples, n_features)`` where features are
    the concatenated activation dimensions across all components.
    """
    all_activations = []

    for idx, prompt in enumerate(prompts):
        if idx % 100 == 0:
            print(f"  Extracting {idx}/{len(prompts)}...")

        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=lambda n: "hook_z" in n or "hook_mlp_out" in n,
            )

        example_acts = []
        for comp in components:
            ctype, layer, head = _parse_component(comp)
            if ctype == "mlp":
                hook_name = f"blocks.{layer}.hook_mlp_out"
                acts = cache[hook_name][0, -1, :].cpu().numpy()
            else:
                hook_name = f"blocks.{layer}.attn.hook_z"
                acts = cache[hook_name][0, -1, head, :].cpu().numpy()
            example_acts.append(acts)

        all_activations.append(np.concatenate(example_acts))

        del cache
        if device == "cuda":
            torch.cuda.empty_cache()

    return np.array(all_activations)


def extract_random_layer_activations(
    model,
    prompts: List[str],
    layer: int = RANDOM_BASELINE_LAYER,
    device: str = "cpu",
) -> np.ndarray:
    """Extract full residual stream at a non-circuit layer for baseline comparison."""
    all_activations = []

    for idx, prompt in enumerate(prompts):
        if idx % 100 == 0:
            print(f"  Random-layer extraction {idx}/{len(prompts)}...")

        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=lambda n: f"blocks.{layer}.hook_resid_post" in n,
            )

        hook_name = f"blocks.{layer}.hook_resid_post"
        acts = cache[hook_name][0, -1, :].cpu().numpy()
        all_activations.append(acts)

        del cache
        if device == "cuda":
            torch.cuda.empty_cache()

    return np.array(all_activations)


# ── Per-pair restoration scores (for bootstrap) ──────────────────────

def extract_per_pair_restoration_scores(
    model,
    faithful_examples: List[Dict],
    unfaithful_examples: List[Dict],
    components: List[str],
    device: str = "cpu",
) -> Dict[str, List[float]]:
    """Compute per-pair restoration scores for each circuit component.

    Uses the faithful prompt as 'clean' and unfaithful as 'corrupted'.
    Returns ``{component_name: [score_pair_0, score_pair_1, ...]}``.
    """
    from shared.patching.hooks import make_head_patch_hook, make_mlp_patch_hook

    n_pairs = min(len(faithful_examples), len(unfaithful_examples))
    per_pair_scores: Dict[str, List[float]] = {comp: [] for comp in components}

    for pair_idx in range(n_pairs):
        if pair_idx % 50 == 0:
            print(f"  Restoration scores: pair {pair_idx}/{n_pairs}...")

        clean_prompt = faithful_examples[pair_idx]["prompt"]
        corrupted_prompt = unfaithful_examples[pair_idx]["prompt"]

        clean_tokens = model.to_tokens(clean_prompt)
        corrupted_tokens = model.to_tokens(corrupted_prompt)

        # Get correct/incorrect token IDs
        correct_str = faithful_examples[pair_idx]["correct_answer"]
        wrong_str = unfaithful_examples[pair_idx]["cot_answer"]
        correct_id = model.to_tokens(f" {correct_str}", prepend_bos=False).squeeze()[-1].item()
        incorrect_id = model.to_tokens(f" {wrong_str}", prepend_bos=False).squeeze()[-1].item()

        with torch.no_grad():
            clean_logits, clean_cache = model.run_with_cache(clean_tokens)
            corrupted_logits = model(corrupted_tokens)

        clean_diff = (clean_logits[0, -1, correct_id] - clean_logits[0, -1, incorrect_id]).item()
        corrupted_diff = (corrupted_logits[0, -1, correct_id] - corrupted_logits[0, -1, incorrect_id]).item()
        gap = clean_diff - corrupted_diff

        if abs(gap) < 1e-4:
            for comp in components:
                per_pair_scores[comp].append(float("nan"))
            continue

        for comp in components:
            ctype, layer, head = _parse_component(comp)

            if ctype == "head":
                hook_name = f"blocks.{layer}.attn.hook_z"
                hook_fn = make_head_patch_hook(clean_cache, layer, head)
            else:
                hook_name = f"blocks.{layer}.hook_mlp_out"
                hook_fn = make_mlp_patch_hook(clean_cache, layer)

            with torch.no_grad():
                patched_logits = model.run_with_hooks(
                    corrupted_tokens, fwd_hooks=[(hook_name, hook_fn)]
                )
            patched_diff = (patched_logits[0, -1, correct_id] - patched_logits[0, -1, incorrect_id]).item()
            score = (patched_diff - corrupted_diff) / gap
            per_pair_scores[comp].append(score)

        del clean_cache
        if device == "cuda":
            torch.cuda.empty_cache()

    return per_pair_scores


# ── Feature names ─────────────────────────────────────────────────────

def build_feature_names(components: List[str], d_head: int = 64, d_model: int = 768) -> List[str]:
    """Build feature name list matching the activation matrix columns."""
    names = []
    for comp in components:
        if comp.endswith("MLP"):
            for i in range(d_model):
                names.append(f"{comp}_dim{i}")
        else:
            for i in range(d_head):
                names.append(f"{comp}_dim{i}")
    return names


# ── Main extraction pipeline ─────────────────────────────────────────

def extract_all(
    device: str = "auto",
    n_pairs: int = 400,
    output_dir: Optional[str] = None,
    skip_restoration: bool = False,
) -> Path:
    """Run the full extraction pipeline.

    Returns the output directory path.
    """
    from transformer_lens import HookedTransformer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if output_dir is None:
        output_dir = str(
            Path(__file__).resolve().parent.parent / "results" / "activations"
        )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Output: {out}")
    print(f"Pairs:  {n_pairs}")

    # 1. Load model
    print("\n── Loading GPT-2 Small ──")
    model = HookedTransformer.from_pretrained(
        "gpt2", device=device,
        fold_ln=False, center_writing_weights=False, center_unembed=False,
    )
    model.eval()

    d_head = model.cfg.d_head
    d_model = model.cfg.d_model

    # 2. Generate dataset
    print("\n── Generating dataset ──")
    faithful, unfaithful = _generate_phase1_dataset(n_pairs=n_pairs, seed=42)
    all_examples = faithful + unfaithful

    # Shuffle with fixed seed (matching Phase 1)
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(all_examples))
    all_examples = [all_examples[i] for i in indices]

    prompts = [ex["prompt"] for ex in all_examples]
    labels = np.array([ex["label"] for ex in all_examples])

    print(f"  Total: {len(all_examples)} ({sum(labels == 0)} faithful, {sum(labels == 1)} unfaithful)")

    # 3. Extract circuit activations
    print("\n── Extracting circuit activations ──")
    X = extract_circuit_activations(model, prompts, KEY_COMPONENTS, device)
    feature_names = build_feature_names(KEY_COMPONENTS, d_head, d_model)

    print(f"  Activation matrix: {X.shape}")
    np.savez(
        out / "activations.npz",
        X=X, y=labels,
        feature_names=np.array(feature_names),
        components=np.array(KEY_COMPONENTS),
    )
    print(f"  Saved: {out / 'activations.npz'}")

    # 4. Extract random-layer baseline
    print("\n── Extracting random-layer baseline ──")
    X_random = extract_random_layer_activations(model, prompts, RANDOM_BASELINE_LAYER, device)
    np.savez(out / "random_layer_activations.npz", X=X_random, y=labels)
    print(f"  Saved: {out / 'random_layer_activations.npz'}")

    # 5. Save metadata
    print("\n── Saving metadata ──")
    # Strip prompts from metadata to keep it small
    metadata = []
    for ex in all_examples:
        meta = {k: v for k, v in ex.items() if k != "prompt"}
        metadata.append(meta)

    with open(out / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"  Saved: {out / 'metadata.json'}")

    # 6. Per-pair restoration scores (for bootstrap experiment)
    if not skip_restoration:
        print("\n── Computing per-pair restoration scores ──")
        per_pair_scores = extract_per_pair_restoration_scores(
            model, faithful, unfaithful, KEY_COMPONENTS, device
        )
        with open(out / "per_pair_scores.json", "w") as f:
            json.dump(per_pair_scores, f, indent=2)
        print(f"  Saved: {out / 'per_pair_scores.json'}")
    else:
        print("\n── Skipping restoration scores (--skip-restoration) ──")

    # 7. Save component type mapping
    with open(out / "component_types.json", "w") as f:
        json.dump(COMPONENT_TYPES, f, indent=2)

    print(f"\n{'='*60}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Files in {out}:")
    for fp in sorted(out.glob("*")):
        size_kb = fp.stat().st_size / 1024
        print(f"  {fp.name:40s} {size_kb:8.1f} KB")

    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract Phase 2A activations")
    parser.add_argument("--device", default="auto", help="cpu, cuda, or auto")
    parser.add_argument("--n-pairs", type=int, default=400)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--skip-restoration", action="store_true",
                        help="Skip per-pair restoration score computation")
    args = parser.parse_args()

    extract_all(
        device=args.device,
        n_pairs=args.n_pairs,
        output_dir=args.output_dir,
        skip_restoration=args.skip_restoration,
    )
