"""
Experiment 2B-1: Circuit Discovery on Qwen2.5-Math

Two-pass activation patching to identify faithful/shortcut components
in a math-capable model. This is the Phase 1 replication on a model
that can actually do arithmetic.

Usage::

    python phase2/2b_scaling/experiments/circuit_discovery.py \\
        --model qwen25-math-1.5b --device auto

Author: Victor Ashioya
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Cannot use `from phase2.2b_scaling...` — digit prefix is invalid Python.
import importlib.util

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # Required for @dataclass to resolve
    spec.loader.exec_module(mod)
    return mod

_src = Path(__file__).resolve().parent.parent / "src"
_registry = _load_module("model_registry", str(_src / "model_registry.py"))
_patching = _load_module("efficient_patching", str(_src / "efficient_patching.py"))
load_model = _registry.load_model
two_pass_patching = _patching.two_pass_patching


def run_circuit_discovery(
    model_key: str = "qwen25-math-1.5b",
    dataset_path: str | None = None,
    device: str = "auto",
    top_k_layers: int = 6,
    max_pairs: int = 200,
    output_dir: str | None = None,
):
    """Run two-pass patching on contrastive pairs to discover circuit components."""

    results_base = Path(__file__).resolve().parent.parent / "results"
    if output_dir:
        results_base = Path(output_dir)
    results_base.mkdir(parents=True, exist_ok=True)

    if dataset_path is None:
        dataset_path = str(results_base / "dataset.json")

    # ── Load dataset ─────────────────────────────────────────────────
    print("=" * 70)
    print("PHASE 2B-1: CIRCUIT DISCOVERY")
    print("=" * 70)

    print(f"\nLoading dataset from {dataset_path}")
    with open(dataset_path) as f:
        dataset = json.load(f)

    # Filter to valid unfaithful pairs (model got right answer despite wrong CoT)
    valid_pairs = [p for p in dataset if p.get("is_valid") and p.get("label") in (0, 1)]
    if len(valid_pairs) > max_pairs:
        valid_pairs = valid_pairs[:max_pairs]
    print(f"  Using {len(valid_pairs)} valid pairs")
    print(f"  Faithful: {sum(1 for p in valid_pairs if p['label'] == 0)}")
    print(f"  Unfaithful: {sum(1 for p in valid_pairs if p['label'] == 1)}")

    # ── Load model ───────────────────────────────────────────────────
    from transformer_lens import HookedTransformer
    if isinstance(model_key, HookedTransformer):
        model = model_key
        model_key = getattr(model.cfg, "model_name", "qwen25-math-1.5b")
    else:
        print(f"\nLoading {model_key}...")
        model = load_model(model_key, device=device)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    print(f"  Layers: {n_layers}, Heads: {n_heads}")
    print(f"  Total heads: {n_layers * n_heads}")
    print(f"  Two-pass strategy: layer sweep ({2 * n_layers} runs) + top-{top_k_layers} head sweep ({top_k_layers * n_heads} runs)")

    # ── Run patching ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RUNNING TWO-PASS PATCHING")
    print(f"{'='*60}")

    all_scores: Dict[str, List[float]] = {}  # component -> list of scores
    start_time = time.time()

    for i, pair in enumerate(valid_pairs):
        if i % 20 == 0:
            elapsed = time.time() - start_time
            rate = elapsed / max(i, 1)
            eta = rate * (len(valid_pairs) - i) / 60
            print(f"  Pair {i}/{len(valid_pairs)} (ETA: {eta:.1f} min)")

        # Tokenize the faithful and unfaithful prompts
        faithful_tokens = model.to_tokens(pair["faithful_prompt"])
        unfaithful_tokens = model.to_tokens(pair["unfaithful_prompt"])

        # Get correct and corrupted answer token IDs
        correct_id = model.to_tokens(f" {pair['correct_answer']}")[0, -1].item()
        corrupted_id = model.to_tokens(f" {pair['corrupted_answer']}")[0, -1].item()

        # Use faithful=clean, unfaithful=corrupted for patching
        # (patching clean into corrupted measures how much each component
        #  contributes to the correct answer)
        try:
            scores = two_pass_patching(
                model,
                clean_tokens=faithful_tokens,
                corrupted_tokens=unfaithful_tokens,
                correct_id=correct_id,
                incorrect_id=corrupted_id,
                top_k_layers=top_k_layers,
            )
        except Exception as e:
            print(f"    Pair {i} failed: {e}")
            continue

        for comp, score in scores.items():
            if comp not in all_scores:
                all_scores[comp] = []
            all_scores[comp].append(score)

    elapsed = time.time() - start_time
    print(f"\n  Patching complete in {elapsed/60:.1f} minutes")

    # ── Aggregate results ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("COMPONENT RANKINGS")
    print(f"{'='*60}")

    component_stats = {}
    for comp, scores in all_scores.items():
        arr = np.array(scores)
        component_stats[comp] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "n_valid": len(scores),
            "ci_lower": float(np.percentile(arr, 2.5)),
            "ci_upper": float(np.percentile(arr, 97.5)),
        }

    # Sort by absolute mean restoration score
    sorted_components = sorted(
        component_stats.items(),
        key=lambda x: abs(x[1]["mean"]),
        reverse=True,
    )

    # Separate layer-level and head-level
    layer_components = [(c, s) for c, s in sorted_components if "hook_" in c]
    head_components = [(c, s) for c, s in sorted_components if "hook_" not in c]

    print(f"\n  Top Layer-Level Components (by |mean restoration|):")
    print(f"  {'Component':<30} {'Mean':>8} {'Std':>8} {'CI':>20}")
    print(f"  {'-'*66}")
    for comp, stats in layer_components[:10]:
        ci = f"[{stats['ci_lower']:.3f}, {stats['ci_upper']:.3f}]"
        print(f"  {comp:<30} {stats['mean']:>8.4f} {stats['std']:>8.4f} {ci:>20}")

    if head_components:
        print(f"\n  Top Head-Level Components (by |mean restoration|):")
        print(f"  {'Component':<12} {'Mean':>8} {'Std':>8} {'CI':>20}")
        print(f"  {'-'*48}")
        for comp, stats in head_components[:15]:
            ci = f"[{stats['ci_lower']:.3f}, {stats['ci_upper']:.3f}]"
            print(f"  {comp:<12} {stats['mean']:>8.4f} {stats['std']:>8.4f} {ci:>20}")

    # ── Save ─────────────────────────────────────────────────────────
    output = {
        "model": model_key,
        "n_pairs": len(valid_pairs),
        "n_layers": n_layers,
        "n_heads": n_heads,
        "top_k_layers": top_k_layers,
        "elapsed_seconds": elapsed,
        "component_stats": component_stats,
        "per_pair_scores": {comp: scores for comp, scores in all_scores.items()},
    }

    output_path = results_base / "circuit_discovery_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")

    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 2B-1: Circuit Discovery")
    parser.add_argument("--model", default="qwen25-math-1.5b")
    parser.add_argument("--dataset", default=None, help="Path to dataset.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--max-pairs", type=int, default=200)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    run_circuit_discovery(
        model_key=args.model,
        dataset_path=args.dataset,
        device=args.device,
        top_k_layers=args.top_k,
        max_pairs=args.max_pairs,
        output_dir=args.output_dir,
    )
