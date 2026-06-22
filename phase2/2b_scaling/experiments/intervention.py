"""
Experiment 2B-3: Intervention — Can we fix unfaithful reasoning?

Zero-ablates identified shortcut components and measures whether
the model shifts from "correct answer despite wrong CoT" to
"answer consistent with wrong CoT."

This is the central experiment of the project: if ablation
changes behavior, it proves the shortcut circuit is causal.

Usage::

    python phase2/2b_scaling/experiments/intervention.py \\
        --model qwen25-math-1.5b --device auto

Author: Victor Ashioya
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformer_lens import HookedTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Cannot use `from phase2.2b_scaling...` — digit prefix is invalid Python.
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "model_registry",
    str(Path(__file__).resolve().parent.parent / "src" / "model_registry.py"),
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["model_registry"] = _mod  # Required for @dataclass to resolve
_spec.loader.exec_module(_mod)
load_model = _mod.load_model

from shared.patching.hooks import make_zero_ablation_hook


def _parse_component(comp: str) -> Tuple[str, int, Optional[int]]:
    """Parse 'L7H6' -> ('head', 7, 6) or 'blocks.5.hook_mlp_out' -> ('mlp', 5, None)."""
    if comp.startswith("L") and "H" in comp:
        parts = comp[1:].split("H")
        return ("head", int(parts[0]), int(parts[1]))
    elif "hook_mlp_out" in comp:
        layer = int(comp.split(".")[1])
        return ("mlp", layer, None)
    elif "hook_attn_out" in comp:
        layer = int(comp.split(".")[1])
        return ("attn", layer, None)
    else:
        raise ValueError(f"Cannot parse component: {comp}")


def _get_model_answer(model: HookedTransformer, tokens: torch.Tensor,
                      fwd_hooks=None) -> Tuple[int, float]:
    """Get model's predicted answer and its probability.

    Returns (predicted_int, probability).
    """
    with torch.no_grad():
        if fwd_hooks:
            logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)
        else:
            logits = model(tokens)

    last_logits = logits[0, -1, :]
    probs = torch.softmax(last_logits, dim=-1)

    # Try top-5 tokens for a parseable number
    top5 = last_logits.topk(5)
    for tid, prob in zip(top5.indices.tolist(), top5.values.tolist()):
        s = model.to_string([tid]).strip()
        try:
            return int(s), probs[tid].item()
        except ValueError:
            continue

    return -1, 0.0


def run_intervention(
    model_key: str = "qwen25-math-1.5b",
    dataset_path: str | None = None,
    circuit_path: str | None = None,
    device: str = "auto",
    n_shortcut_components: int = 5,
    output_dir: str | None = None,
):
    """Run intervention experiments on identified shortcut components."""

    results_base = Path(__file__).resolve().parent.parent / "results"
    if output_dir:
        results_base = Path(output_dir)

    if dataset_path is None:
        dataset_path = str(results_base / "dataset.json")
    if circuit_path is None:
        circuit_path = str(results_base / "circuit_discovery_results.json")

    # ── Load data ────────────────────────────────────────────────────
    print("=" * 70)
    print("PHASE 2B-3: INTERVENTION EXPERIMENTS")
    print("=" * 70)

    with open(dataset_path) as f:
        dataset = json.load(f)
    with open(circuit_path) as f:
        circuit = json.load(f)

    # Get top shortcut components (highest |restoration score|)
    comp_stats = circuit["component_stats"]
    # Filter to head-level components (not layer-level)
    head_comps = {k: v for k, v in comp_stats.items() if "hook_" not in k}
    sorted_comps = sorted(head_comps.items(), key=lambda x: abs(x[1]["mean"]), reverse=True)
    target_components = [comp for comp, _ in sorted_comps[:n_shortcut_components]]

    print(f"\nTarget components for ablation:")
    for comp in target_components:
        stats = comp_stats[comp]
        print(f"  {comp}: mean={stats['mean']:.4f}")

    # Filter to unfaithful pairs (model got correct despite wrong CoT)
    unfaithful = [p for p in dataset if p.get("label") == 1]
    faithful = [p for p in dataset if p.get("label") == 0]
    print(f"\n  Unfaithful pairs: {len(unfaithful)}")
    print(f"  Faithful pairs:   {len(faithful)}")

    # ── Load model ───────────────────────────────────────────────────
    from transformer_lens import HookedTransformer
    if isinstance(model_key, HookedTransformer):
        model = model_key
        model_key = getattr(model.cfg, "model_name", "qwen25-math-1.5b")
    else:
        print(f"\nLoading {model_key}...")
        model = load_model(model_key, device=device)

    # ── Run ablation experiments ─────────────────────────────────────
    results = {}
    start_time = time.time()

    for n_ablate in range(1, len(target_components) + 1):
        ablate_set = target_components[:n_ablate]
        print(f"\n{'='*60}")
        print(f"ABLATING {n_ablate} COMPONENT(S): {', '.join(ablate_set)}")
        print(f"{'='*60}")

        # Build hooks for this ablation set
        hooks = []
        for comp in ablate_set:
            comp_type, layer, head_idx = _parse_component(comp)
            if comp_type == "head":
                hook_name = f"blocks.{layer}.attn.hook_z"
                hooks.append((hook_name, make_zero_ablation_hook(layer, head_idx)))
            elif comp_type == "mlp":
                hook_name = f"blocks.{layer}.hook_mlp_out"
                hooks.append((hook_name, make_zero_ablation_hook(layer)))
            elif comp_type == "attn":
                hook_name = f"blocks.{layer}.hook_attn_out"
                hooks.append((hook_name, make_zero_ablation_hook(layer)))

        # Test on unfaithful pairs
        intervention_success = 0
        intervention_total = 0

        for pair in unfaithful:
            tokens = model.to_tokens(pair["unfaithful_prompt"])

            # Baseline (no ablation)
            base_answer, base_prob = _get_model_answer(model, tokens)

            # Ablated
            ablated_answer, ablated_prob = _get_model_answer(model, tokens, fwd_hooks=hooks)

            # Success = model shifts from correct to CoT-consistent (wrong)
            if base_answer == pair["correct_answer"]:
                intervention_total += 1
                if ablated_answer == pair["corrupted_answer"]:
                    intervention_success += 1
                elif ablated_answer != pair["correct_answer"]:
                    # Model changed answer but not to CoT-consistent
                    pass  # Count as partial

        # Test on faithful pairs (ablation should NOT hurt these)
        faithful_preserved = 0
        faithful_total = 0

        for pair in faithful:
            tokens = model.to_tokens(pair["faithful_prompt"])

            base_answer, _ = _get_model_answer(model, tokens)
            ablated_answer, _ = _get_model_answer(model, tokens, fwd_hooks=hooks)

            if base_answer == pair["correct_answer"]:
                faithful_total += 1
                if ablated_answer == pair["correct_answer"]:
                    faithful_preserved += 1

        # Compute rates
        success_rate = intervention_success / max(intervention_total, 1)
        preservation_rate = faithful_preserved / max(faithful_total, 1)

        ablation_key = "+".join(ablate_set)
        results[ablation_key] = {
            "components": ablate_set,
            "n_components": n_ablate,
            "intervention_success": intervention_success,
            "intervention_total": intervention_total,
            "success_rate": success_rate,
            "faithful_preserved": faithful_preserved,
            "faithful_total": faithful_total,
            "preservation_rate": preservation_rate,
        }

        print(f"\n  Intervention success rate:   {success_rate:.1%} ({intervention_success}/{intervention_total})")
        print(f"  Faithful preservation rate: {preservation_rate:.1%} ({faithful_preserved}/{faithful_total})")

        target_met = success_rate > 0.20
        print(f"  Target (>20%):              {'PASS' if target_met else 'FAIL'}")

    elapsed = time.time() - start_time

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("INTERVENTION SUMMARY")
    print(f"{'='*70}")

    print(f"\n  {'Components':<30} {'Success':>10} {'Preserve':>10}")
    print(f"  {'-'*50}")
    for key, r in results.items():
        print(f"  {key:<30} {r['success_rate']:>9.1%} {r['preservation_rate']:>9.1%}")

    best = max(results.values(), key=lambda r: r["success_rate"])
    print(f"\n  Best intervention: {'+'.join(best['components'])}")
    print(f"  Success rate: {best['success_rate']:.1%}")
    print(f"  Elapsed: {elapsed/60:.1f} minutes")

    # ── Save ─────────────────────────────────────────────────────────
    output = {
        "model": model_key,
        "target_components": target_components,
        "results": results,
        "elapsed_seconds": elapsed,
    }

    output_path = results_base / "intervention_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")

    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 2B-3: Intervention")
    parser.add_argument("--model", default="qwen25-math-1.5b")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--circuit", default=None, help="Path to circuit_discovery_results.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-components", type=int, default=5)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    run_intervention(
        model_key=args.model,
        dataset_path=args.dataset,
        circuit_path=args.circuit,
        device=args.device,
        n_shortcut_components=args.n_components,
        output_dir=args.output_dir,
    )
