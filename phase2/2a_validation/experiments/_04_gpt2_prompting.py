"""
Experiment 04 — GPT-2 Completion-Style Prompting
==================================================

Tests whether GPT-2 can perform arithmetic under any prompt format.
If so, enables intervention experiments on the original model.

Three formats tested:
- Format A: ``"23 + 45 = "`` (bare equation)
- Format B: ``"The sum of 23 and 45 is "`` (natural language)
- Format C: Few-shot with 3 worked examples

Research Question (RQ6): Can completion-style prompting elicit
arithmetic behaviour from GPT-2?

Usage::

    python phase2/2a_validation/experiments/04_gpt2_prompting.py \\
        [--device auto] [--n-problems 100]

Author: Victor Ashioya
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Configuration ─────────────────────────────────────────────────────

BASELINE_THRESHOLD = 0.05  # 5% accuracy → try intervention
N_PROBLEMS = 100


def _generate_problems(n: int, seed: int = 42) -> List[Tuple[int, int, int]]:
    """Generate n arithmetic problems as (a, b, correct_answer)."""
    rng = np.random.RandomState(seed)
    problems = []
    for _ in range(n):
        a = rng.randint(10, 50)
        b = rng.randint(10, 50)
        problems.append((int(a), int(b), int(a + b)))
    return problems


def _format_a(a: int, b: int) -> str:
    """Bare equation: ``23 + 45 = ``"""
    return f"{a} + {b} ="


def _format_b(a: int, b: int) -> str:
    """Natural language: ``The sum of 23 and 45 is ``"""
    return f"The sum of {a} and {b} is"


def _format_c(a: int, b: int) -> str:
    """Few-shot: 3 examples + target."""
    return (
        "12 + 34 = 46\n"
        "21 + 15 = 36\n"
        "33 + 44 = 77\n"
        f"{a} + {b} ="
    )


def _format_original(a: int, b: int) -> str:
    """Phase 1 format for comparison."""
    a_u, a_t = a % 10, a // 10
    b_u, b_t = b % 10, b // 10
    units = a_u + b_u
    tens = a_t + b_t
    return (
        f"Q: What is {a}+{b}?\n"
        f"Steps: units={a_u}+{b_u}={units}, tens={a_t}+{b_t}={tens}.\n"
        f"A:"
    )


FORMATS = {
    "format_a_equation": _format_a,
    "format_b_natural": _format_b,
    "format_c_fewshot": _format_c,
    "format_original": _format_original,
}


def evaluate_format(
    model,
    problems: List[Tuple[int, int, int]],
    format_fn,
    format_name: str,
) -> Dict:
    """Evaluate arithmetic accuracy for a specific prompt format.

    Checks whether the correct answer token is in the top-k predictions.
    """
    correct_top1 = 0
    correct_top5 = 0
    correct_top10 = 0
    total_p_correct = 0.0
    details = []

    for a, b, correct in problems:
        prompt = format_fn(a, b)
        tokens = model.to_tokens(prompt)

        with torch.no_grad():
            logits = model(tokens)

        # Get the last position logits
        last_logits = logits[0, -1, :]
        probs = torch.softmax(last_logits, dim=-1)

        # Try both with and without space prefix
        correct_str_variants = [f" {correct}", str(correct)]
        best_p = 0.0
        best_rank = 999999

        for cs in correct_str_variants:
            try:
                correct_ids = model.to_tokens(cs, prepend_bos=False).squeeze()
                if correct_ids.dim() == 0:
                    cid = correct_ids.item()
                else:
                    cid = correct_ids[0].item()

                p = probs[cid].item()
                # Rank
                sorted_indices = torch.argsort(probs, descending=True)
                rank = (sorted_indices == cid).nonzero(as_tuple=True)[0].item() + 1

                best_p = max(best_p, p)
                best_rank = min(best_rank, rank)
            except Exception:
                continue

        if best_rank == 1:
            correct_top1 += 1
        if best_rank <= 5:
            correct_top5 += 1
        if best_rank <= 10:
            correct_top10 += 1
        total_p_correct += best_p

        # Get top-5 predicted tokens for logging
        top5_ids = torch.argsort(probs, descending=True)[:5]
        top5_tokens = [model.to_string(tid.unsqueeze(0)).strip() for tid in top5_ids]
        top5_probs = [probs[tid].item() for tid in top5_ids]

        details.append({
            "a": a, "b": b, "correct": correct,
            "p_correct": best_p,
            "rank": best_rank,
            "top5_tokens": top5_tokens,
            "top5_probs": [round(p, 4) for p in top5_probs],
        })

    n = len(problems)
    return {
        "format": format_name,
        "accuracy_top1": correct_top1 / n,
        "accuracy_top5": correct_top5 / n,
        "accuracy_top10": correct_top10 / n,
        "mean_p_correct": total_p_correct / n,
        "n_correct_top1": correct_top1,
        "n_problems": n,
        "details": details,
    }


def run_experiment(device: str = "auto", n_problems: int = N_PROBLEMS) -> dict:
    """Run all prompt format evaluations."""
    from transformer_lens import HookedTransformer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ───────────────────────────────────────────────────
    print(f"Loading GPT-2 Small on {device}...")
    model = HookedTransformer.from_pretrained(
        "gpt2", device=device,
        fold_ln=False, center_writing_weights=False, center_unembed=False,
    )
    model.eval()

    # ── Generate problems ────────────────────────────────────────────
    problems = _generate_problems(n_problems)
    print(f"Generated {n_problems} arithmetic problems")

    # ── Evaluate each format ─────────────────────────────────────────
    all_results = {"formats": {}}

    for name, fmt_fn in FORMATS.items():
        print(f"\n{'='*60}")
        print(f"FORMAT: {name}")
        print(f"{'='*60}")

        # Show example prompt
        a, b, c = problems[0]
        print(f"  Example: {repr(fmt_fn(a, b))}")

        fmt_results = evaluate_format(model, problems, fmt_fn, name)
        # Don't save full details in summary (too large)
        summary = {k: v for k, v in fmt_results.items() if k != "details"}
        all_results["formats"][name] = summary

        print(f"  Top-1 accuracy:  {fmt_results['accuracy_top1']:.1%} ({fmt_results['n_correct_top1']}/{n_problems})")
        print(f"  Top-5 accuracy:  {fmt_results['accuracy_top5']:.1%}")
        print(f"  Top-10 accuracy: {fmt_results['accuracy_top10']:.1%}")
        print(f"  Mean P(correct): {fmt_results['mean_p_correct']:.4f}")

        # Show some examples
        print(f"\n  Sample predictions:")
        for d in fmt_results["details"][:5]:
            pred = d["top5_tokens"][0] if d["top5_tokens"] else "?"
            print(f"    {d['a']}+{d['b']}={d['correct']}  → predicted '{pred}' (rank {d['rank']}, p={d['p_correct']:.4f})")

        # Save detailed results per format
        detail_path = results_dir / f"04_{name}_details.json"
        with open(detail_path, "w") as f:
            json.dump(fmt_results, f, indent=2)

    # ── Comparison table ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("COMPARISON TABLE")
    print(f"{'='*60}")
    print(f"{'Format':<25} {'Top1':>6} {'Top5':>6} {'Top10':>6} {'P(corr)':>8}")
    print("-" * 55)
    for name, res in all_results["formats"].items():
        print(f"{name:<25} {res['accuracy_top1']:>6.1%} {res['accuracy_top5']:>6.1%} "
              f"{res['accuracy_top10']:>6.1%} {res['mean_p_correct']:>8.4f}")

    # ── Best format ──────────────────────────────────────────────────
    best_format = max(all_results["formats"].items(), key=lambda x: x[1]["accuracy_top1"])
    all_results["best_format"] = best_format[0]
    all_results["best_accuracy"] = best_format[1]["accuracy_top1"]

    # ── Intervention decision ────────────────────────────────────────
    print(f"\n{'='*60}")
    print("INTERVENTION DECISION")
    print(f"{'='*60}")

    can_intervene = best_format[1]["accuracy_top1"] > BASELINE_THRESHOLD
    all_results["can_intervene"] = can_intervene

    if can_intervene:
        print(f"  Best format ({best_format[0]}): {best_format[1]['accuracy_top1']:.1%} > {BASELINE_THRESHOLD:.0%}")
        print("  → PROCEED with intervention experiments on GPT-2")
        all_results["intervention_recommendation"] = (
            f"Use {best_format[0]} format for GPT-2 intervention experiments. "
            f"Ablate L7H6 and measure shift rate."
        )
    else:
        print(f"  Best format ({best_format[0]}): {best_format[1]['accuracy_top1']:.1%} ≤ {BASELINE_THRESHOLD:.0%}")
        print("  → CONFIRMED: GPT-2 cannot do arithmetic in any prompt format")
        print("  → Intervention experiments require Phase 2B (Qwen/Gemma)")
        all_results["intervention_recommendation"] = (
            "GPT-2 arithmetic failure confirmed across all prompt formats. "
            "Intervention experiments are intractable on GPT-2 — proceed to Phase 2B."
        )

    # ── Save ─────────────────────────────────────────────────────────
    output_path = results_dir / "04_gpt2_prompting_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiment 04: GPT-2 Prompting")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-problems", type=int, default=N_PROBLEMS)
    args = parser.parse_args()

    run_experiment(device=args.device, n_problems=args.n_problems)
