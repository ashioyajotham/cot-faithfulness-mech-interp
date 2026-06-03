"""
Phase 2A — Run All Validation Experiments
==========================================

Master orchestrator that:
1. Extracts activations (or uses cached if available)
2. Runs Experiments 01–04 in sequence
3. Collects all results into a summary
4. Evaluates gate conditions (2A-G1, 2A-G2, 2A-G3)
5. Prints go/no-go recommendation for Phase 2B

Usage::

    # Full pipeline (requires GPT-2 model download)
    python phase2/2a_validation/run_all_2a.py

    # Skip extraction (use pre-computed activations)
    python phase2/2a_validation/run_all_2a.py --skip-extraction

    # Skip restoration scores (faster, but Exp 03 won't work)
    python phase2/2a_validation/run_all_2a.py --skip-restoration

Author: Victor Ashioya
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────
# '2a_validation' starts with a digit → not a valid Python package name.
# We use importlib to load modules from the filesystem directly.

_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR / "src"
_EXP_DIR = _THIS_DIR / "experiments"
PROJECT_ROOT = _THIS_DIR.parent.parent

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(_SRC_DIR))

RESULTS_DIR = _THIS_DIR / "results"
ACTIVATIONS_DIR = RESULTS_DIR / "activations"


def _load_module(name: str, filepath: Path):
    """Import a module from an absolute filepath."""
    spec = importlib.util.spec_from_file_location(name, str(filepath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(
    device: str = "auto",
    skip_extraction: bool = False,
    skip_restoration: bool = False,
    n_pairs: int = 400,
):
    start_time = time.time()

    print("=" * 70)
    print("PHASE 2A — VALIDATION EXPERIMENTS")
    print("When Models Lie to Please: Mechanistic Detection of Unfaithful CoT")
    print("=" * 70)

    summary = {"experiments": {}, "gates": {}}

    # ── Step 0: Activation Extraction ─────────────────────────────────
    if not skip_extraction:
        print("\n" + "=" * 70)
        print("STEP 0: Activation Extraction")
        print("=" * 70)
        extract_mod = _load_module("extract_activations", _SRC_DIR / "extract_activations.py")
        extract_mod.extract_all(
            device=device,
            n_pairs=n_pairs,
            output_dir=str(ACTIVATIONS_DIR),
            skip_restoration=skip_restoration,
        )
    else:
        print("\n[Skipping extraction — using cached activations]")
        if not (ACTIVATIONS_DIR / "activations.npz").exists():
            print(f"ERROR: {ACTIVATIONS_DIR / 'activations.npz'} not found!")
            print("Run without --skip-extraction first.")
            sys.exit(1)

    # ── Step 1: Probe Selectivity ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 1: Experiment 01 — Probe Selectivity (Hewitt-Liang)")
    print("=" * 70)
    exp01 = _load_module("exp01", _EXP_DIR / "_01_probe_selectivity.py")
    results_01 = exp01.run_experiment(str(ACTIVATIONS_DIR))
    summary["experiments"]["01_probe_selectivity"] = results_01

    # ── Step 2: False Negative Analysis ───────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 2: Experiment 02 — False Negative Analysis")
    print("=" * 70)
    exp02 = _load_module("exp02", _EXP_DIR / "_02_false_negative_analysis.py")
    results_02 = exp02.run_experiment(str(ACTIVATIONS_DIR))
    summary["experiments"]["02_false_negative_analysis"] = results_02

    # ── Step 3: Bootstrap Significance ────────────────────────────────
    if not skip_restoration:
        print("\n" + "=" * 70)
        print("STEP 3: Experiment 03 — Bootstrap Significance")
        print("=" * 70)
        exp03 = _load_module("exp03", _EXP_DIR / "_03_bootstrap_significance.py")
        results_03 = exp03.run_experiment(str(ACTIVATIONS_DIR))
        summary["experiments"]["03_bootstrap_significance"] = results_03
    else:
        print("\n[Skipping Experiment 03 — requires restoration scores]")
        summary["experiments"]["03_bootstrap_significance"] = {"skipped": True}

    # ── Step 4: GPT-2 Prompting ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 4: Experiment 04 — GPT-2 Prompting Formats")
    print("=" * 70)
    exp04 = _load_module("exp04", _EXP_DIR / "_04_gpt2_prompting.py")
    results_04 = exp04.run_experiment(device=device)
    summary["experiments"]["04_gpt2_prompting"] = results_04

    # ── Gate Evaluation ───────────────────────────────────────────────
    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("GATE EVALUATION — Phase 2A → Phase 2B")
    print("=" * 70)

    g1 = results_01.get("gate_2a_g1_pass", False)
    g2 = summary["experiments"]["03_bootstrap_significance"].get("gate_2a_g2_pass", None)
    g3 = results_02.get("gate_2a_g3_pass", False)

    summary["gates"] = {
        "2A-G1 (selectivity > 0)": {"pass": g1, "detail": results_01.get("gate_2a_g1_detail", "")},
        "2A-G2 (L7H6 rank 1 > 90%)": {"pass": g2, "detail": summary["experiments"]["03_bootstrap_significance"].get("gate_2a_g2_detail", "skipped")},
        "2A-G3 (FN cluster found)": {"pass": g3, "detail": results_02.get("gate_2a_g3_detail", "")},
    }

    print(f"\n  Gate 2A-G1 (Probe selective):    {'PASS ✓' if g1 else 'FAIL ✗'}")
    print(f"  Gate 2A-G2 (L7H6 dominant):      {'PASS ✓' if g2 else ('SKIP' if g2 is None else 'FAIL ✗')}")
    print(f"  Gate 2A-G3 (FN cluster):          {'PASS ✓' if g3 else 'INCONCLUSIVE ◌'}")

    proceed = g1  # G1 is the critical gate per the proposal
    summary["proceed_to_phase2b"] = proceed
    summary["elapsed_seconds"] = elapsed

    print(f"\n  {'→ PROCEED to Phase 2B ✓' if proceed else '→ REVIEW before proceeding to Phase 2B ⚠'}")
    print(f"\n  Total time: {elapsed/60:.1f} minutes")

    # ── Save summary ─────────────────────────────────────────────────
    output_path = RESULTS_DIR / "phase2a_summary.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Summary saved to {output_path}")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run all Phase 2A experiments")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-extraction", action="store_true")
    parser.add_argument("--skip-restoration", action="store_true")
    parser.add_argument("--n-pairs", type=int, default=400)
    args = parser.parse_args()

    main(
        device=args.device,
        skip_extraction=args.skip_extraction,
        skip_restoration=args.skip_restoration,
        n_pairs=args.n_pairs,
    )
