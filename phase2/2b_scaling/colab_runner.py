"""
Phase 2B Colab Runner
======================

Runs the full Phase 2B pipeline on Google Colab:
1. Dataset generation (contrastive pairs with grounded labels)
2. Circuit discovery (two-pass activation patching)
3. Intervention (zero-ablation of shortcut components)

Usage on Colab::

    !git clone https://github.com/ashioyajotham/cot-faithfulness-mech-interp.git
    %cd cot-faithfulness-mech-interp
    !pip install -e ".[phase2b]" -q
    !python phase2/2b_scaling/colab_runner.py --model qwen25-math-1.5b --device auto
"""

import importlib.util
import os
import sys
import time
from pathlib import Path

# ── HF Token setup (Colab secrets or env var) ─────────────────────────
def _setup_hf_token():
    """Auto-detect HF token from Colab secrets or environment."""
    if os.environ.get("HF_TOKEN"):
        print(f"  HF_TOKEN found in environment")
        return
    try:
        from google.colab import userdata
        token = userdata.get("HF_TOKEN")
        if token:
            os.environ["HF_TOKEN"] = token
            print(f"  HF_TOKEN loaded from Colab secrets")
            return
    except (ImportError, Exception):
        pass
    print("  Warning: No HF_TOKEN found. Downloads may be rate-limited.")

_setup_hf_token()

# ── Path setup ────────────────────────────────────────────────────────
PROJECT_ROOT = os.getcwd()
sys.path.insert(0, PROJECT_ROOT)

RESULTS_DIR = Path(PROJECT_ROOT) / "phase2" / "2b_scaling" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _import_from_path(module_name: str, file_path: str):
    """Import a module from an absolute path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # Required for @dataclass to resolve
    spec.loader.exec_module(module)
    return module


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Phase 2B Pipeline Runner")
    parser.add_argument("--model", default="qwen25-math-1.5b")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-pairs", type=int, default=250)
    parser.add_argument("--max-discovery-pairs", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--skip-dataset", action="store_true")
    parser.add_argument("--skip-discovery", action="store_true")
    args = parser.parse_args()

    overall_start = time.time()

    print("=" * 70)
    print("PHASE 2B -- FULL PIPELINE")
    print("When Models Lie to Please: Mechanistic Detection of Unfaithful CoT")
    print("=" * 70)
    print(f"\nModel:  {args.model}")
    print(f"Device: {args.device}")
    print(f"Pairs:  {args.n_pairs}")

    # ── Step 1: Dataset Generation ───────────────────────────────────
    dataset_path = RESULTS_DIR / "dataset.json"

    if not args.skip_dataset:
        print(f"\n{'='*70}")
        print("STEP 1: DATASET GENERATION")
        print(f"{'='*70}")

        dataset_gen = _import_from_path(
            "dataset_generator",
            str(Path(PROJECT_ROOT) / "phase2" / "2b_scaling" / "src" / "dataset_generator.py"),
        )
        _registry = _import_from_path(
            "model_registry",
            str(Path(PROJECT_ROOT) / "phase2" / "2b_scaling" / "src" / "model_registry.py"),
        )

        problems = dataset_gen.generate_problems(args.n_pairs)
        pairs = dataset_gen.generate_contrastive_pairs(problems)

        model = _registry.load_model(args.model, device=args.device)
        pairs = dataset_gen.evaluate_pairs(model, pairs)
        dataset_gen.save_dataset(pairs, dataset_path)

        # Free model memory for next step
        del model
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print(f"\n  Skipping dataset generation (using {dataset_path})")

    # ── Step 2: Circuit Discovery ────────────────────────────────────
    circuit_path = RESULTS_DIR / "circuit_discovery_results.json"

    if not args.skip_discovery:
        print(f"\n{'='*70}")
        print("STEP 2: CIRCUIT DISCOVERY")
        print(f"{'='*70}")

        discovery = _import_from_path(
            "circuit_discovery",
            str(Path(PROJECT_ROOT) / "phase2" / "2b_scaling" / "experiments" / "circuit_discovery.py"),
        )
        discovery.run_circuit_discovery(
            model_key=args.model,
            dataset_path=str(dataset_path),
            device=args.device,
            top_k_layers=args.top_k,
            max_pairs=args.max_discovery_pairs,
            output_dir=str(RESULTS_DIR),
        )

        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print(f"\n  Skipping discovery (using {circuit_path})")

    # ── Step 2.5: Detection Probe ────────────────────────────────────
    print(f"\n{'='*70}")
    print("STEP 2.5: DETECTION PROBE & DUAL-METRIC ANALYSIS")
    print(f"{'='*70}")

    detection = _import_from_path(
        "detection_probe",
        str(Path(PROJECT_ROOT) / "phase2" / "2b_scaling" / "experiments" / "detection_probe.py"),
    )
    detection.run_detection_probe(
        model_key=args.model,
        dataset_path=str(dataset_path),
        circuit_path=str(circuit_path),
        device=args.device,
        output_dir=str(RESULTS_DIR),
    )

    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Step 3: Intervention ─────────────────────────────────────────
    print(f"\n{'='*70}")
    print("STEP 3: INTERVENTION")
    print(f"{'='*70}")

    intervention = _import_from_path(
        "intervention",
        str(Path(PROJECT_ROOT) / "phase2" / "2b_scaling" / "experiments" / "intervention.py"),
    )
    intervention.run_intervention(
        model_key=args.model,
        dataset_path=str(dataset_path),
        circuit_path=str(circuit_path),
        device=args.device,
        output_dir=str(RESULTS_DIR),
    )

    # ── Summary ──────────────────────────────────────────────────────
    total_time = time.time() - overall_start
    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Results in: {RESULTS_DIR}")

    # List result files
    for f in sorted(RESULTS_DIR.glob("*.json")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:40s} {size_kb:8.1f} KB")


if __name__ == "__main__":
    main()
