"""
Phase 2A Colab Runner
======================

Copy this file to Google Colab and run all cells.  It installs
dependencies, clones the repo, runs the full Phase 2A pipeline
(activation extraction + 4 experiments), and saves results.

After running, download the ``phase2/2a_validation/results/`` folder
and place it in your local repo.

Runtime: ~2–4 hours on Colab T4/A100 (mostly activation extraction).
"""

# ── Cell 1: Install dependencies ──────────────────────────────────────
# !pip install -q 'transformer-lens>=2.0' 'transformers>=4.40,<4.46' \
#     torch scikit-learn scipy einops jaxtyping matplotlib seaborn numpy wandb

# ── Cell 2: Clone repo ───────────────────────────────────────────────
# !git clone https://github.com/ashioyajotham/cot-faithfulness-mech-interp.git
# %cd cot-faithfulness-mech-interp
# !pip install -e ".[phase2a]" -q

# ── Cell 3: Path setup ──────────────────────────────────────────────
import importlib.util
import sys
import os
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = os.getcwd()
sys.path.insert(0, PROJECT_ROOT)

# '2a_validation' starts with a digit → can't use dotted imports.
# Use importlib to load modules from filesystem.
SRC_DIR = Path(PROJECT_ROOT) / "phase2" / "2a_validation" / "src"
EXP_DIR = Path(PROJECT_ROOT) / "phase2" / "2a_validation" / "experiments"
sys.path.insert(0, str(SRC_DIR))

def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, str(filepath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# ── Cell 4: Run activation extraction ────────────────────────────────
print("=" * 70)
print("PHASE 2A: ACTIVATION EXTRACTION")
print("=" * 70)

extract_mod = _load_module("extract_activations", SRC_DIR / "extract_activations.py")
extract_mod.extract_all(device="auto", n_pairs=400, skip_restoration=False)

# ── Cell 5: Run Experiment 01 — Probe Selectivity ────────────────────
print("\n" + "=" * 70)
print("EXPERIMENT 01: PROBE SELECTIVITY")
print("=" * 70)

exp01 = _load_module("exp01", EXP_DIR / "_01_probe_selectivity.py")
results_01 = exp01.run_experiment("phase2/2a_validation/results/activations")

# ── Cell 6: Run Experiment 02 — False Negative Analysis ──────────────
print("\n" + "=" * 70)
print("EXPERIMENT 02: FALSE NEGATIVE ANALYSIS")
print("=" * 70)

exp02 = _load_module("exp02", EXP_DIR / "_02_false_negative_analysis.py")
results_02 = exp02.run_experiment("phase2/2a_validation/results/activations")

# ── Cell 7: Run Experiment 03 — Bootstrap Significance ───────────────
print("\n" + "=" * 70)
print("EXPERIMENT 03: BOOTSTRAP SIGNIFICANCE")
print("=" * 70)

exp03 = _load_module("exp03", EXP_DIR / "_03_bootstrap_significance.py")
results_03 = exp03.run_experiment("phase2/2a_validation/results/activations")

# ── Cell 8: Run Experiment 04 — GPT-2 Prompting Formats ──────────────
print("\n" + "=" * 70)
print("EXPERIMENT 04: GPT-2 PROMPTING FORMATS")
print("=" * 70)

exp04 = _load_module("exp04", EXP_DIR / "_04_gpt2_prompting.py")
results_04 = exp04.run_experiment(device="auto")

# ── Cell 9: Summary ──────────────────────────────────────────────────
import json

summary = {
    "01_selectivity": results_01,
    "02_false_negatives": results_02,
    "03_bootstrap": results_03,
    "04_prompting": results_04,
    "gates": {
        "G1_selectivity_pass": results_01.get("gate_2a_g1_pass", False),
        "G2_l7h6_dominance_pass": results_03.get("gate_2a_g2_pass", None),
        "G3_fn_cluster_pass": results_02.get("gate_2a_g3_pass", False),
    }
}

with open("phase2/2a_validation/results/phase2a_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

print("\n" + "=" * 70)
print("PHASE 2A COMPLETE")
print("=" * 70)
print(f"Gate G1 (selectivity > 0):    {'PASS ✓' if summary['gates']['G1_selectivity_pass'] else 'FAIL ✗'}")
print(f"Gate G2 (L7H6 dominant):      {'PASS ✓' if summary['gates']['G2_l7h6_dominance_pass'] else 'FAIL ✗'}")
print(f"Gate G3 (FN cluster):         {'PASS ✓' if summary['gates']['G3_fn_cluster_pass'] else 'INCONCLUSIVE ◌'}")

# ── Cell 10: Download results ────────────────────────────────────────
# Zip the results for download
# !cd phase2/2a_validation && zip -r /content/phase2a_results.zip results/
# from google.colab import files
# files.download('/content/phase2a_results.zip')

print("\n→ Download phase2a_results.zip and extract into your local repo's")
print("  phase2/2a_validation/results/ directory.")
