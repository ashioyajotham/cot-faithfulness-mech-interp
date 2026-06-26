"""
Experiment 2B-2: Detection Probe & Dual-Metric Analysis

Trains a logistic regression probe on Qwen circuit activations to
classify faithful vs unfaithful CoT. Then:
  1. Computes dual metrics (probe coefficients vs restoration scores)
  2. Runs Hewitt-Liang selectivity test
  3. Tests distributed signal hypothesis (random-layer baseline)

Usage::

    python phase2/2b_scaling/experiments/detection_probe.py \\
        --model qwen25-math-1.5b --device auto

Author: Victor Ashioya
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load model_registry via importlib (2b_scaling has digit prefix)
_spec = importlib.util.spec_from_file_location(
    "model_registry",
    str(Path(__file__).resolve().parent.parent / "src" / "model_registry.py"),
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["model_registry"] = _mod  # Required for @dataclass to resolve
_spec.loader.exec_module(_mod)
load_model = _mod.load_model


# ── Activation extraction ─────────────────────────────────────────────

def extract_circuit_activations(
    model,
    pairs: List[dict],
    circuit_components: List[str],
    position: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract activations from circuit components for all pairs.

    Returns (X, y) where X is [n_pairs, n_features] and y is [n_pairs].
    """
    all_features = []
    all_labels = []

    for i, pair in enumerate(pairs):
        if i % 50 == 0:
            print(f"  Extracting activations: {i}/{len(pairs)}")

        # Run on unfaithful prompt (where the model decision matters)
        tokens = model.to_tokens(pair["unfaithful_prompt"])

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)

        # Extract features from each circuit component
        features = []
        for comp in circuit_components:
            if comp.startswith("L") and "H" in comp:
                # Head component: L{layer}H{head}
                parts = comp[1:].split("H")
                layer, head = int(parts[0]), int(parts[1])
                hook_name = f"blocks.{layer}.attn.hook_z"
                act = cache[hook_name][0, position, head, :].cpu().numpy()
            elif "hook_mlp_out" in comp:
                layer = int(comp.split(".")[1])
                act = cache[comp][0, position, :].cpu().numpy()
            elif "hook_attn_out" in comp:
                layer = int(comp.split(".")[1])
                act = cache[comp][0, position, :].cpu().numpy()
            else:
                continue
            features.append(act)

        if features:
            all_features.append(np.concatenate(features))
            all_labels.append(pair["label"])

    X = np.array(all_features)
    y = np.array(all_labels)
    print(f"  Feature matrix: {X.shape}, Labels: {y.shape}")
    print(f"  Class balance: {np.mean(y):.2f} (1=unfaithful)")
    return X, y


def extract_layer_activations(
    model,
    pairs: List[dict],
    layer: int,
    position: int = -1,
) -> np.ndarray:
    """Extract full residual stream activations from a single layer."""
    all_features = []

    for i, pair in enumerate(pairs):
        tokens = model.to_tokens(pair["unfaithful_prompt"])
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)

        hook_name = f"blocks.{layer}.hook_resid_post"
        act = cache[hook_name][0, position, :].cpu().numpy()
        all_features.append(act)

    return np.array(all_features)


# ── Probe training ────────────────────────────────────────────────────

def train_probe(X: np.ndarray, y: np.ndarray, n_splits: int = 5):
    """Train logistic regression probe with cross-validation.

    Returns dict with accuracy, AUC, per-fold results, and fitted probe.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        probe = LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs", random_state=42
        )
        probe.fit(X_train, y_train)

        y_pred = probe.predict(X_test)
        y_prob = probe.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = 0.5

        fold_results.append({"fold": fold, "accuracy": acc, "auc": auc})

    # Train final probe on all data for coefficient analysis
    final_probe = LogisticRegression(
        max_iter=1000, C=1.0, solver="lbfgs", random_state=42
    )
    final_probe.fit(X, y)

    mean_acc = np.mean([r["accuracy"] for r in fold_results])
    mean_auc = np.mean([r["auc"] for r in fold_results])

    return {
        "mean_accuracy": float(mean_acc),
        "mean_auc": float(mean_auc),
        "fold_results": fold_results,
        "probe": final_probe,
    }


def train_mlp_probe(X: np.ndarray, y: np.ndarray, n_splits: int = 5):
    """Train 2-layer MLP probe with cross-validation.

    Tests whether faithfulness signal is non-linearly encoded.
    Returns dict with accuracy, AUC, and per-fold results.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Standardize features for MLP stability
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        mlp = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=42,
            learning_rate_init=0.001,
        )
        mlp.fit(X_train_s, y_train)

        y_pred = mlp.predict(X_test_s)
        y_prob = mlp.predict_proba(X_test_s)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = 0.5

        fold_results.append({"fold": fold, "accuracy": acc, "auc": auc})

    mean_acc = np.mean([r["accuracy"] for r in fold_results])
    mean_auc = np.mean([r["auc"] for r in fold_results])

    return {
        "mean_accuracy": float(mean_acc),
        "mean_auc": float(mean_auc),
        "fold_results": fold_results,
    }


def extract_multi_layer_activations(
    model,
    pairs: List[dict],
    layers: List[int],
    position: int = -1,
) -> np.ndarray:
    """Extract residual stream activations from multiple layers.

    Concatenates `hook_resid_post` from each specified layer at the
    given position.  Returns [n_pairs, n_layers * d_model].
    """
    all_features = []
    hook_names = [f"blocks.{l}.hook_resid_post" for l in layers]

    for i, pair in enumerate(pairs):
        if i % 50 == 0:
            print(f"  Extracting multi-layer activations: {i}/{len(pairs)}")
        tokens = model.to_tokens(pair["unfaithful_prompt"])
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens, names_filter=lambda n: n in hook_names
            )
        feats = []
        for hn in hook_names:
            feats.append(cache[hn][0, position, :].cpu().numpy())
        all_features.append(np.concatenate(feats))
        del cache

    return np.array(all_features)


# ── Dual-metric analysis ─────────────────────────────────────────────

def compute_probe_importance(
    probe: LogisticRegression,
    circuit_components: List[str],
    model_cfg,
) -> Dict[str, float]:
    """Compute per-component importance from probe coefficients.

    Mirrors Phase 1 methodology: mean(|coef|) per component.
    """
    coef = probe.coef_[0]  # Shape: [n_features]
    d_head = model_cfg.d_head
    d_model = model_cfg.d_model

    importance = {}
    idx = 0
    for comp in circuit_components:
        if comp.startswith("L") and "H" in comp:
            dim = d_head
        else:
            dim = d_model
        comp_coef = coef[idx:idx + dim]
        importance[comp] = float(np.mean(np.abs(comp_coef)))
        idx += dim

    return importance


# ── Selectivity test ──────────────────────────────────────────────────

def run_selectivity_test(
    X: np.ndarray,
    y: np.ndarray,
    n_control_runs: int = 10,
) -> Dict:
    """Hewitt-Liang selectivity: compare probe on real vs scrambled labels.

    Selectivity = linguistic_accuracy - control_accuracy.
    """
    # Linguistic task accuracy
    linguistic = train_probe(X, y)

    # Control task: scrambled labels
    control_accs = []
    rng = np.random.RandomState(42)
    for run in range(n_control_runs):
        y_scrambled = rng.permutation(y)
        ctrl = train_probe(X, y_scrambled)
        control_accs.append(ctrl["mean_accuracy"])

    mean_control = float(np.mean(control_accs))
    selectivity = linguistic["mean_accuracy"] - mean_control

    return {
        "linguistic_accuracy": linguistic["mean_accuracy"],
        "linguistic_auc": linguistic["mean_auc"],
        "control_accuracy_mean": mean_control,
        "control_accuracy_std": float(np.std(control_accs)),
        "selectivity": float(selectivity),
        "pass": selectivity > 0,
    }


# ── Distributed signal test ──────────────────────────────────────────

def run_distributed_signal_test(
    model,
    pairs: List[dict],
    y: np.ndarray,
    test_layers: List[int],
) -> Dict:
    """Test if random non-circuit layers carry faithfulness signal.

    Trains probes on full residual-stream activations from arbitrary layers.
    """
    results = {}

    for layer in test_layers:
        print(f"  Testing layer {layer}...")
        X_layer = extract_layer_activations(model, pairs, layer)
        probe_result = train_probe(X_layer, y)
        results[f"layer_{layer}"] = {
            "accuracy": probe_result["mean_accuracy"],
            "auc": probe_result["mean_auc"],
        }

    return results


# ── Main ──────────────────────────────────────────────────────────────

def run_detection_probe(
    model_key: str = "qwen25-math-1.5b",
    dataset_path: str | None = None,
    circuit_path: str | None = None,
    device: str = "auto",
    top_k_components: int = 15,
    output_dir: str | None = None,
):
    """Full detection probe pipeline."""

    results_base = Path(__file__).resolve().parent.parent / "results"
    if output_dir:
        results_base = Path(output_dir)

    if dataset_path is None:
        dataset_path = str(results_base / "dataset.json")
    if circuit_path is None:
        circuit_path = str(results_base / "circuit_discovery_results.json")

    print("=" * 70)
    print("PHASE 2B-2: DETECTION PROBE & DUAL-METRIC ANALYSIS")
    print("=" * 70)

    # Load data
    with open(dataset_path) as f:
        dataset = json.load(f)
    with open(circuit_path) as f:
        circuit = json.load(f)

    valid_pairs = [p for p in dataset if p.get("is_valid") and p.get("label") in (0, 1)]
    print(f"\n  Valid pairs: {len(valid_pairs)}")

    # Get top circuit components by |restoration score|
    comp_stats = circuit["component_stats"]
    head_comps = {k: v for k, v in comp_stats.items() if "hook_" not in k}
    sorted_comps = sorted(head_comps.items(), key=lambda x: abs(x[1]["mean"]), reverse=True)
    circuit_components = [comp for comp, _ in sorted_comps[:top_k_components]]
    print(f"  Circuit components ({len(circuit_components)}): {circuit_components}")

    # Load model
    from transformer_lens import HookedTransformer
    if isinstance(model_key, HookedTransformer):
        model = model_key
        model_key = getattr(model.cfg, "model_name", "qwen25-math-1.5b")
    else:
        print(f"\n  Loading {model_key}...")
        model = load_model(model_key, device=device)

    # ── Step 1: Extract circuit activations ───────────────────────────
    print(f"\n{'='*60}")
    print("STEP 1: EXTRACT CIRCUIT ACTIVATIONS")
    print(f"{'='*60}")

    X, y = extract_circuit_activations(model, valid_pairs, circuit_components)

    # ── Step 2: Train probe ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 2: TRAIN DETECTION PROBE")
    print(f"{'='*60}")

    probe_result = train_probe(X, y)
    print(f"\n  Cross-val accuracy: {probe_result['mean_accuracy']:.3f}")
    print(f"  Cross-val AUC:      {probe_result['mean_auc']:.3f}")

    # ── Step 3: Dual-metric analysis ─────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 3: DUAL-METRIC ANALYSIS")
    print(f"{'='*60}")

    # Probe coefficient importance
    probe_importance = compute_probe_importance(
        probe_result["probe"], circuit_components, model.cfg
    )

    # Restoration score importance (from circuit discovery)
    restoration_importance = {
        comp: abs(comp_stats[comp]["mean"]) for comp in circuit_components
    }

    # Compare rankings
    probe_ranked = sorted(probe_importance.items(), key=lambda x: x[1], reverse=True)
    restoration_ranked = sorted(restoration_importance.items(), key=lambda x: x[1], reverse=True)

    print(f"\n  {'Rank':<6} {'Probe Top':>15} {'Coef':>8}  |  {'Restoration Top':>15} {'Score':>8}")
    print(f"  {'-'*60}")
    for i in range(min(10, len(probe_ranked))):
        p_comp, p_val = probe_ranked[i]
        r_comp, r_val = restoration_ranked[i]
        print(f"  {i+1:<6} {p_comp:>15} {p_val:>8.4f}  |  {r_comp:>15} {r_val:>8.4f}")

    # Check if #1 diverges (the dual-metric finding)
    top_probe = probe_ranked[0][0]
    top_restoration = restoration_ranked[0][0]
    metrics_diverge = top_probe != top_restoration
    print(f"\n  Top probe component:       {top_probe}")
    print(f"  Top restoration component: {top_restoration}")
    print(f"  Dual-metric divergence:    {'YES (replicates Phase 2A)' if metrics_diverge else 'NO (same component)'}")

    # ── Step 4: Selectivity test ─────────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 4: HEWITT-LIANG SELECTIVITY")
    print(f"{'='*60}")

    selectivity = run_selectivity_test(X, y)
    print(f"\n  Linguistic accuracy: {selectivity['linguistic_accuracy']:.3f}")
    print(f"  Control accuracy:    {selectivity['control_accuracy_mean']:.3f} +/- {selectivity['control_accuracy_std']:.3f}")
    print(f"  Selectivity:         {selectivity['selectivity']:.3f}")
    print(f"  PASS:                {selectivity['pass']}")

    # ── Step 5: Distributed signal test ──────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 5: DISTRIBUTED SIGNAL TEST")
    print(f"{'='*60}")

    n_layers = model.cfg.n_layers
    # Test 3 random non-circuit layers + the middle layer
    test_layers = sorted(set([
        n_layers // 4,
        n_layers // 2,
        3 * n_layers // 4,
    ]))
    print(f"  Testing layers: {test_layers}")

    distributed = run_distributed_signal_test(model, valid_pairs, y, test_layers)

    print(f"\n  Circuit probe accuracy:  {probe_result['mean_accuracy']:.3f}")
    for layer_key, stats in distributed.items():
        print(f"  {layer_key} accuracy:     {stats['accuracy']:.3f}")

    circuit_acc = probe_result["mean_accuracy"]
    best_layer_acc = max(s["accuracy"] for s in distributed.values())
    gap = circuit_acc - best_layer_acc
    print(f"\n  Circuit vs best random layer gap: {gap:.3f}")
    print(f"  Signal is {'LOCALIZED' if gap > 0.05 else 'DISTRIBUTED'}")

    # ── Step 6: MLP probe on circuit activations ─────────────────────
    print(f"\n{'='*60}")
    print("STEP 6: MLP PROBE (NON-LINEAR) ON CIRCUIT ACTIVATIONS")
    print(f"{'='*60}")

    mlp_result = train_mlp_probe(X, y)
    print(f"\n  MLP cross-val accuracy: {mlp_result['mean_accuracy']:.3f}")
    print(f"  MLP cross-val AUC:      {mlp_result['mean_auc']:.3f}")
    print(f"  Linear AUC:             {probe_result['mean_auc']:.3f}")
    mlp_gain = mlp_result['mean_auc'] - probe_result['mean_auc']
    print(f"  MLP gain over linear:   {mlp_gain:+.3f}")
    if mlp_gain > 0.05:
        print("  => Signal IS non-linearly encoded in circuit heads")
    else:
        print("  => No significant non-linear gain")

    # ── Step 7: Full-stream probe (multi-layer residual) ─────────────
    print(f"\n{'='*60}")
    print("STEP 7: FULL-STREAM PROBE (MULTI-LAYER RESIDUAL)")
    print(f"{'='*60}")

    # Use 4 strategically spaced layers for the full-stream probe
    stream_layers = sorted(set([
        0,                    # input encoding
        n_layers // 3,        # early processing
        2 * n_layers // 3,    # mid-late processing
        n_layers - 1,         # final layer
    ]))
    print(f"  Layers for full-stream probe: {stream_layers}")
    print(f"  Feature dim: {len(stream_layers)} × {model.cfg.d_model} = {len(stream_layers) * model.cfg.d_model}")

    X_stream = extract_multi_layer_activations(model, valid_pairs, stream_layers)
    print(f"  Full-stream feature matrix: {X_stream.shape}")

    stream_linear = train_probe(X_stream, y)
    stream_mlp = train_mlp_probe(X_stream, y)

    print(f"\n  Full-stream linear AUC:  {stream_linear['mean_auc']:.3f}")
    print(f"  Full-stream MLP AUC:     {stream_mlp['mean_auc']:.3f}")
    print(f"  Circuit linear AUC:      {probe_result['mean_auc']:.3f}")
    print(f"  Circuit MLP AUC:         {mlp_result['mean_auc']:.3f}")

    full_stream_results = {
        "layers": stream_layers,
        "feature_dim": int(X_stream.shape[1]),
        "linear": {
            "accuracy": stream_linear["mean_accuracy"],
            "auc": stream_linear["mean_auc"],
            "fold_results": stream_linear["fold_results"],
        },
        "mlp": {
            "accuracy": stream_mlp["mean_accuracy"],
            "auc": stream_mlp["mean_auc"],
            "fold_results": stream_mlp["fold_results"],
        },
    }

    # ── Save results ─────────────────────────────────────────────────
    output = {
        "model": model_key,
        "n_pairs": len(valid_pairs),
        "circuit_components": circuit_components,
        "probe": {
            "accuracy": probe_result["mean_accuracy"],
            "auc": probe_result["mean_auc"],
            "fold_results": probe_result["fold_results"],
        },
        "mlp_probe": {
            "accuracy": mlp_result["mean_accuracy"],
            "auc": mlp_result["mean_auc"],
            "fold_results": mlp_result["fold_results"],
            "gain_over_linear": float(mlp_gain),
        },
        "full_stream_probe": full_stream_results,
        "dual_metric": {
            "probe_importance": probe_importance,
            "restoration_importance": restoration_importance,
            "top_probe": top_probe,
            "top_restoration": top_restoration,
            "diverges": metrics_diverge,
        },
        "selectivity": selectivity,
        "distributed_signal": distributed,
        "circuit_vs_random_gap": gap,
        "signal_localized": gap > 0.05,
    }

    # Remove non-serializable probe object
    output_path = results_base / "detection_probe_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")

    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 2B-2: Detection Probe")
    parser.add_argument("--model", default="qwen25-math-1.5b")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--circuit", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    run_detection_probe(
        model_key=args.model,
        dataset_path=args.dataset,
        circuit_path=args.circuit,
        device=args.device,
        top_k_components=args.top_k,
        output_dir=args.output_dir,
    )
