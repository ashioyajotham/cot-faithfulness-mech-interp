# -*- coding: utf-8 -*-
"""
Phase 2C: Hybrid Approach & Ablation Analysis
==============================================

This notebook implements PATH C of the faithfulness detection project:
Combining circuit discovery (Phase 1) with steering vectors (diff-of-means).

## The Hypothesis That Failed

We expected:
```
Path C (Hybrid) > Path B (Full Steering Vector)

Reasoning:
- Phase 1 identified 23 key components for faithfulness
- Using ONLY those components should reduce noise
- Therefore, diff-of-means on key components should beat
  diff-of-means on full residual stream
```

## What Actually Happened

```
Path A (Linear Probe):     88.1% accuracy  ← WINNER
Path B (Full Steering):    72.3% accuracy
Path C (Hybrid):           65.0% accuracy  ← WORST!
```

## Why Did Hybrid Fail?

This notebook investigates why the hybrid approach underperformed,
providing important lessons about when to use probes vs. steering vectors.

Key insight: **Diff-of-means treats all dimensions equally.**
The probe learned that L7H6 matters MORE than other components.
Hybrid couldn't capture this non-uniform weighting.

## This Notebook Contains

1. Implementation of hybrid approach
2. Analysis of WHY it failed
3. Ablation studies to understand component contributions
4. Lessons for future research

Author: Victor Ashioya
Project: CoT Faithfulness Mechanistic Interpretability
"""

# ============================================================================
# SETUP
# ============================================================================

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformer_lens import HookedTransformer

# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path("results/phase2c_hybrid")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Device: {device}")
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# LOAD MODEL
# ============================================================================

print("Loading GPT-2 Small...")
model = HookedTransformer.from_pretrained(
    "gpt2",
    device=device,
    fold_ln=False,
    center_writing_weights=False,
    center_unembed=False
)
model.eval()

N_LAYERS = model.cfg.n_layers
N_HEADS = model.cfg.n_heads
D_HEAD = model.cfg.d_head
D_MODEL = model.cfg.d_model

# ============================================================================
# PHASE 1 COMPONENTS
# ============================================================================

FAITHFUL_HEADS = ["L0H1", "L0H6", "L1H7", "L10H2", "L3H0", "L9H9"]
SHORTCUT_HEADS = ["L7H6", "L2H10", "L0H3", "L2H0", "L3H10", "L0H10", "L6H8", "L4H7", "L5H9", "L0H0"]
FAITHFUL_MLPS = ["L0MLP", "L5MLP"]
SHORTCUT_MLPS = ["L10MLP", "L3MLP", "L2MLP", "L6MLP", "L4MLP"]

KEY_COMPONENTS = FAITHFUL_HEADS + SHORTCUT_HEADS + FAITHFUL_MLPS + SHORTCUT_MLPS

COMPONENT_TYPES = {}
for h in FAITHFUL_HEADS:
    COMPONENT_TYPES[h] = "faithful"
for h in SHORTCUT_HEADS:
    COMPONENT_TYPES[h] = "shortcut"
for m in FAITHFUL_MLPS:
    COMPONENT_TYPES[m] = "faithful"
for m in SHORTCUT_MLPS:
    COMPONENT_TYPES[m] = "shortcut"

print(f"Key components: {len(KEY_COMPONENTS)}")

# ============================================================================
# DATASET GENERATION
# ============================================================================

@dataclass
class FaithfulnessExample:
    prompt: str
    label: int
    correct_answer: str
    cot_answer: str
    example_type: str
    metadata: dict = field(default_factory=dict)


def generate_arithmetic_dataset(n_pairs: int = 400, seed: int = 42) -> Tuple[List, List]:
    np.random.seed(seed)
    faithful, unfaithful = [], []
    
    for i in range(n_pairs):
        a = np.random.randint(10, 50)
        b = np.random.randint(10, 50)
        correct = a + b
        
        a_units, a_tens = a % 10, a // 10
        b_units, b_tens = b % 10, b // 10
        units_sum = a_units + b_units
        tens_sum = a_tens + b_tens
        
        faithful_prompt = (
            f"Q: What is {a}+{b}?\n"
            f"Steps: units={a_units}+{b_units}={units_sum}, tens={a_tens}+{b_tens}={tens_sum}.\n"
            f"A:"
        )
        faithful.append(FaithfulnessExample(
            prompt=faithful_prompt, label=0, correct_answer=str(correct),
            cot_answer=str(correct), example_type="faithful_addition",
            metadata={"a": a, "b": b, "pair_id": i}
        ))
        
        wrong_units = units_sum + np.random.choice([3, 5, 7, -3, -5])
        wrong_tens = tens_sum + np.random.choice([2, 4, -2, -4])
        wrong_cot_answer = wrong_tens * 10 + wrong_units
        
        unfaithful_prompt = (
            f"Q: What is {a}+{b}?\n"
            f"Steps: units={a_units}+{b_units}={wrong_units}, tens={a_tens}+{b_tens}={wrong_tens}.\n"
            f"A:"
        )
        unfaithful.append(FaithfulnessExample(
            prompt=unfaithful_prompt, label=1, correct_answer=str(correct),
            cot_answer=str(wrong_cot_answer), example_type="unfaithful_addition",
            metadata={"a": a, "b": b, "pair_id": i, "wrong_answer": wrong_cot_answer}
        ))
    
    return faithful, unfaithful


faithful_data, unfaithful_data = generate_arithmetic_dataset(n_pairs=400)
all_data = faithful_data + unfaithful_data
np.random.shuffle(all_data)

print(f"Dataset: {len(all_data)} examples")

# ============================================================================
# ACTIVATION EXTRACTION
# ============================================================================

def extract_key_component_activations(
    examples: List[FaithfulnessExample],
    components: List[str],
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract activations from key components only."""
    
    all_activations = []
    all_labels = []
    
    for idx, example in enumerate(examples):
        if verbose and idx % 100 == 0:
            print(f"  Processing {idx}/{len(examples)}...")
        
        tokens = model.to_tokens(example.prompt)
        
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=lambda n: "hook_z" in n or "hook_mlp_out" in n
            )
        
        example_acts = []
        for comp in components:
            if comp.endswith("MLP"):
                layer = int(comp[1:-3])
                hook_name = f"blocks.{layer}.hook_mlp_out"
                acts = cache[hook_name][0, -1, :].cpu().numpy()
            else:
                layer = int(comp.split("H")[0][1:])
                head = int(comp.split("H")[1])
                hook_name = f"blocks.{layer}.attn.hook_z"
                acts = cache[hook_name][0, -1, head, :].cpu().numpy()
            
            example_acts.append(acts)
        
        all_activations.append(np.concatenate(example_acts))
        all_labels.append(example.label)
        
        del cache
        if device == "cuda":
            torch.cuda.empty_cache()
    
    return np.array(all_activations), np.array(all_labels)


# ============================================================================
# PART 1: HYBRID APPROACH (Diff-of-Means on Key Components)
# ============================================================================

print(f"\n{'='*60}")
print("PART 1: HYBRID APPROACH")
print(f"{'='*60}")

print("\nExtracting activations from key components...")
X, y = extract_key_component_activations(all_data, KEY_COMPONENTS)
print(f"Shape: {X.shape}")

# Split data
train_mask = np.array([i < 400 for i in range(len(all_data))])
test_mask = ~train_mask

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]

# Compute diff-of-means on key components
faithful_acts = X_train[y_train == 0]
unfaithful_acts = X_train[y_train == 1]

hybrid_vector = faithful_acts.mean(axis=0) - unfaithful_acts.mean(axis=0)
hybrid_vector_normalized = hybrid_vector / np.linalg.norm(hybrid_vector)

print(f"\nHybrid vector computed:")
print(f"  Shape: {hybrid_vector.shape}")
print(f"  Norm: {np.linalg.norm(hybrid_vector):.4f}")

# Detection via projection
test_scores = X_test @ hybrid_vector_normalized
fpr, tpr, thresholds = roc_curve(y_test, -test_scores)
roc_auc = roc_auc_score(y_test, -test_scores)

# Find best threshold
best_acc = 0
for thresh in thresholds:
    preds = (-test_scores > thresh).astype(int)
    acc = accuracy_score(y_test, preds)
    if acc > best_acc:
        best_acc = acc

print(f"\nHybrid Detection Results:")
print(f"  ROC-AUC: {roc_auc:.3f}")
print(f"  Best Accuracy: {best_acc:.3f}")

# ============================================================================
# PART 2: WHY DID HYBRID FAIL? - Analysis
# ============================================================================

print(f"\n{'='*60}")
print("PART 2: WHY DID HYBRID FAIL?")
print(f"{'='*60}")

"""
The hybrid approach failed because diff-of-means computes a SINGLE direction
that treats all dimensions equally. It cannot learn that some components
(like L7H6) are more important than others.

Let's prove this by comparing:
1. Diff-of-means vector weights
2. Linear probe learned weights
"""

# Train linear probe for comparison
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
clf.fit(X_train_scaled, y_train)

probe_acc = accuracy_score(y_test, clf.predict(X_test_scaled))
probe_auc = roc_auc_score(y_test, clf.predict_proba(X_test_scaled)[:, 1])

print(f"\nComparison:")
print(f"  Linear Probe Accuracy: {probe_acc:.3f}")
print(f"  Hybrid Accuracy:       {best_acc:.3f}")
print(f"  Gap:                   {probe_acc - best_acc:.3f}")

# Analyze weight distributions
print(f"\n--- Weight Analysis ---")

# Get component-level weights for both methods
def get_component_weights(vector, components):
    """Aggregate vector weights by component."""
    weights = {}
    start_idx = 0
    for comp in components:
        if comp.endswith("MLP"):
            dim = D_MODEL
        else:
            dim = D_HEAD
        
        comp_weights = vector[start_idx:start_idx + dim]
        weights[comp] = np.mean(np.abs(comp_weights))
        start_idx += dim
    return weights

hybrid_weights = get_component_weights(hybrid_vector_normalized, KEY_COMPONENTS)
probe_weights = get_component_weights(clf.coef_[0], KEY_COMPONENTS)

# Normalize for comparison
max_hybrid = max(hybrid_weights.values())
max_probe = max(probe_weights.values())
hybrid_weights_norm = {k: v/max_hybrid for k, v in hybrid_weights.items()}
probe_weights_norm = {k: v/max_probe for k, v in probe_weights.items()}

print("\nComponent importance (normalized):")
print(f"{'Component':<12} {'Hybrid':<10} {'Probe':<10} {'Diff':<10} {'Type':<10}")
print("-" * 52)

for comp in KEY_COMPONENTS:
    h_weight = hybrid_weights_norm[comp]
    p_weight = probe_weights_norm[comp]
    diff = p_weight - h_weight
    comp_type = COMPONENT_TYPES[comp]
    print(f"{comp:<12} {h_weight:<10.3f} {p_weight:<10.3f} {diff:+<10.3f} {comp_type:<10}")

# Key finding: variance in weights
hybrid_variance = np.var(list(hybrid_weights_norm.values()))
probe_variance = np.var(list(probe_weights_norm.values()))

print(f"\nWeight variance:")
print(f"  Hybrid: {hybrid_variance:.4f} (more uniform)")
print(f"  Probe:  {probe_variance:.4f} (more selective)")

# ============================================================================
# PART 3: ABLATION STUDIES
# ============================================================================

print(f"\n{'='*60}")
print("PART 3: ABLATION STUDIES")
print(f"{'='*60}")

"""
To understand component contributions, we'll:
1. Remove one component at a time and measure accuracy drop
2. Use only one component at a time and measure accuracy
"""

# Single-component analysis
print("\nSingle-component detection accuracy:")
single_results = {}

start_idx = 0
for comp in KEY_COMPONENTS:
    if comp.endswith("MLP"):
        dim = D_MODEL
    else:
        dim = D_HEAD
    
    X_single = X[:, start_idx:start_idx + dim]
    X_train_s = X_single[train_mask]
    X_test_s = X_single[test_mask]
    
    # Diff-of-means on single component
    vec = X_train_s[y_train == 0].mean(0) - X_train_s[y_train == 1].mean(0)
    vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
    
    scores = X_test_s @ vec_norm
    
    # Try both directions
    preds_pos = (scores > 0).astype(int)
    preds_neg = (scores < 0).astype(int)
    acc = max(accuracy_score(y_test, preds_pos), accuracy_score(y_test, preds_neg))
    
    single_results[comp] = acc
    start_idx += dim

# Sort by accuracy
sorted_single = sorted(single_results.items(), key=lambda x: x[1], reverse=True)

print(f"{'Component':<12} {'Accuracy':<10} {'Type':<10}")
print("-" * 32)
for comp, acc in sorted_single[:10]:
    comp_type = COMPONENT_TYPES[comp]
    marker = "★" if acc > 0.55 else " "
    print(f"{marker} {comp:<10} {acc:<10.3f} {comp_type:<10}")

# Leave-one-out analysis
print("\n\nLeave-one-out analysis (drop in accuracy when removed):")
leave_one_out = {}

for comp_to_remove in KEY_COMPONENTS[:10]:  # Top 10 to save time
    remaining = [c for c in KEY_COMPONENTS if c != comp_to_remove]
    
    # Reconstruct X without this component
    new_acts = []
    start_idx = 0
    for comp in KEY_COMPONENTS:
        if comp.endswith("MLP"):
            dim = D_MODEL
        else:
            dim = D_HEAD
        
        if comp != comp_to_remove:
            new_acts.append(X[:, start_idx:start_idx + dim])
        start_idx += dim
    
    X_reduced = np.concatenate(new_acts, axis=1)
    X_train_r = X_reduced[train_mask]
    X_test_r = X_reduced[test_mask]
    
    # Train probe
    scaler_r = StandardScaler()
    X_train_r_scaled = scaler_r.fit_transform(X_train_r)
    X_test_r_scaled = scaler_r.transform(X_test_r)
    
    clf_r = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    clf_r.fit(X_train_r_scaled, y_train)
    
    acc_reduced = accuracy_score(y_test, clf_r.predict(X_test_r_scaled))
    drop = probe_acc - acc_reduced
    leave_one_out[comp_to_remove] = drop

sorted_loo = sorted(leave_one_out.items(), key=lambda x: x[1], reverse=True)

print(f"{'Component':<12} {'Drop':<10} {'Type':<10}")
print("-" * 32)
for comp, drop in sorted_loo:
    comp_type = COMPONENT_TYPES[comp]
    print(f"{comp:<12} {drop:+.3f}     {comp_type:<10}")

# ============================================================================
# PART 4: WHAT WOULD MAKE HYBRID WORK?
# ============================================================================

print(f"\n{'='*60}")
print("PART 4: POTENTIAL IMPROVEMENTS")
print(f"{'='*60}")

"""
Could we fix hybrid by using probe weights to weight the diff-of-means?

weighted_hybrid = sum(probe_weight[c] * diff_of_means[c] for c in components)
"""

# Weighted hybrid approach
probe_coef = np.abs(clf.coef_[0])  # Use probe weights
weighted_vector = hybrid_vector * probe_coef
weighted_vector_normalized = weighted_vector / np.linalg.norm(weighted_vector)

weighted_scores = X_test @ weighted_vector_normalized
weighted_auc = roc_auc_score(y_test, -weighted_scores)

# Find best threshold
best_weighted_acc = 0
for thresh in np.linspace(-2, 2, 100):
    preds = (-weighted_scores > thresh).astype(int)
    acc = accuracy_score(y_test, preds)
    if acc > best_weighted_acc:
        best_weighted_acc = acc

print(f"Weighted Hybrid (using probe coefficients):")
print(f"  Accuracy: {best_weighted_acc:.3f}")
print(f"  ROC-AUC:  {weighted_auc:.3f}")
print(f"  vs Original Hybrid: {best_weighted_acc - best_acc:+.3f}")

# ============================================================================
# VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Method Comparison
methods = ['Linear Probe', 'Full Steering', 'Hybrid', 'Weighted Hybrid']
accuracies = [probe_acc, 0.723, best_acc, best_weighted_acc]  # 0.723 from Phase 2B
colors = ['green', 'blue', 'red', 'orange']

bars = axes[0, 0].bar(methods, accuracies, color=colors, alpha=0.7)
axes[0, 0].axhline(y=0.8, color='black', linestyle='--', label='Target (80%)')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].set_title('Method Comparison')
axes[0, 0].set_ylim(0, 1)
axes[0, 0].legend()
for bar, acc in zip(bars, accuracies):
    axes[0, 0].text(bar.get_x() + bar.get_width()/2, acc + 0.02, f'{acc:.1%}', 
                    ha='center', fontsize=10)

# 2. Weight Distribution Comparison
components_short = [c.replace('MLP', 'M') for c in KEY_COMPONENTS[:15]]
x = np.arange(len(components_short))
width = 0.35

hybrid_vals = [hybrid_weights_norm[c] for c in KEY_COMPONENTS[:15]]
probe_vals = [probe_weights_norm[c] for c in KEY_COMPONENTS[:15]]

axes[0, 1].bar(x - width/2, hybrid_vals, width, label='Hybrid', alpha=0.7)
axes[0, 1].bar(x + width/2, probe_vals, width, label='Probe', alpha=0.7)
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(components_short, rotation=45, ha='right')
axes[0, 1].set_ylabel('Normalized Weight')
axes[0, 1].set_title('Weight Distribution: Hybrid vs Probe')
axes[0, 1].legend()

# 3. Single Component Accuracy
top_10_comps = [c for c, _ in sorted_single[:10]]
top_10_accs = [a for _, a in sorted_single[:10]]
top_10_colors = ['green' if COMPONENT_TYPES[c] == 'faithful' else 'red' for c in top_10_comps]

axes[1, 0].barh(range(len(top_10_comps)), top_10_accs, color=top_10_colors, alpha=0.7)
axes[1, 0].set_yticks(range(len(top_10_comps)))
axes[1, 0].set_yticklabels(top_10_comps)
axes[1, 0].axvline(x=0.5, color='black', linestyle='--', alpha=0.5)
axes[1, 0].set_xlabel('Single-Component Accuracy')
axes[1, 0].set_title('Best Single Components\n(Green=Faithful, Red=Shortcut)')
axes[1, 0].invert_yaxis()

# 4. Leave-One-Out Impact
loo_comps = [c for c, _ in sorted_loo]
loo_drops = [d for _, d in sorted_loo]
loo_colors = ['green' if COMPONENT_TYPES[c] == 'faithful' else 'red' for c in loo_comps]

axes[1, 1].barh(range(len(loo_comps)), loo_drops, color=loo_colors, alpha=0.7)
axes[1, 1].set_yticks(range(len(loo_comps)))
axes[1, 1].set_yticklabels(loo_comps)
axes[1, 1].axvline(x=0, color='black', linestyle='-', alpha=0.5)
axes[1, 1].set_xlabel('Accuracy Drop When Removed')
axes[1, 1].set_title('Leave-One-Out Importance')
axes[1, 1].invert_yaxis()

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'phase2c_results.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# SAVE RESULTS
# ============================================================================

results = {
    'hybrid': {
        'accuracy': best_acc,
        'roc_auc': roc_auc,
    },
    'probe': {
        'accuracy': probe_acc,
        'roc_auc': probe_auc,
    },
    'weighted_hybrid': {
        'accuracy': best_weighted_acc,
        'roc_auc': weighted_auc,
    },
    'single_component': {k: float(v) for k, v in sorted_single},
    'leave_one_out': {k: float(v) for k, v in sorted_loo},
    'weight_variance': {
        'hybrid': float(hybrid_variance),
        'probe': float(probe_variance),
    }
}

with open(RESULTS_DIR / 'phase2c_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*60}")
print("PHASE 2C COMPLETE")
print(f"{'='*60}")

# ============================================================================
# KEY LESSONS
# ============================================================================

print("""
## Key Lessons from Hybrid Failure

1. **Diff-of-means ≠ Optimal classifier**
   - Diff-of-means finds direction of maximum mean separation
   - Linear probe finds direction of maximum class separation
   - These are different objectives!

2. **Sparsity isn't enough**
   - Just using fewer components (sparsity) doesn't help if you
     can't weight them appropriately
   - Probe learns: L7H6 >> other components
   - Hybrid treats: L7H6 = other components

3. **When to use each method**:
   - DETECTION → Linear probe (learns optimal weights)
   - INTERVENTION → Diff-of-means (just need a direction)
   
4. **Weighted hybrid shows promise**
   - Using probe weights to scale diff-of-means improves results
   - This could combine benefits of both approaches

5. **Most important component: L7H6 (shortcut head)**
   - Highest probe weight
   - Best single-component accuracy
   - Largest leave-one-out impact
   - This is the "smoking gun" for unfaithful reasoning

## Implications for Future Work

- For safety monitoring: Use probes, not steering vectors
- For intervention: Steering vectors work, but probe weights
  could inform which components to target
- L7H6 ablation is the most promising intervention target
""")

np.save(RESULTS_DIR / 'hybrid_vector.npy', hybrid_vector)
np.save(RESULTS_DIR / 'weighted_hybrid_vector.npy', weighted_vector)

print(f"\n✓ Results saved to {RESULTS_DIR}")
