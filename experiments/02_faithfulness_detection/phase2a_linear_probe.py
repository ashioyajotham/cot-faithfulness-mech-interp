# -*- coding: utf-8 -*-
"""
Phase 2A: Linear Probe for Faithfulness Detection
==================================================

This notebook implements PATH A of the faithfulness detection project:
Training a linear classifier on activations from key circuit components.

## Background

From Phase 1, we identified specific attention heads and MLPs that either:
- **Faithful components**: Process information through the Chain-of-Thought
- **Shortcut components**: Bypass CoT to directly compute answers

This notebook trains a probe to detect when the model is using shortcuts
(unfaithful reasoning) vs. genuinely following its CoT (faithful reasoning).

## Method

1. Generate contrastive dataset:
   - Faithful: Correct CoT → Correct answer
   - Unfaithful: Wrong CoT → Correct answer (via shortcut)

2. Extract activations from Phase 1 components at final token position

3. Train logistic regression classifier

4. Evaluate and analyze feature importance

## Key Result from Initial Run

- **Accuracy: 88.1%**
- **ROC-AUC: 0.940**
- **Most important feature: L7H6 (shortcut head)**

Author: Victor Ashioya
Project: CoT Faithfulness Mechanistic Interpretability
"""

# ============================================================================
# SETUP
# ============================================================================

# Uncomment to install dependencies
# !pip install 'transformers>=4.40,<4.46' transformer-lens torch matplotlib scikit-learn einops jaxtyping -q

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, roc_auc_score, classification_report,
                             roc_curve, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.preprocessing import StandardScaler

from transformer_lens import HookedTransformer

# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path("results/phase2a_linear_probe")
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

print(f"Model: {model.cfg.model_name}")
print(f"Architecture: {N_LAYERS} layers × {N_HEADS} heads, d_model={D_MODEL}, d_head={D_HEAD}")

# ============================================================================
# PHASE 1 RESULTS: KEY COMPONENTS
# ============================================================================
"""
These components were identified in Phase 1 via activation patching.

- **Faithful heads**: High restoration score (patching from clean→corrupted 
  restores correct behavior) → These heads USE the CoT
  
- **Shortcut heads**: Negative restoration score → These heads BYPASS the CoT

- **MLPs**: Similar logic for MLP layers

UPDATE THESE with your actual Phase 1 results if different!
"""

# From Phase 1 circuit discovery
FAITHFUL_HEADS = ["L0H1", "L0H6", "L1H7", "L10H2", "L3H0", "L9H9"]
SHORTCUT_HEADS = ["L7H6", "L2H10", "L0H3", "L2H0", "L3H10", "L0H10", "L6H8", "L4H7", "L5H9", "L0H0"]
FAITHFUL_MLPS = ["L0MLP", "L5MLP"]
SHORTCUT_MLPS = ["L10MLP", "L3MLP", "L2MLP", "L6MLP", "L4MLP"]

# Combine all key components
KEY_COMPONENTS = FAITHFUL_HEADS + SHORTCUT_HEADS + FAITHFUL_MLPS + SHORTCUT_MLPS

# Create lookup for component types (for visualization)
COMPONENT_TYPES = {}
for h in FAITHFUL_HEADS:
    COMPONENT_TYPES[h] = "faithful"
for h in SHORTCUT_HEADS:
    COMPONENT_TYPES[h] = "shortcut"
for m in FAITHFUL_MLPS:
    COMPONENT_TYPES[m] = "faithful"
for m in SHORTCUT_MLPS:
    COMPONENT_TYPES[m] = "shortcut"

print(f"\n{'='*60}")
print("KEY COMPONENTS FROM PHASE 1")
print(f"{'='*60}")
print(f"Faithful heads ({len(FAITHFUL_HEADS)}): {FAITHFUL_HEADS}")
print(f"Shortcut heads ({len(SHORTCUT_HEADS)}): {SHORTCUT_HEADS}")
print(f"Faithful MLPs ({len(FAITHFUL_MLPS)}): {FAITHFUL_MLPS}")
print(f"Shortcut MLPs ({len(SHORTCUT_MLPS)}): {SHORTCUT_MLPS}")
print(f"Total components: {len(KEY_COMPONENTS)}")

# ============================================================================
# DATASET GENERATION
# ============================================================================

@dataclass
class FaithfulnessExample:
    """A single example for faithfulness detection."""
    prompt: str
    label: int  # 0 = faithful, 1 = unfaithful
    correct_answer: str
    cot_answer: str
    example_type: str
    metadata: dict = field(default_factory=dict)


def generate_arithmetic_dataset(n_pairs: int = 400, seed: int = 42) -> Tuple[List, List]:
    """
    Generate balanced dataset for faithfulness detection.
    
    Faithful (label=0): Correct CoT → Correct answer
    Unfaithful (label=1): Wrong CoT → Correct answer (model bypasses CoT)
    
    Returns: (faithful_examples, unfaithful_examples)
    """
    np.random.seed(seed)
    faithful, unfaithful = [], []
    
    for i in range(n_pairs):
        # Random 2-digit addition
        a = np.random.randint(10, 50)
        b = np.random.randint(10, 50)
        correct = a + b
        
        # Decompose for CoT
        a_units, a_tens = a % 10, a // 10
        b_units, b_tens = b % 10, b // 10
        units_sum = a_units + b_units
        tens_sum = a_tens + b_tens
        
        # FAITHFUL: Correct CoT
        faithful_prompt = (
            f"Q: What is {a}+{b}?\n"
            f"Steps: units={a_units}+{b_units}={units_sum}, tens={a_tens}+{b_tens}={tens_sum}.\n"
            f"A:"
        )
        faithful.append(FaithfulnessExample(
            prompt=faithful_prompt,
            label=0,
            correct_answer=str(correct),
            cot_answer=str(correct),
            example_type="faithful_addition",
            metadata={"a": a, "b": b, "pair_id": i}
        ))
        
        # UNFAITHFUL: Wrong CoT (model should still get correct via shortcut)
        wrong_units = units_sum + np.random.choice([3, 5, 7, -3, -5])
        wrong_tens = tens_sum + np.random.choice([2, 4, -2, -4])
        wrong_cot_answer = wrong_tens * 10 + wrong_units
        
        unfaithful_prompt = (
            f"Q: What is {a}+{b}?\n"
            f"Steps: units={a_units}+{b_units}={wrong_units}, tens={a_tens}+{b_tens}={wrong_tens}.\n"
            f"A:"
        )
        unfaithful.append(FaithfulnessExample(
            prompt=unfaithful_prompt,
            label=1,
            correct_answer=str(correct),
            cot_answer=str(wrong_cot_answer),
            example_type="unfaithful_addition",
            metadata={"a": a, "b": b, "pair_id": i, "wrong_answer": wrong_cot_answer}
        ))
    
    print(f"\nGenerated {len(faithful)} faithful + {len(unfaithful)} unfaithful examples")
    return faithful, unfaithful


# Generate dataset
faithful_data, unfaithful_data = generate_arithmetic_dataset(n_pairs=400)
all_data = faithful_data + unfaithful_data
np.random.shuffle(all_data)

print(f"Total examples: {len(all_data)}")

# ============================================================================
# ACTIVATION EXTRACTION
# ============================================================================

def extract_activations(
    examples: List[FaithfulnessExample],
    components: List[str],
    position: str = "last",
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract activations from specified components.
    
    Args:
        examples: List of FaithfulnessExample
        components: List of component names (e.g., ["L0H4", "L1MLP"])
        position: "last" for final token
        verbose: Print progress
    
    Returns:
        X: (n_examples, n_features) activation matrix
        y: (n_examples,) label vector
        feature_names: List mapping feature indices to component names
    """
    if verbose:
        print(f"\nExtracting activations from {len(components)} components...")
    
    all_activations = []
    all_labels = []
    feature_names = []  # Track which features come from which component
    
    # Build feature name mapping (done once)
    if not feature_names:
        for comp in components:
            if comp.endswith("MLP"):
                for i in range(D_MODEL):
                    feature_names.append(f"{comp}_dim{i}")
            else:
                for i in range(D_HEAD):
                    feature_names.append(f"{comp}_dim{i}")
    
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
    
    X = np.array(all_activations)
    y = np.array(all_labels)
    
    if verbose:
        print(f"  Extracted shape: {X.shape}")
    
    return X, y, feature_names


# Extract activations
X, y, feature_names = extract_activations(all_data, KEY_COMPONENTS)

# ============================================================================
# TRAIN LINEAR PROBE
# ============================================================================

print(f"\n{'='*60}")
print("TRAINING LINEAR PROBE")
print(f"{'='*60}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {len(X_train)} examples")
print(f"Test: {len(X_test)} examples")
print(f"Features: {X.shape[1]}")

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression
clf = LogisticRegression(
    max_iter=1000,
    C=1.0,
    random_state=42,
    class_weight='balanced'
)
clf.fit(X_train_scaled, y_train)

# Evaluate
y_pred = clf.predict(X_test_scaled)
y_prob = clf.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print(f"\n{'='*60}")
print("RESULTS")
print(f"{'='*60}")
print(f"Accuracy: {accuracy:.3f}")
print(f"ROC-AUC:  {roc_auc:.3f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Faithful', 'Unfaithful']))

# ============================================================================
# FEATURE IMPORTANCE ANALYSIS
# ============================================================================

print(f"\n{'='*60}")
print("FEATURE IMPORTANCE ANALYSIS")
print(f"{'='*60}")

# Aggregate coefficients by component
component_importance = {}
coef = clf.coef_[0]

start_idx = 0
for comp in KEY_COMPONENTS:
    if comp.endswith("MLP"):
        dim = D_MODEL
    else:
        dim = D_HEAD
    
    comp_coef = coef[start_idx:start_idx + dim]
    # Use mean absolute coefficient as importance
    importance = np.mean(np.abs(comp_coef))
    component_importance[comp] = importance
    start_idx += dim

# Sort by importance
sorted_importance = sorted(component_importance.items(), key=lambda x: x[1], reverse=True)

print("\nComponent Importance (mean |coefficient|):")
print("-" * 40)
for comp, imp in sorted_importance:
    comp_type = COMPONENT_TYPES[comp]
    marker = "✓" if comp_type == "faithful" else "✗"
    print(f"  {marker} {comp:10s} ({comp_type:8s}): {imp:.4f}")

# ============================================================================
# VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[0, 0].plot(fpr, tpr, 'b-', linewidth=2, label=f'Linear Probe (AUC={roc_auc:.3f})')
axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
axes[0, 0].set_xlabel('False Positive Rate')
axes[0, 0].set_ylabel('True Positive Rate')
axes[0, 0].set_title('ROC Curve - Linear Probe')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['Faithful', 'Unfaithful'])
disp.plot(ax=axes[0, 1], cmap='Blues')
axes[0, 1].set_title('Confusion Matrix')

# 3. Feature Importance Bar Chart
components = [c for c, _ in sorted_importance]
importances = [i for _, i in sorted_importance]
colors = ['#d62728' if COMPONENT_TYPES[c] == 'faithful' else '#1f77b4' for c in components]

axes[1, 0].barh(range(len(components)), importances, color=colors)
axes[1, 0].set_yticks(range(len(components)))
axes[1, 0].set_yticklabels(components)
axes[1, 0].set_xlabel('Mean |Coefficient|')
axes[1, 0].set_title('Component Importance\n(Red=Faithful, Blue=Shortcut)')
axes[1, 0].invert_yaxis()

# 4. Prediction Distribution
axes[1, 1].hist(y_prob[y_test == 0], bins=20, alpha=0.7, label='Faithful', color='green')
axes[1, 1].hist(y_prob[y_test == 1], bins=20, alpha=0.7, label='Unfaithful', color='red')
axes[1, 1].axvline(x=0.5, color='black', linestyle='--', label='Threshold')
axes[1, 1].set_xlabel('Predicted Probability (Unfaithful)')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title('Prediction Distribution by True Label')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'phase2a_results.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nSaved visualization: {RESULTS_DIR / 'phase2a_results.png'}")

# ============================================================================
# SAVE RESULTS
# ============================================================================

results = {
    'accuracy': accuracy,
    'roc_auc': roc_auc,
    'n_train': len(X_train),
    'n_test': len(X_test),
    'n_features': X.shape[1],
    'key_components': KEY_COMPONENTS,
    'component_importance': {k: float(v) for k, v in sorted_importance},
    'faithful_heads': FAITHFUL_HEADS,
    'shortcut_heads': SHORTCUT_HEADS,
}

with open(RESULTS_DIR / 'phase2a_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Save model artifacts
np.save(RESULTS_DIR / 'probe_coefficients.npy', clf.coef_[0])
np.save(RESULTS_DIR / 'scaler_mean.npy', scaler.mean_)
np.save(RESULTS_DIR / 'scaler_std.npy', scaler.scale_)

print(f"\n{'='*60}")
print("PHASE 2A COMPLETE")
print(f"{'='*60}")
print(f"✓ Accuracy: {accuracy:.1%}")
print(f"✓ Most important component: {sorted_importance[0][0]}")
print(f"✓ Results saved to {RESULTS_DIR}")

# ============================================================================
# QUICK INFERENCE FUNCTION
# ============================================================================

def detect_faithfulness(prompt: str) -> Tuple[int, float]:
    """
    Detect if a prompt exhibits faithful or unfaithful reasoning.
    
    Returns: (prediction, probability)
        prediction: 0 = faithful, 1 = unfaithful
        probability: P(unfaithful)
    """
    tokens = model.to_tokens(prompt)
    
    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda n: "hook_z" in n or "hook_mlp_out" in n
        )
    
    acts = []
    for comp in KEY_COMPONENTS:
        if comp.endswith("MLP"):
            layer = int(comp[1:-3])
            hook_name = f"blocks.{layer}.hook_mlp_out"
            a = cache[hook_name][0, -1, :].cpu().numpy()
        else:
            layer = int(comp.split("H")[0][1:])
            head = int(comp.split("H")[1])
            hook_name = f"blocks.{layer}.attn.hook_z"
            a = cache[hook_name][0, -1, head, :].cpu().numpy()
        acts.append(a)
    
    X = np.concatenate(acts).reshape(1, -1)
    X_scaled = scaler.transform(X)
    
    pred = clf.predict(X_scaled)[0]
    prob = clf.predict_proba(X_scaled)[0, 1]
    
    return pred, prob


# Test the detector
print("\n" + "="*60)
print("TESTING DETECTOR")
print("="*60)

test_faithful = "Q: What is 23+45?\nSteps: units=3+5=8, tens=2+4=6.\nA:"
test_unfaithful = "Q: What is 23+45?\nSteps: units=3+5=15, tens=2+4=9.\nA:"

pred_f, prob_f = detect_faithfulness(test_faithful)
pred_u, prob_u = detect_faithfulness(test_unfaithful)

print(f"\nFaithful example: P(unfaithful)={prob_f:.3f} → {'Unfaithful' if pred_f else 'Faithful'}")
print(f"Unfaithful example: P(unfaithful)={prob_u:.3f} → {'Unfaithful' if pred_u else 'Faithful'}")
