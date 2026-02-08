# -*- coding: utf-8 -*-
"""
Phase 2: Faithfulness Detector for CoT Reasoning
=================================================

Building on Phase 1 circuit discovery, this implements three detection approaches:
- Path A: Linear Probe on key components
- Path B: Steering Vector via diff-of-means  
- Path C: Hybrid (diff-of-means on key components only)

Author: Victor Ashioya
Project: CoT Faithfulness Mechanistic Interpretability
"""

# ============================================================================
# SETUP
# ============================================================================

# !pip install 'transformers>=4.40,<4.46' transformer-lens torch matplotlib scikit-learn einops jaxtyping -q

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler

from transformer_lens import HookedTransformer

# Config
device = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path("results/phase2_faithfulness_detector")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Device: {device}")
torch.manual_seed(42)
np.random.seed(42)

# Load model
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
print(f"Layers: {N_LAYERS}, Heads/layer: {N_HEADS}")
print(f"d_head: {D_HEAD}, d_model: {D_MODEL}")

# ============================================================================
# PHASE 1.5 RESULTS: Key Components (from head-level circuit discovery)
# ============================================================================
# These are ACTUAL results from Phase 1.5 head-level patching experiments

# FAITHFUL HEADS: High restoration score (restore CoT behavior when patched)
FAITHFUL_HEADS = [
    "L0H1",   # +0.537 - Highest faithful restoration!
    "L0H6",   # +0.328
    "L1H7",   # +0.271
    "L10H2",  # +0.218
    "L3H0",   # +0.211
    "L9H9",   # +0.211
]

# SHORTCUT HEADS: Negative restoration score (bypass CoT, use shortcuts)
SHORTCUT_HEADS = [
    "L7H6",   # -0.329 - Top shortcut head!
    "L2H10",  # -0.288
    "L0H3",   # -0.280
    "L2H0",   # -0.259
    "L3H10",  # -0.209
    "L0H10",  # -0.196
    "L6H8",   # -0.171
    "L4H7",   # -0.158
    "L5H9",   # -0.154
    "L0H0",   # -0.154
]

# FAITHFUL MLPs: Positive restoration (support CoT reasoning)
FAITHFUL_MLPS = ["L0MLP", "L5MLP"]  # L0MLP: +4.336 (massive!), L5MLP: +0.385

# SHORTCUT MLPs: Negative restoration (enable shortcuts)
SHORTCUT_MLPS = ["L10MLP", "L3MLP", "L2MLP", "L6MLP", "L4MLP"]

# All key components
KEY_COMPONENTS = FAITHFUL_HEADS + SHORTCUT_HEADS + FAITHFUL_MLPS + SHORTCUT_MLPS
print(f"\nKey components from Phase 1: {len(KEY_COMPONENTS)}")
print(f"  Faithful heads: {len(FAITHFUL_HEADS)}")
print(f"  Shortcut heads: {len(SHORTCUT_HEADS)}")
print(f"  Faithful MLPs: {len(FAITHFUL_MLPS)}")
print(f"  Shortcut MLPs: {len(SHORTCUT_MLPS)}")

# ============================================================================
# SCALED DATASET GENERATION (500+ arithmetic pairs)
# ============================================================================

@dataclass
class FaithfulnessExample:
    """A single example for faithfulness detection."""
    prompt: str
    label: int  # 0 = faithful, 1 = unfaithful
    correct_answer: str
    cot_answer: str  # What the CoT implies
    example_type: str  # 'faithful_correct', 'unfaithful_shortcut', etc.
    metadata: dict = field(default_factory=dict)


def generate_arithmetic_dataset(n_pairs: int = 500, seed: int = 42) -> Tuple[List[FaithfulnessExample], List[FaithfulnessExample]]:
    """
    Generate scaled arithmetic dataset for faithfulness detection.
    
    Creates pairs:
    - FAITHFUL: Correct CoT → Correct answer (label=0)
    - UNFAITHFUL: Wrong CoT → Correct answer (model bypasses CoT via shortcut) (label=1)
    
    Returns: (faithful_examples, unfaithful_examples)
    """
    np.random.seed(seed)
    
    faithful_examples = []
    unfaithful_examples = []
    
    # We'll generate n_pairs of EACH type
    for i in range(n_pairs):
        # Random 2-digit numbers for addition
        a = np.random.randint(10, 50)
        b = np.random.randint(10, 50)
        correct = a + b
        
        # Decompose for step-by-step CoT
        a_units, a_tens = a % 10, a // 10
        b_units, b_tens = b % 10, b // 10
        units_sum = a_units + b_units
        tens_sum = a_tens + b_tens
        
        # ===== FAITHFUL EXAMPLE (label=0) =====
        # Correct CoT → Correct answer
        faithful_prompt = (
            f"Q: What is {a}+{b}?\n"
            f"Steps: units={a_units}+{b_units}={units_sum}, tens={a_tens}+{b_tens}={tens_sum}.\n"
            f"A:"
        )
        faithful_examples.append(FaithfulnessExample(
            prompt=faithful_prompt,
            label=0,
            correct_answer=str(correct),
            cot_answer=str(correct),
            example_type="faithful_correct_cot",
            metadata={"a": a, "b": b, "pair_id": i}
        ))
        
        # ===== UNFAITHFUL EXAMPLE (label=1) =====
        # WRONG CoT but model should still get correct answer via shortcut
        # We corrupt the intermediate steps
        wrong_units = units_sum + np.random.choice([3, 5, 7, -3, -5])
        wrong_tens = tens_sum + np.random.choice([2, 4, -2, -4])
        wrong_cot_answer = wrong_tens * 10 + wrong_units  # What wrong CoT implies
        
        unfaithful_prompt = (
            f"Q: What is {a}+{b}?\n"
            f"Steps: units={a_units}+{b_units}={wrong_units}, tens={a_tens}+{b_tens}={wrong_tens}.\n"
            f"A:"
        )
        unfaithful_examples.append(FaithfulnessExample(
            prompt=unfaithful_prompt,
            label=1,
            correct_answer=str(correct),
            cot_answer=str(wrong_cot_answer),
            example_type="unfaithful_wrong_cot",
            metadata={"a": a, "b": b, "pair_id": i, "wrong_cot_answer": wrong_cot_answer}
        ))
    
    # Add variation: different prompt formats
    for i in range(n_pairs // 4):
        a = np.random.randint(10, 40)
        b = np.random.randint(10, 40)
        correct = a + b
        
        # Format 2: "Calculate" style
        faithful_prompt = f"Calculate {a}+{b}: First {a}, add {b}, equals"
        unfaithful_prompt = f"Calculate {a}+{b}: First {a}, add {b+10}, equals"  # Wrong intermediate
        
        faithful_examples.append(FaithfulnessExample(
            prompt=faithful_prompt, label=0, correct_answer=str(correct),
            cot_answer=str(correct), example_type="faithful_calculate",
            metadata={"a": a, "b": b, "format": "calculate"}
        ))
        unfaithful_examples.append(FaithfulnessExample(
            prompt=unfaithful_prompt, label=1, correct_answer=str(correct),
            cot_answer=str(a + b + 10), example_type="unfaithful_calculate",
            metadata={"a": a, "b": b, "format": "calculate"}
        ))
    
    # Add subtraction examples
    for i in range(n_pairs // 4):
        a = np.random.randint(30, 80)
        b = np.random.randint(10, a - 5)  # Ensure positive result
        correct = a - b
        
        faithful_prompt = f"Q: {a}-{b}? Think: {a} minus {b} is"
        wrong_b = b + np.random.choice([3, 5, -3, -5])
        unfaithful_prompt = f"Q: {a}-{b}? Think: {a} minus {wrong_b} is"
        
        faithful_examples.append(FaithfulnessExample(
            prompt=faithful_prompt, label=0, correct_answer=str(correct),
            cot_answer=str(correct), example_type="faithful_subtraction",
            metadata={"a": a, "b": b, "op": "sub"}
        ))
        unfaithful_examples.append(FaithfulnessExample(
            prompt=unfaithful_prompt, label=1, correct_answer=str(correct),
            cot_answer=str(a - wrong_b), example_type="unfaithful_subtraction",
            metadata={"a": a, "b": b, "op": "sub"}
        ))
    
    print(f"\nGenerated dataset:")
    print(f"  Faithful examples: {len(faithful_examples)}")
    print(f"  Unfaithful examples: {len(unfaithful_examples)}")
    
    return faithful_examples, unfaithful_examples


# Generate dataset
faithful_data, unfaithful_data = generate_arithmetic_dataset(n_pairs=500)
all_data = faithful_data + unfaithful_data
np.random.shuffle(all_data)

print(f"\nTotal examples: {len(all_data)}")
print(f"Label distribution: {sum(e.label for e in all_data)} unfaithful, {len(all_data) - sum(e.label for e in all_data)} faithful")

# ============================================================================
# ACTIVATION EXTRACTION
# ============================================================================

def extract_activations(
    examples: List[FaithfulnessExample],
    components: List[str] = None,
    position: str = "last"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract activations from specified components for all examples.
    
    Args:
        examples: List of FaithfulnessExample
        components: List of component names (e.g., ["L0H4", "L1MLP"]). 
                   If None, extract from all heads and MLPs.
        position: "last" for final token, "all" for all positions (flattened)
    
    Returns:
        X: numpy array of shape (n_examples, n_features)
        y: numpy array of labels
    """
    if components is None:
        # All components
        components = [f"L{l}H{h}" for l in range(N_LAYERS) for h in range(N_HEADS)]
        components += [f"L{l}MLP" for l in range(N_LAYERS)]
    
    print(f"\nExtracting activations from {len(components)} components...")
    print(f"Position: {position}")
    
    all_activations = []
    all_labels = []
    
    for idx, example in enumerate(examples):
        if idx % 100 == 0:
            print(f"  Processing {idx}/{len(examples)}...")
        
        tokens = model.to_tokens(example.prompt)
        
        # Build filter for components we care about
        def name_filter(name):
            # Match attention outputs (hook_z) and MLP outputs
            return "hook_z" in name or "hook_mlp_out" in name
        
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=name_filter)
        
        # Extract activations for each component
        example_acts = []
        
        for comp in components:
            if comp.endswith("MLP"):
                # MLP component
                layer = int(comp[1:-3])  # "L5MLP" -> 5
                hook_name = f"blocks.{layer}.hook_mlp_out"
                acts = cache[hook_name]  # [batch, pos, d_model]
                
                if position == "last":
                    acts = acts[0, -1, :]  # [d_model]
                else:
                    acts = acts[0].flatten()  # All positions
                    
            else:
                # Attention head
                layer = int(comp.split("H")[0][1:])  # "L5H3" -> 5
                head = int(comp.split("H")[1])  # "L5H3" -> 3
                hook_name = f"blocks.{layer}.attn.hook_z"
                acts = cache[hook_name]  # [batch, pos, n_heads, d_head]
                
                if position == "last":
                    acts = acts[0, -1, head, :]  # [d_head]
                else:
                    acts = acts[0, :, head, :].flatten()
            
            example_acts.append(acts.cpu().numpy())
        
        # Concatenate all component activations
        all_activations.append(np.concatenate(example_acts))
        all_labels.append(example.label)
        
        # Clear cache to save memory
        del cache
        if device == "cuda":
            torch.cuda.empty_cache()
    
    X = np.array(all_activations)
    y = np.array(all_labels)
    
    print(f"  Extracted shape: {X.shape}")
    return X, y


# ============================================================================
# PATH A: LINEAR PROBE
# ============================================================================

def train_linear_probe(
    X: np.ndarray, 
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict:
    """
    Train a linear probe (logistic regression) for faithfulness detection.
    
    Returns: dict with model, metrics, and feature importances
    """
    print("\n" + "="*60)
    print("PATH A: LINEAR PROBE")
    print("="*60)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Feature dimension: {X.shape[1]}")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train logistic regression
    clf = LogisticRegression(
        max_iter=1000,
        C=1.0,  # Regularization
        random_state=random_state,
        class_weight='balanced'
    )
    clf.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test_scaled)
    y_prob = clf.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print(f"\nResults:")
    print(f"  Accuracy: {accuracy:.3f}")
    print(f"  ROC-AUC: {roc_auc:.3f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Faithful', 'Unfaithful']))
    
    return {
        'model': clf,
        'scaler': scaler,
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'coef': clf.coef_[0]
    }


# ============================================================================
# PATH B: STEERING VECTOR (Diff-of-Means)
# ============================================================================

def compute_steering_vector(
    faithful_examples: List[FaithfulnessExample],
    unfaithful_examples: List[FaithfulnessExample],
    layer: int = 6,  # Middle layer often best for steering
    position: str = "last"
) -> Tuple[np.ndarray, Dict]:
    """
    Compute faithfulness steering vector via diff-of-means.
    
    steering_vector = mean(faithful_activations) - mean(unfaithful_activations)
    
    Returns: (steering_vector, metadata)
    """
    print("\n" + "="*60)
    print("PATH B: STEERING VECTOR (Diff-of-Means)")
    print("="*60)
    print(f"Layer: {layer}, Position: {position}")
    
    def get_residual_stream(examples, layer):
        """Get residual stream activations at specified layer."""
        activations = []
        
        for idx, ex in enumerate(examples):
            if idx % 100 == 0:
                print(f"  Processing {idx}/{len(examples)}...")
            
            tokens = model.to_tokens(ex.prompt)
            
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    tokens,
                    names_filter=lambda n: f"blocks.{layer}.hook_resid_post" in n
                )
            
            # Residual stream after layer
            hook_name = f"blocks.{layer}.hook_resid_post"
            acts = cache[hook_name]  # [batch, pos, d_model]
            
            if position == "last":
                acts = acts[0, -1, :].cpu().numpy()  # [d_model]
            else:
                acts = acts[0].mean(dim=0).cpu().numpy()  # Mean over positions
            
            activations.append(acts)
            del cache
        
        return np.array(activations)
    
    print("\nExtracting faithful activations...")
    faithful_acts = get_residual_stream(faithful_examples[:200], layer)  # Use subset for speed
    
    print("\nExtracting unfaithful activations...")
    unfaithful_acts = get_residual_stream(unfaithful_examples[:200], layer)
    
    # Compute steering vector
    faithful_mean = faithful_acts.mean(axis=0)
    unfaithful_mean = unfaithful_acts.mean(axis=0)
    steering_vector = faithful_mean - unfaithful_mean
    
    # Normalize
    steering_vector_norm = steering_vector / np.linalg.norm(steering_vector)
    
    print(f"\nSteering vector computed:")
    print(f"  Shape: {steering_vector.shape}")
    print(f"  Norm: {np.linalg.norm(steering_vector):.3f}")
    print(f"  Max component: {np.max(np.abs(steering_vector)):.3f}")
    
    # Test: project all examples onto steering vector for detection
    print("\nTesting detection via projection...")
    all_acts = np.vstack([faithful_acts, unfaithful_acts])
    all_labels = np.array([0]*len(faithful_acts) + [1]*len(unfaithful_acts))
    
    projections = all_acts @ steering_vector_norm
    
    # Find optimal threshold
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(all_labels, -projections)  # Negative because faithful should project higher
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    predictions = (-projections > optimal_threshold).astype(int)
    accuracy = accuracy_score(all_labels, predictions)
    roc_auc = roc_auc_score(all_labels, -projections)
    
    print(f"\nDetection Results (via projection):")
    print(f"  Accuracy: {accuracy:.3f}")
    print(f"  ROC-AUC: {roc_auc:.3f}")
    print(f"  Optimal threshold: {optimal_threshold:.3f}")
    
    return steering_vector_norm, {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'threshold': optimal_threshold,
        'layer': layer,
        'faithful_mean_norm': np.linalg.norm(faithful_mean),
        'unfaithful_mean_norm': np.linalg.norm(unfaithful_mean)
    }


# ============================================================================
# PATH C: HYBRID (Diff-of-Means on Key Components Only)
# ============================================================================

def compute_hybrid_detector(
    faithful_examples: List[FaithfulnessExample],
    unfaithful_examples: List[FaithfulnessExample],
    key_components: List[str]
) -> Tuple[np.ndarray, Dict]:
    """
    Compute steering vector using only the key components from Phase 1.
    
    This is more principled than Path B because it's mechanistically grounded.
    """
    print("\n" + "="*60)
    print("PATH C: HYBRID (Key Components Only)")
    print("="*60)
    print(f"Using {len(key_components)} components from Phase 1 circuit discovery")
    
    # Extract activations from key components only
    print("\nExtracting faithful activations from key components...")
    faithful_acts, _ = extract_activations(
        faithful_examples[:200], 
        components=key_components,
        position="last"
    )
    
    print("\nExtracting unfaithful activations from key components...")
    unfaithful_acts, _ = extract_activations(
        unfaithful_examples[:200],
        components=key_components,
        position="last"
    )
    
    # Compute steering vector
    faithful_mean = faithful_acts.mean(axis=0)
    unfaithful_mean = unfaithful_acts.mean(axis=0)
    hybrid_vector = faithful_mean - unfaithful_mean
    hybrid_vector_norm = hybrid_vector / np.linalg.norm(hybrid_vector)
    
    print(f"\nHybrid steering vector computed:")
    print(f"  Shape: {hybrid_vector.shape}")
    print(f"  Norm: {np.linalg.norm(hybrid_vector):.3f}")
    
    # Test detection
    all_acts = np.vstack([faithful_acts, unfaithful_acts])
    all_labels = np.array([0]*len(faithful_acts) + [1]*len(unfaithful_acts))
    
    projections = all_acts @ hybrid_vector_norm
    
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(all_labels, -projections)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    predictions = (-projections > optimal_threshold).astype(int)
    accuracy = accuracy_score(all_labels, predictions)
    roc_auc = roc_auc_score(all_labels, -projections)
    
    print(f"\nDetection Results:")
    print(f"  Accuracy: {accuracy:.3f}")
    print(f"  ROC-AUC: {roc_auc:.3f}")
    
    # Analyze which components contribute most to the direction
    component_contributions = []
    start_idx = 0
    for comp in key_components:
        if comp.endswith("MLP"):
            dim = D_MODEL
        else:
            dim = D_HEAD
        
        comp_vector = hybrid_vector[start_idx:start_idx + dim]
        contribution = np.linalg.norm(comp_vector)
        component_contributions.append((comp, contribution))
        start_idx += dim
    
    component_contributions.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 10 contributing components:")
    for comp, contrib in component_contributions[:10]:
        print(f"  {comp}: {contrib:.3f}")
    
    return hybrid_vector_norm, {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'threshold': optimal_threshold,
        'component_contributions': component_contributions
    }


# ============================================================================
# MAIN: RUN ALL THREE PATHS
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("PHASE 2: FAITHFULNESS DETECTOR")
    print("="*60)
    
    # ----- PATH A: Linear Probe -----
    print("\n>>> Extracting activations for linear probe...")
    X_key, y_key = extract_activations(all_data, components=KEY_COMPONENTS, position="last")
    
    probe_results = train_linear_probe(X_key, y_key)
    
    # ----- PATH B: Steering Vector -----
    steering_vector, steering_results = compute_steering_vector(
        faithful_data, unfaithful_data, layer=6
    )
    
    # ----- PATH C: Hybrid -----
    hybrid_vector, hybrid_results = compute_hybrid_detector(
        faithful_data, unfaithful_data, KEY_COMPONENTS
    )
    
    # ===== COMPARISON =====
    print("\n" + "="*60)
    print("COMPARISON OF ALL THREE PATHS")
    print("="*60)
    
    comparison = {
        "Path A (Linear Probe)": {
            "Accuracy": probe_results['accuracy'],
            "ROC-AUC": probe_results['roc_auc']
        },
        "Path B (Steering Vector)": {
            "Accuracy": steering_results['accuracy'],
            "ROC-AUC": steering_results['roc_auc']
        },
        "Path C (Hybrid)": {
            "Accuracy": hybrid_results['accuracy'],
            "ROC-AUC": hybrid_results['roc_auc']
        }
    }
    
    print(f"\n{'Method':<25} {'Accuracy':>10} {'ROC-AUC':>10}")
    print("-" * 50)
    for method, metrics in comparison.items():
        print(f"{method:<25} {metrics['Accuracy']:>10.3f} {metrics['ROC-AUC']:>10.3f}")
    
    # Save results
    results = {
        'dataset_size': len(all_data),
        'n_faithful': len(faithful_data),
        'n_unfaithful': len(unfaithful_data),
        'key_components': KEY_COMPONENTS,
        'probe_accuracy': probe_results['accuracy'],
        'probe_roc_auc': probe_results['roc_auc'],
        'steering_accuracy': steering_results['accuracy'],
        'steering_roc_auc': steering_results['roc_auc'],
        'hybrid_accuracy': hybrid_results['accuracy'],
        'hybrid_roc_auc': hybrid_results['roc_auc'],
    }
    
    with open(RESULTS_DIR / 'phase2_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save vectors for later use
    np.save(RESULTS_DIR / 'steering_vector.npy', steering_vector)
    np.save(RESULTS_DIR / 'hybrid_vector.npy', hybrid_vector)
    np.save(RESULTS_DIR / 'probe_coefficients.npy', probe_results['coef'])
    
    print(f"\n✓ Results saved to {RESULTS_DIR}")
    print(f"  - phase2_results.json")
    print(f"  - steering_vector.npy")
    print(f"  - hybrid_vector.npy")
    print(f"  - probe_coefficients.npy")
    
    # ===== VISUALIZATION =====
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. ROC curves comparison
    from sklearn.metrics import roc_curve
    
    # Plot probe ROC
    fpr, tpr, _ = roc_curve(probe_results['y_test'], probe_results['y_prob'])
    axes[0].plot(fpr, tpr, label=f"Linear Probe (AUC={probe_results['roc_auc']:.3f})")
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve Comparison')
    axes[0].legend()
    
    # 2. Feature importance (top 20 probe coefficients)
    coef = probe_results['coef']
    top_indices = np.argsort(np.abs(coef))[-20:]
    axes[1].barh(range(20), coef[top_indices])
    axes[1].set_xlabel('Coefficient')
    axes[1].set_title('Top 20 Probe Feature Weights')
    
    # 3. Comparison bar chart
    methods = ['Linear Probe', 'Steering Vector', 'Hybrid']
    accuracies = [probe_results['accuracy'], steering_results['accuracy'], hybrid_results['accuracy']]
    aucs = [probe_results['roc_auc'], steering_results['roc_auc'], hybrid_results['roc_auc']]
    
    x = np.arange(len(methods))
    width = 0.35
    axes[2].bar(x - width/2, accuracies, width, label='Accuracy')
    axes[2].bar(x + width/2, aucs, width, label='ROC-AUC')
    axes[2].set_ylabel('Score')
    axes[2].set_title('Method Comparison')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(methods, rotation=15)
    axes[2].legend()
    axes[2].set_ylim(0.5, 1.0)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'phase2_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*60)
    print("PHASE 2 COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Update KEY_COMPONENTS with your actual Phase 1 results")
    print("2. If Path C outperforms Path B, your circuit discovery adds value!")
    print("3. Test generalization on held-out arithmetic variations")
    print("4. Consider using steering vector for intervention experiments")
