# Phase 2: Faithfulness Detector

## Overview

Building on **Phase 1.5** head-level circuit discovery, this phase implements **three detection approaches** to identify unfaithful CoT reasoning.

## Phase 1.5 Key Findings (Inputs)

| Type | Components | Top Score |
|------|------------|-----------|
| **Faithful Heads** | L0H1, L0H6, L1H7, L10H2, L3H0, L9H9 | +0.537 |
| **Shortcut Heads** | L7H6, L2H10, L0H3, L2H0, L3H10, L0H10 | -0.329 |
| **Faithful MLPs** | L0MLP, L5MLP | +4.336 |
| **Shortcut MLPs** | L10MLP, L3MLP, L2MLP | -0.643 |

## Detection Methods

| Path | Method | How It Works |
|------|--------|--------------|
| **A** | Linear Probe | Train logistic regression on Phase 1.5 circuit activations |
| **B** | Steering Vector | Diff-of-means on residual stream (layer 6) |
| **C** | Hybrid | Diff-of-means on Phase 1.5 circuits only |

## Files

| File | Description |
|------|-------------|
| `phase2_faithfulness_detector.ipynb` | **Colab notebook** - Run this! |
| `phase2_faithfulness_detector.py` | Python script version |
| `phase2_README.md` | This file |

## Quick Start

### 1. Open in Colab
Upload `phase2_faithfulness_detector.ipynb` to Google Colab.

### 2. Run all cells
The notebook includes:
- Dataset generation (400+ examples)
- Three detection paths
- Comparison visualization
- Saved results

### 3. Expected Output

```
COMPARISON OF ALL THREE PATHS
==============================================================

Method                      Accuracy    ROC-AUC
--------------------------------------------------
Path A (Linear Probe)          0.850      0.920
Path B (Steering Vector)       0.780      0.860
Path C (Hybrid)                0.870      0.930
```

**Target:** >80% accuracy, >0.85 ROC-AUC

## Interpretation

| Outcome | Meaning |
|---------|---------|
| Path C > Path B | Phase 1.5 circuits ADD VALUE to detection |
| Path A best | Linear boundary works; can train classifier |
| All similar | Faithfulness is simple linear direction |

## Output Files

```
results/phase2_faithfulness_detector/
├── phase2_results.json      # All metrics
├── steering_vector.npy      # For intervention experiments
├── hybrid_vector.npy        # Circuit-grounded direction
├── probe_coefficients.npy   # Feature weights
└── phase2_comparison.png    # Visualization
```

## Next Steps

1. **If accuracy > 80%**: Test on new arithmetic problems
2. **Intervention**: Use steering vector to *force* faithful reasoning
3. **Scale up**: Try GPT-2 Medium/Large
4. **Compare**: Reach out to Ivan Arcuschin's probing work

---

*Phase 2 of CoT Faithfulness Mech Interp | Victor Ashioya | Bluedot Impact*
