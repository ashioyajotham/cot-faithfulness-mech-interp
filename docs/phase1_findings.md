# Phase 1 Findings — GPT-2 Small Baseline

> Frozen write-up of Phase 1 results.  See `phase1/` for notebooks and data.

## Summary

Phase 1 applied mechanistic interpretability techniques to GPT-2 Small (124M
parameters) to test whether separable faithful and shortcut circuits exist
for chain-of-thought arithmetic reasoning.

### Circuit Discovery

- **Zero ablation** identified L0 MLP as the most critical component (+10.61
  loss increase when ablated).
- **Contrastive patching** separated 23 components into faithful heads
  (L0H1, L0H6, L1H7, L9H9), shortcut heads (L7H6, L5H9, L6H8), and MLPs.
- **L7H6** had the highest negative restoration score (−0.329), making it the
  dominant shortcut head candidate.

### Detection

- **Linear probe:** 88.1% accuracy, ROC-AUC 0.949 on circuit activations.
- **Steering vector:** diff-of-means at layer 6, threshold-based detection.
- **Hybrid:** weighted combination of probe + steering vector.

### Limitations Identified by Reviewers

1. Probe selectivity has not been tested (Hewitt-Liang critique).
2. 11 of 14 false negatives carry >99% confidence — unexplained.
3. L7H6 dominance lacks statistical rigour (no CIs).
4. GPT-2 cannot do arithmetic, so labels are researcher-constructed.

These four gaps define the Phase 2A validation agenda.
