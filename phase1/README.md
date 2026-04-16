# Phase 1 — GPT-2 Small Baseline (FROZEN)

> **This directory is archived.** Nothing here should be modified after
> the v1.0.0 tag.  All Phase 2 work lives in `../phase2/` and imports
> shared code from `../shared/`.

## What Phase 1 Established

- A contrastive pair dataset (600 examples, arithmetic domain) with
  ground-truth faithfulness labels at the activation level
- 23 causally-verified circuit components identified via activation
  patching (hook_z, per-head granularity)
- L7H6 identified as the dominant shortcut head (restoration score
  −0.329; probe coefficient ~20% higher than next component)
- Linear probe: 88.1% accuracy, ROC-AUC 0.949
- Confirmed that GPT-2 cannot do arithmetic, invalidating intervention
  experiments on this model while leaving detection results intact

## Directory Structure

```
phase1/
├── experiments/
│   ├── circuit_discovery/          # Stage 1A/1B + head-level (1.5)
│   └── faithfulness_detection/     # Linear probe, steering vector, hybrid
├── src/                            # GPT-2-specific utilities
│   ├── analysis/
│   ├── models/
│   ├── interventions/
│   ├── data/
│   └── visualization/
├── config/                         # GPT-2 model & experiment configs
└── results/                        # All Phase 1 outputs (do not modify)
```

## Reproducing

```bash
# From the repo root:
pip install -e ".[phase1]"
jupyter lab phase1/experiments/circuit_discovery/phase1_circuit_discovery.ipynb
```

Notebooks are self-contained and use TransformerLens directly.
