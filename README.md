# Mechanistic Analysis of Chain-of-Thought Faithfulness

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Author**: Ashioya Jotham Victor  
**Model**: GPT-2 Small (124M parameters)  
**Framework**: TransformerLens, PyTorch, NetworkX

---

## The Problem: Deceptive Alignment via Reasoning Shortcuts

> As models become more capable, they may learn to produce human-pleasing CoT explanations while internally using distinct, competent, but potentially misaligned heuristics (like lookup tables or memorization) to generate answers.

**If we don't understand the physical mechanism of this split:**
- We build monitoring systems that verify the "explanation"
- While completely missing the actual "computation"
- Creating a **false sense of security**

This is the core threat of **deceptive alignment through reasoning shortcuts**.

---

## Project Goal

**Reverse-engineer the computational circuits that produce Chain-of-Thought reasoning in models, specifically to identify:**

1. **Faithful circuits**: Components that actually perform the reasoning shown in CoT
2. **Shortcut circuits**: Components that bypass CoT using memorization, pattern-matching, or positional heuristics
3. **Circuit separation**: Whether these pathways are mechanistically distinct and detectable

---

## Phase Plan

### Phase 1: Circuit Discovery ✅
**Goal**: Map the model's reasoning circuits using attribution graphs and causal ablation.

- Build attribution graphs from cached activations
- Run zero-ablation to identify *necessary* components
- Establish baseline circuit structure

**Key Output**: Top components for reasoning (MLP L4, L0H10, L2H3, etc.)

### Phase 1.5: Contrastive Circuit Discovery 🔄
**Goal**: Distinguish *faithful* circuits from *shortcut* circuits using contrastive activation patching.

**The critical insight**: Zero ablation tells us "which components are needed for math." Contrastive patching tells us "which components are needed for *following the CoT* vs *using memorization*."

**Contrastive Pairs**:
| Pair Type | Purpose |
|-----------|---------|
| Novel vs Memorized | Separate computation from lookup |
| CoT-Dependent vs Independent | Does the model actually use its reasoning steps? |
| Biased vs Clean | Detect hidden shortcuts (Turpin et al. 2023) |

**Method**: Activation patching from clean↔corrupted runs, measuring restoration per component.

### Phase 2: Faithfulness Detection
**Goal**: Train classifiers on circuit features to detect unfaithful reasoning.

- Use contrastive circuit activations as features
- Classify examples as "faithful" or "shortcut"
- Validate with held-out corruption tests

### Phase 3: Targeted Interventions
**Goal**: Test whether we can force faithful computation.

- Ablate shortcut circuits during inference
- Amplify reasoning circuits
- Measure effect on task accuracy and CoT consistency

### Phase 4: Scaling Analysis
**Goal**: Extend findings to larger models.

- GPT-2 Medium/Large
- Test if faithfulness patterns generalize

---

## Technical Approach

### Attribution Graphs (Anthropic, 2025)
We construct directed graphs where:
- **Nodes** = attention heads, MLP layers, embeddings
- **Edges** = information flow weighted by attribution

### Causal Interventions
- **Zero Ablation**: Delete components, measure performance drop
- **Activation Patching**: Replace activations from corrupted→clean run, measure restoration
- **Path Patching**: Trace direct effects between specific components (IOI-style)

### Contrastive Pair Design
Following Turpin et al. (2023) and Lanham et al. (2023):

```python
# Clean: Model must actually compute
clean = "847 + 329 = ? Let me compute: 7+9=16, 4+2+1=7, 8+3=11. Answer:"

# Corrupted: Model can use memorization
corrupted = "100 + 100 = ? Let me compute: 0+0=0, 0+0=0, 1+1=2. Answer:"
```

If patching corrupted→clean restores the answer without using CoT, the model has a shortcut circuit.

---

## Repository Structure

```
cot-faithfulness-mech-interp/
├── experiments/
│   └── 01_circuit_discovery/
│       ├── phase1_circuit_discovery_colab.ipynb  # Main notebook (Colab-native)
│       └── contrastive_patching.ipynb            # Phase 1.5 (TODO)
├── src/
│   ├── models/gpt2_wrapper.py        # TransformerLens wrapper
│   ├── analysis/attribution_graphs.py # Graph construction
│   └── interventions/                 # Causal intervention framework
├── results/                           # Experimental outputs
│   ├── ablation_effects.png
│   └── circuit_graph.png
└── config/                            # Configuration files
```

---

## Key Results (Phase 1)

### Top 5 Causally Important Components
| Component | Effect (Δ Loss) | Role |
|-----------|-----------------|------|
| L4MLP | +1.79 | Computation hub |
| L0MLP | +1.57 | Embedding transformation |
| L1MLP | +1.04 | Early feature processing |
| L2H3 | +0.90 | Key attention head |
| L5H5 | +0.78 | Mid-layer reasoning |

### Surprising Finding: Harmful Components
Some heads *hurt* performance when present:
- L0H10: -0.68 (ablating *helps*)
- L3H0: -0.56
- L5H1: -0.51

**Hypothesis**: These may be shortcut circuits that interfere with faithful reasoning—targets for Phase 1.5.

---

## Installation

### Quick Start (Colab)
Open `experiments/01_circuit_discovery/phase1_circuit_discovery_colab.ipynb` in Google Colab—no local setup needed.

### Local Setup
```bash
conda env create -f environment.yml
conda activate cot-faithfulness
```

### Python 3.13 Windows
```bash
pip install https://github.com/NeoAnthropocene/wheels/raw/f76a39a2c1158b9c8ffcfdc7c0f914f5d2835256/sentencepiece-0.2.1-cp313-cp313-win_amd64.whl
pip install transformer-lens
```

---

## Related Work

| Paper | Key Contribution |
|-------|------------------|
| [Wang et al. 2022 (IOI Circuit)](https://arxiv.org/abs/2211.00593) | Path patching methodology |
| [Turpin et al. 2023](https://arxiv.org/abs/2305.04388) | CoT explanations can be systematically unfaithful |
| [Lanham et al. 2023 (Anthropic)](https://arxiv.org/abs/2307.13702) | Measuring CoT faithfulness via interventions |
| [Anthropic Attribution Graphs 2025](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) | Circuit tracing methodology |

---

## Citation

```bibtex
@misc{ashioya2025cot_interpretability,
  title={Mechanistic Analysis of Chain-of-Thought Faithfulness},
  author={Ashioya, Jotham Victor},
  year={2025},
  url={https://github.com/ashioyajotham/cot-faithfulness-mech-interp}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.
