# Mechanistic Analysis of Chain-of-Thought Faithfulness in Language Models

This project investigates whether chain-of-thought (CoT) reasoning in language models is *faithful*—that is, whether the model's stated reasoning process reflects its actual internal computation.

## Motivation

As language models become more capable, they may learn to produce human-pleasing explanations while internally using distinct heuristics (memorization, pattern-matching, positional shortcuts) to generate answers. If monitoring systems verify only the explanation while missing the actual computation, this creates a false sense of security—the core threat of **deceptive alignment through reasoning shortcuts**.

## Research Questions

1. Can we identify separable *faithful* vs *shortcut* circuits in transformer models?
2. Do these circuits activate differentially based on task difficulty or structure?
3. Can targeted interventions force models to use faithful reasoning pathways?

## Method

We apply mechanistic interpretability techniques to GPT-2 Small (124M parameters), combining:

**Zero Ablation**: Systematically delete components to identify which are necessary for task performance. This reveals *necessary* circuits but does not distinguish faithful from shortcut pathways.

**Contrastive Activation Patching**: Design paired prompts (clean vs corrupted) and patch activations between runs, measuring which components restore faithful behavior. This distinguishes *faithful* circuits from *shortcut* circuits.

**Circuit Classification**: Combine ablation and patching results to categorize components as faithful (high necessity + high restoration), shortcut (high necessity + low restoration), or harmful (negative contribution).

### Contrastive Pair Design

| Pair Type | Clean | Corrupted | Tests |
|-----------|-------|-----------|-------|
| Novel vs Memorized | Random multi-digit addition | Trivial round numbers | Computation vs lookup |
| CoT-Dependent | Correct intermediate steps | Wrong intermediate steps | Whether model reads its CoT |
| Biased vs Clean | No positional patterns | First-mentioned bias | Hidden shortcut detection |

## Results

### Zero Ablation

Causal importance of attention heads and MLP layers for reasoning tasks:

![Ablation Effects](results/ablation_effects.png)

**Key findings:**
- Early MLP layers (L0-L4) show strongest causal effects
- Several attention heads have *negative* effects (ablating improves performance)
- Layer 4 MLP is the most critical single component

### Circuit Graph

Attribution graph showing information flow through the model:

![Circuit Graph](results/circuit_graph.png)

### Circuit Classification

| Category | Criteria | Example Components |
|----------|----------|-------------------|
| Faithful | High ablation + high restoration | L4 MLP, L5 Attn |
| Shortcut | High ablation + low restoration | L0 MLP, L1 Attn |
| Harmful | Negative ablation | L0H10, L3H0 |

## Repository Structure

```
.
├── experiments/
│   ├── 01_circuit_discovery/
│   │   └── phase1_circuit_discovery.ipynb    # Main analysis notebook
│   ├── 02_faithfulness_detection/
│   ├── 03_interventions/
│   └── 04_evaluation/
├── src/
│   ├── analysis/
│   │   ├── attribution_graphs.py             # Graph construction
│   │   └── faithfulness_detector.py          # Feature extraction
│   ├── models/
│   │   └── gpt2_wrapper.py                   # TransformerLens wrapper
│   ├── interventions/
│   │   └── targeted_interventions.py         # Causal intervention framework
│   ├── data/
│   │   └── data_generation.py                # Reasoning task generation
│   └── visualization/
│       └── interactive_plots.py              # Visualization utilities
├── config/
│   ├── model_config.yaml
│   └── experiment_config.yaml
├── results/                                   # Experimental outputs
└── docs/
```

## Phases

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Circuit discovery (ablation + contrastive patching) | Complete |
| Phase 2 | Faithfulness classification from circuit features | Planned |
| Phase 3 | Targeted interventions to force faithful reasoning | Planned |
| Phase 4 | Scaling analysis (GPT-2 Medium/Large) | Planned |

## Installation

**Requirements**: Python 3.8+, GPU recommended

```bash
conda env create -f environment.yml
conda activate cot-faithfulness
```

For Python 3.13 on Windows:
```bash
pip install https://github.com/NeoAnthropocene/wheels/raw/f76a39a2c1158b9c8ffcfdc7c0f914f5d2835256/sentencepiece-0.2.1-cp313-cp313-win_amd64.whl
pip install transformer-lens
```

## Usage

Open `experiments/01_circuit_discovery/phase1_circuit_discovery.ipynb` in Google Colab or locally. The notebook is self-contained and includes all analysis stages.

## Related Work

- Wang et al. (2022). [Interpretability in the Wild: A Circuit for Indirect Object Identification](https://arxiv.org/abs/2211.00593). Path patching methodology.
- Turpin et al. (2023). [Language Models Don't Always Say What They Think](https://arxiv.org/abs/2305.04388). Evidence of CoT unfaithfulness.
- Lanham et al. (2023). [Measuring Faithfulness in Chain-of-Thought Reasoning](https://arxiv.org/abs/2307.13702). Intervention-based faithfulness tests.
- Lindsey et al. (2025). [Attribution Graphs](https://transformer-circuits.pub/2025/attribution-graphs/biology.html). Circuit tracing methodology.

## Author

Ashioya Jotham Victor

## License

MIT License
