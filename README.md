# Mechanistic Analysis of Chain-of-Thought Faithfulness in Language Models

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Python 3.13 Compatible](https://img.shields.io/badge/python-3.13%20compatible-brightgreen.svg)](https://github.com/google/sentencepiece/issues/1104)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Author**: Ashioya Jotham Victor

**Model**: GPT-2 Small (124M parameters)

**Framework**: TransformerLens, PyTorch, NetworkX

---

## Overview

This work applies mechanistic interpretability techniques to reverse-engineer the computational circuits underlying chain-of-thought (CoT) reasoning in GPT-2 Small. Adapting Anthropic's [Attribution Graphs methodology](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) (Lindsey et al., 2025)—which uses cross-layer transcoders and local replacement models to trace circuits in Claude 3.5 Haiku—we develop a direct activation analysis pipeline for GPT-2 via TransformerLens hooks to:

1. Construct and analyze attribution graphs mapping information flow during reasoning
2. Identify critical circuit components through causal interventions (activation ablation and patching)
3. Quantify mechanistic contributions of individual attention heads and MLP features
4. Evaluate circuit quality via intervention-based metrics (faithfulness, completeness, minimality)

While Anthropic's approach uses sparse feature dictionaries (30M features) for interpretable decomposition, our implementation directly analyzes GPT-2's native computational components (attention heads, MLP layers) to identify minimal circuits sufficient for CoT reasoning across arithmetic, physics, and logical reasoning tasks. This allows systematic investigation of CoT faithfulness patterns—including motivated reasoning, hallucination, and backward chaining—through graph-based circuit analysis.

## Methodological Foundation

### Circuit Discovery via Attribution Graphs

Following Anthropic's circuit discovery methodology, we construct directed attribution graphs where:

- **Nodes** represent computational units (attention heads, MLP features, embeddings)
- **Edges** represent information flow with attribution weights
- **Node attributes** track layer index, feature type, and causal importance

The attribution graph construction process:

1. Cache model activations during forward pass through TransformerLens hooks
2. Extract per-layer computations (attention head outputs, MLP pre/post-activation states)
3. Build edge weights based on activation strengths and cross-layer dependencies
4. Apply pruning threshold to create sparse, interpretable subgraphs

Current implementation constructs graphs with 602 nodes and 27,600 edges across a 12-layer transformer, with dominant information flow from layer 9 through 11.

### Causal Intervention Protocol

To identify circuit components causally necessary for correct reasoning, we employ:

**Activation Ablation**: Zero out individual attention heads or MLP features at specific token positions and measure effect on final prediction through cross-entropy loss.

**Activation Patching**: Replace activations from a 'corrupted run' (incorrect reasoning) with clean activations from a correct run. Restored predictions indicate causally necessary components.

This methodology is directly comparable to intervention protocols in the IOI circuit work, enabling:

- Per-component effect quantification
- Identification of minimal sufficient circuits
- Cross-task generalization analysis

### Feature-Level Analysis

Building on sparse autoencoder (SAE) interpretability work (Cunningham et al., 2023), we decompose MLP features to identify monosemantic vs. polysemantic activations. This allows distinguishing:

- General computational features (active across task types)
- Reasoning-specific features (active for CoT problems)
- Artifact features (spurious activations)

## Implementation Structure

```tree
cot-faithfulness-mech-interp/
├── src/
│   ├── models/
│   │   └── gpt2_wrapper.py           # GPT-2 with activation caching
│   ├── analysis/
│   │   ├── attribution_graphs.py     # Graph construction from activations
│   │   └── faithfulness_detector.py  # Feature extraction for analysis
│   ├── interventions/
│   │   └── targeted_interventions.py # Causal intervention framework
│   ├── data/
│   │   └── data_generation.py        # Reasoning task generation
│   └── visualization/
│       └── interactive_plots.py      # Interactive graph/ablation plots
├── experiments/
│   ├── phase1_circuit_discovery.ipynb        # Attribution & ablation analysis
│   ├── phase2_faithfulness_detection.ipynb   # ML classification framework
│   ├── phase3_targeted_interventions.ipynb   # Intervention experiments
│   └── phase4_evaluation_analysis.ipynb      # Comparative analysis
├── config/
│   ├── model_config.yaml             # Model hyperparameters
│   ├── experiment_config.yaml        # Analysis settings
│   └── paths_config.yaml             # Data/output paths
├── data/                             # Datasets and cached results
├── results/                          # Experimental outputs
├── docs/                             # Documentation
└── requirements.txt                  # Dependencies
```

## Technical Approach

### Task Design

We evaluate circuit discovery across three reasoning domains:

**Arithmetic Reasoning**: Simple multi-digit addition/subtraction with intermediate step verification.

**Physics Reasoning**: Qualitative physics reasoning (forces, motion, energy conservation) with step-by-step derivations.

**Logical Reasoning**: Propositional logic with transitive relations, requiring chained inference steps.

Each task is instantiated as a completion prompt without explicit "=" terminal markers to avoid spurious token-level shortcuts.

### Model Selection

We use GPT-2 Small (124M parameters) as the analysis target because:

1. Full model size enables comprehensive mechanistic analysis within compute constraints
2. Sufficient reasoning capability for CoT tasks (verified via generation quality assessment)
3. Established precedent for circuit discovery (IOI circuit paper uses identical model)
4. Manageable activation cache memory footprint (enables full-model analysis)

### Activation Collection

Using TransformerLens's hook system:

```python
cache = model.generate_with_cache(
    prompt=reasoning_prompt,
    max_new_tokens=80
)
```

This captures all layer computations for subsequent analysis.

## Key Results

### Circuit Contribution Analysis

Causal ablation across four reasoning examples identifies top contributors:

```python
Component Importance (Mean Causal Effect on Prediction):
- MLP(L0): 15.14  [Embedding transformation, token interaction]
- Head(L0.0): 2.81 [Early attention, position tracking]
- Head(L5.1): 2.45 [Mid-layer reasoning head]
- MLP(L4): 2.35   [Feature composition]
- Head(L0.8): 1.07 [Early attention pattern]
```

Observation: Early layers (L0-L1) show strongest causal effects, suggesting embedding-level transformations are critical for CoT routing.

### Information Flow Patterns

Hub analysis reveals:

- **Source hubs** (high outgoing weights): L10 MLP features (496, 373, 481)
- **Sink hubs** (high incoming weights): L11 attention heads
- **Critical flow**: L9 residual → L10 MLP → L11 attention → prediction

This L9→L11 bottleneck represents approximately 40% of total attribution weight, indicating compressed information flow through deep layers.

### Graph Statistics

- **Nodes**: 602 (1 embedding, 48 attention heads, 48 MLPs, 1 output)
- **Edges**: 27,600
- **Mean edge weight**: 1.144
- **95th percentile**: 3.531
- **Sparsity**: 92.4% (consistent with mechanistic interpretability expectations)

## Installation

### Prerequisites

- Python 3.8+ (tested on 3.13.5)
- GPU recommended (CPU fallback supported)
- 16GB RAM, 50GB disk space

### Setup (Conda)

```bash
conda env create -f environment.yml
conda activate cot-faithfulness
```

### Python 3.13 Windows Compatibility

Due to upstream sentencepiece packaging issues, install pre-built wheel:

```bash
pip install https://github.com/NeoAnthropocene/wheels/raw/f76a39a2c1158b9c8ffcfdc7c0f914f5d2835256/sentencepiece-0.2.1-cp313-cp313-win_amd64.whl
pip install transformer-lens
```

See [google/sentencepiece#1104](https://github.com/google/sentencepiece/issues/1104) for details.

## Usage

### Phase 1: Circuit Discovery

Open and run experiments/phase1_circuit_discovery.ipynb for end-to-end attribution graph construction and causal ablation analysis.

### Phase 2: Causal Ablation

Quantify per-component importance with activation ablation and patching studies.

### Phase 3: Visualization

Interactive graph exploration with hierarchical layouts and attribution visualization.

## Evaluation Criteria

Following the IOI circuit framework (Wang et al., 2022), we evaluate discovered circuits via:

**Faithfulness**: Does the identified circuit produce correct outputs when isolated (with other components ablated)?

**Completeness**: Does the circuit capture all necessary components, or are critical features missing?

**Minimality**: Are all circuit components necessary, or can spurious components be removed?

These metrics enable quantitative circuit quality assessment beyond visual inspection.

## Related Work

This work directly builds on:

- **[Attribution Graphs - Anthropic (2025)](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)**: Core methodology for circuit tracing in language models; introduces cross-layer transcoders (CLTs), attribution graph construction, and systematic validation protocols. Demonstrates circuit analysis on Claude 3.5 Haiku across multi-step reasoning, chain-of-thought faithfulness, and hidden goal detection
- **[IOI Circuit - Wang et al. (2022)](https://arxiv.org/abs/2211.00593)**: Foundational circuit discovery methodology for GPT-2 small; identifies 26 attention heads across 7 classes for indirect object identification task
- **[SAEs for Mechanistic Interpretability - Cunningham et al. (2023)](https://arxiv.org/abs/2309.08600)**: Feature-level interpretability through sparse autoencoders; demonstrates monosemanticity and causal responsibility of learned features
- **[Anthropic Circuits Research](https://transformer-circuits.pub/)**: Comprehensive framework for reverse-engineering transformer circuits; establishes best practices for attribution analysis and intervention protocols

## Citation

```bibtex
@misc{ashioya2025cot_interpretability,
  title={Mechanistic Analysis of Chain-of-Thought Faithfulness},
  author={Ashioya, Jotham Victor},
  year={2025},
  url={https://github.com/ashioyajotham/cot-faithfulness-mech-interp}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

---
