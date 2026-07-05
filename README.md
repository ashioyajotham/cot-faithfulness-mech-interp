# Mechanistic Analysis of Chain-of-Thought

This project investigates whether chain-of-thought (CoT) reasoning in language models is *faithful* — whether the model's stated reasoning process reflects its actual internal computation. We use mechanistic interpretability techniques (activation patching, linear probes, steering vectors) to identify separable *faithful* vs *shortcut* circuits in transformers, and build detectors that work at the activation level.

The long-term safety case: if models develop separable circuits for "produce a CoT that looks correct" and "compute the actual answer," CoT monitoring gives weaker guarantees than assumed. This work builds detectors that operate below the text surface.

## Project Status

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | GPT-2 Small baseline — circuit discovery, detection probe, dataset | **Complete** (`v1.0.0`) |
| **Phase 2A** | Validate Phase 1 claims — probe selectivity, error analysis, bootstrap CI | **Complete** |
| **Phase 2B** | Scale to Qwen2.5-1.5B and 7B — circuit discovery, dual-metric, intervention, MLP probes | **Complete** |

## Key Results

### Phase 1 — Circuit Discovery & Detection

- **23 causally-verified circuit components** identified via activation patching (`hook_z`, per-head granularity)
- **L7H6** identified as the most *discriminative* shortcut head — highest mean |coefficient| in probe (0.140, ~40% above next component)
- **88.1% detection accuracy** (ROC-AUC 0.949) via linear probe on circuit activations
- **Separable faithful/shortcut circuits** confirmed: early-layer faithful heads (L0H1, L0MLP), mid-to-late shortcut heads (L7H6, L5H9)
- **GPT-2 cannot do arithmetic** — invalidates intervention experiments but leaves detection intact, motivating Phase 2B

Phase 1 results are archived at `phase1/results/` and fully reproducible from the frozen notebooks.

### Phase 2A — Validation (New findings)

| Gate | Condition | Result | Detail |
|------|-----------|--------|--------|
| **2A-G1** | Probe selectivity > 0 | **PASS** | Selectivity = 0.110; scramble degradation = 0.183 |
| **2A-G2** | L7H6 rank 1 in >90% bootstrap | **FAIL** | L7H6 rank 1 in 0% of samples (mean rank 12/23) |
| **2A-G3** | FN cluster identified | **INCONCLUSIVE** | 51 FNs, no carry overrepresentation (1.27x) |

**Key finding — probe coefficients vs. causal restoration scores measure different things:**

| Metric | Measures | #1 Component | Interpretation |
|--------|----------|--------------|----------------|
| Probe coefficient (Phase 1) | Which activations the *classifier* relies on most | **L7H6** (0.140) | Best per-dimension discriminator for faithful vs unfaithful |
| Restoration score (Phase 2A) | Which component's activations affect the *model output* most | **L0MLP** (0.721) | Most causally influential on the model's logit difference |

These are **complementary, not contradictory**: L7H6 has the most distinctive activation *pattern* for classification, while L0MLP has the largest *causal effect* on model computation. The Phase 1 claim about L7H6 being the top discriminative component holds; the Phase 2A finding that L0MLP dominates causally is new.

Additional findings:
- **Probe AUC = 0.98** (CI: [0.980, 0.991]) — Phase 2A actually *improved* on Phase 1's 0.949
- **Distributed signal**: Layer 8 (non-circuit) achieves 0.906 accuracy vs circuit's 0.925 — faithfulness leaves fingerprints across the entire residual stream
- **Cross-pair stability**: Spearman rho = 0.195 (p=0.37) — component rankings are not stable across pair subsets

Phase 2A results are at `phase2/2a_validation/results/`.

### Phase 2B — Cross-Architecture & Cross-Scale Replication

Full pipeline ran on Qwen2.5-1.5B-Instruct (Colab T4, 50 min) and Qwen2.5-7B-Instruct (Colab A100, 64 min).

#### Three-Scale Comparison

| Finding | GPT-2 (124M) | Qwen 1.5B | Qwen 7B |
|---------|:------------:|:---------:|:-------:|
| Layer 0 attn is #1 layer | ✅ 0.721 | ✅ 1.023 | ✅ **0.944** |
| Dual-metric divergence | L7H6 ≠ L0MLP | L20H9 ≠ L20H5 | **L23H12 ≠ L23H7** |
| Distributed signal | Layer 8 > circuit | Layer 21 > circuit | **Layer 14 > circuit** |
| Circuit probe AUC | 0.98 | 0.617 | **0.536** |
| Selectivity | 0.110 ✅ | 0.069 ✅ | **0.037** ✅ |
| Best intervention | N/A | 2.6% | **3.4%** |
| Faithful preservation | N/A | 95.3% | **97.8–100%** |
| Unfaithful ratio | ~50% | 60% | **69%** |

#### Six Scaling Findings

1. **Layer 0 attention is universal** — top layer component in all three models (the "input bottleneck")
2. **Linear probes degrade monotonically** — AUC: 0.98 → 0.617 → 0.536 as models scale
3. **Dual-metric divergence is universal** — top discriminative ≠ top causal in all three models
4. **Shortcut circuits consolidate** — 7B concentrates ALL 15 top heads in Layer 23
5. **Larger models are more unfaithful** — 69% unfaithful at 7B vs 60% at 1.5B
6. **Faithful preservation improves with scale** — dual-circuit separation becomes cleaner

#### Probe Architecture Comparison (Steps G+H)

| Probe Type | 1.5B AUC | 7B AUC | Notes |
|-----------|----------|--------|-------|
| Circuit linear | 0.617 | 0.536 | Degrades with scale |
| Circuit MLP | 0.532 | 0.527 | No non-linear gain |
| Full-stream linear | **0.660** | 0.579 | Best at 1.5B |
| Full-stream MLP | 0.561 | **0.616** | Best at 7B — non-linear encoding emerges |

**Key insight**: Full-stream MLP is the *only* probe that improves from 1.5B → 7B. The faithfulness signal becomes non-linearly distributed across the residual stream as models scale.

#### Intervention Detail (7B)

| Ablation Set | Success Rate | Faithful Preserved |
|--------------|-------------|-------------------|
| L23H7 | 0.6% | 100% |
| +L23H15 | 1.8% | 100% |
| +L23H16 | 3.1% | 100% |
| +L23H9 | 2.9% | 98.8% |
| +L23H25 | **3.4%** | 97.8% |

Phase 2B results are at `phase2/2b_scaling/results/qwen25-math-1.5b/` and `phase2/2b_scaling/results/qwen25-math-7b/`.

## Method Overview

```
                   Clean prompt          Corrupted prompt
                       │                      │
               ┌───────▼───────┐      ┌───────▼───────┐
               │  Forward pass │      │  Forward pass │
               │  (cache acts) │      │  (cache acts) │
               └───────┬───────┘      └───────┬───────┘
                       │                      │
                       └──────┬───────────────┘
                              ▼
                   Activation Patching
                   (per-head, per-layer)
                              │
                              ▼
                    Restoration scores
                   ────────────────────
                    │                │
            Linear Probe       Steering Vector
            (selectivity,      (difference-of-means
             MDL, bootstrap)    classification)
                    │                │
                    └───────┬────────┘
                            ▼
                   Faithfulness Detector
```

1. **Contrastive pair generation** — matching faithful/unfaithful prompts that differ only in CoT correctness
2. **Activation patching** — replace activations from the corrupted run with the clean run, measure restoration score per component
3. **Probe training** — train logistic regression on the activation patterns to classify faithful vs unfaithful reasoning
4. **Validation** (Phase 2A) — Hewitt-Liang selectivity, bootstrap CIs, error analysis on high-confidence failures
5. **Scaling** (Phase 2B) — two-pass efficient patching for large models, cross-model circuit alignment

## Repository Structure

```
cot-faithfulness-mech-interp/
│
├── phase1/                              # GPT-2 Small baseline — FROZEN (v1.0.0)
│   ├── experiments/
│   │   ├── circuit_discovery/           # Activation patching notebooks (Stage 1A/1B/1.5)
│   │   └── faithfulness_detection/      # Linear probe, steering vector, hybrid analysis
│   ├── src/                             # GPT-2-specific code (wrapper, attribution graphs)
│   │   ├── models/gpt2_wrapper.py
│   │   ├── analysis/                    # Attribution graphs, faithfulness detector
│   │   ├── interventions/               # Targeted interventions
│   │   ├── data/data_generation.py      # Arithmetic dataset generator
│   │   └── visualization/               # Interactive Plotly dashboards
│   ├── config/                          # GPT-2 model + experiment configs
│   └── results/                         # All Phase 1 outputs (do not modify)
│       ├── phase1_circuit_discovery/    # Ablation heatmaps, restoration PNGs
│       ├── phase1_5_head_level/         # Per-head restoration scores
│       └── phase2_faithfulness_detection/ # Probe results, steering vectors, coefficients
│
├── phase2/                              # Phase 2 experiment code
│   ├── 2a_validation/                   # Probe selectivity, error analysis, bootstrap CI
│   │   ├── experiments/                 # _01_probe_selectivity.py ... _04_gpt2_prompting.py
│   │   ├── src/                         # extract_activations.py, selectivity.py, error_analysis.py, bootstrap.py
│   │   ├── results/                     # Experiment JSONs + activation data (from Colab GPU)
│   │   ├── run_all_2a.py               # Master orchestrator + gate evaluation
│   │   └── colab_runner.py             # Colab-ready entry point
│   └── 2b_scaling/                      # Qwen2.5-1.5B-Instruct scaling
│       ├── src/                         # model_registry.py, dataset_generator.py, efficient_patching.py
│       ├── experiments/                 # circuit_discovery.py, detection_probe.py, intervention.py
│       ├── results/                     # JSON results from Colab runs
│       ├── config/                      # qwen_config.yaml
│       └── colab_runner.py             # Colab-ready entry point (--skip-dataset, --skip-discovery, --skip-probe)
│
├── shared/                              # Model-agnostic library (used by both phases)
│   ├── patching/                        # hooks.py, restoration.py, contrastive.py
│   ├── probing/                         # linear_probe.py, steering_vector.py, control_tasks.py
│   ├── data/                            # pair_generator.py, tokenization.py, loaders.py
│   └── visualization/                   # heatmaps.py, roc_curves.py, circuit_graphs.py
│
├── datasets/                            # Standalone contrastive pair datasets
│   ├── gpt2_arithmetic/                 # 600-pair dataset + extraction script
│   │   └── dataset_extraction.py
│   ├── qwen_arithmetic/                 # (Phase 2B — to be generated)
│   └── dataset_card.md                  # HuggingFace dataset card
│
├── modal_jobs/                          # Remote GPU execution (Modal)
│   ├── phase2a_runner.py
│   ├── phase2b_qwen_runner.py
│   └── phase2b_gemma_runner.py
│
├── tests/                               # Unit tests for shared/ library
│   ├── test_patching.py                 # Restoration score + hook tests
│   ├── test_probing.py                  # Probe + control task tests
│   └── test_data_generation.py          # Pair generator + split tests
│
├── docs/                                # Documentation
│   ├── phase1_findings.md               # Phase 1 results summary
│   ├── phase2_proposal.md               # Full Phase 2 research proposal
│   ├── PROJECT_STRUCTURE.md             # Repository layout rationale & migration plan
│   └── TROUBLESHOOTING.md               # Setup troubleshooting (Python 3.13, CUDA, etc.)
│
├── .github/workflows/tests.yml          # CI — runs tests on push/PR
├── pyproject.toml                       # Package config + optional dependency groups
├── requirements.txt                     # Pinned dependencies for reproducibility
└── .gitignore
```

## Quick Start

### Reproduce Phase 1 results (GPT-2, no GPU needed for exploration)

```bash
git clone https://github.com/ashioyajotham/cot-faithfulness-mech-interp
cd cot-faithfulness-mech-interp

pip install -e ".[phase1]"
jupyter lab phase1/experiments/circuit_discovery/phase1_circuit_discovery.ipynb
```

### Run the test suite (no GPU needed)

```bash
pip install -e ".[phase2a]"
pytest tests/ -v
```

### Phase 2A — validation experiments (GPU for extraction, CPU for analysis)

```bash
pip install -e ".[phase2a]"

# Full pipeline (extraction + experiments, needs GPU)
python phase2/2a_validation/run_all_2a.py --device auto

# Skip extraction if activations already extracted (CPU-only)
python phase2/2a_validation/run_all_2a.py --skip-extraction

# Or run individual experiments
python phase2/2a_validation/experiments/_01_probe_selectivity.py
python phase2/2a_validation/experiments/_02_false_negative_analysis.py
python phase2/2a_validation/experiments/_03_bootstrap_significance.py
```

### Phase 2B — scaling to Qwen2.5-Math (GPU required)

```bash
pip install -e ".[phase2b]"

# Full pipeline: dataset generation + circuit discovery + detection probe + intervention
python phase2/2b_scaling/colab_runner.py --model qwen25-math-1.5b --device auto

# Skip early steps to re-run only intervention (reuses existing dataset/circuit/probe)
python phase2/2b_scaling/colab_runner.py --model qwen25-math-1.5b --device auto \
  --skip-dataset --skip-discovery --skip-probe

# Or run individual steps
python phase2/2b_scaling/src/dataset_generator.py --model qwen25-math-1.5b
python phase2/2b_scaling/experiments/circuit_discovery.py --model qwen25-math-1.5b
python phase2/2b_scaling/experiments/detection_probe.py --model qwen25-math-1.5b
python phase2/2b_scaling/experiments/intervention.py --model qwen25-math-1.5b
```

## Research Questions

### Phase 2A — Validation (complete)

| RQ | Question | Answer |
|----|----------|--------|
| **RQ1** | Does the probe satisfy Hewitt-Liang selectivity? | **Yes** (selectivity=0.110 > 0). Scramble ablation confirms structural reliance (degradation=0.183). |
| **RQ2** | What explains probe false negatives? | **Irreducible noise** — 51 FNs with no carry overrepresentation or magnitude clustering. |
| **RQ3** | Is L7H6's dominance as top shortcut head robust? | **As discriminator, yes** (highest probe coef). **As causal driver, no** — L0MLP dominates restoration scores. These measure different things. |

### Phase 2B — Scaling (complete)

| RQ | Question | Answer |
|----|----------|--------|
| **RQ4** | Does the same circuit structure emerge across scales? | **Yes** — Layer 0 attention is top layer in all three models. Dual-metric divergence replicates across GPT-2, 1.5B, and 7B. Circuit head positions differ but structural patterns are universal. |
| **RQ5** | Does ablating shortcut heads change model behaviour? | **Yes, weakly** — max 3.4% at 7B (6/174 unfaithful pairs flip). Shortcut circuit is highly distributed; at 7B, all top heads are in Layer 23 but ablating 5/28 only shifts 3.4%. |
| **RQ6** | Is faithful reasoning preserved under shortcut ablation? | **Yes, increasingly so** — 97.8–100% at 7B (up from 95.3% at 1.5B). Dual-circuit separation is cleaner at scale. |
| **RQ7** | Do linear probes scale? | **No** — AUC degrades monotonically: 0.98 → 0.617 → 0.536. Full-stream MLP probe is the only method that improves at 7B (AUC 0.616). |

## The `shared/` Library

The `shared/` package contains all model-agnostic code extracted from Phase 1, ensuring one tested implementation across models:

- **`shared.patching`** — hook factories (`make_head_patch_hook`, `make_zero_ablation_hook`), `compute_restoration_score`, `run_full_patching_sweep`, `ContrastivePair` runner
- **`shared.probing`** — `FaithfulnessProbe` (logistic regression with selectivity via Hewitt-Liang control tasks, MDL, bootstrap CI), `SteeringVectorDetector` (difference-of-means), `generate_control_labels`
- **`shared.data`** — `generate_arithmetic_dataset` (addition/subtraction/multiplication/mixed), `verified_tokenize`, `load_jsonl`, `stratified_split`
- **`shared.visualization`** — `plot_head_restoration_heatmap`, `plot_roc_curve`, `plot_circuit_graph`

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `transformer-lens` | Hooked transformer models for activation access |
| `torch` | Tensor computation + GPU |
| `scikit-learn` | Linear probes, metrics |
| `scipy` | Statistical tests (bootstrap, rank correlation) |
| `plotly` / `matplotlib` | Visualization |
| `modal` | Remote GPU execution (Phase 2B) |
| `wandb` | Experiment tracking (Phase 2B) |

See [`pyproject.toml`](pyproject.toml) for full dependency specification with optional groups.

## Related Work

- Wang et al. (2022). [Interpretability in the Wild](https://arxiv.org/abs/2211.00593) — path patching methodology
- Turpin et al. (2023). [Language Models Don't Always Say What They Think](https://arxiv.org/abs/2305.04388) — CoT unfaithfulness evidence
- Chen et al. (2025). [Reasoning Models Don't Always Say What They Think](https://arxiv.org/abs/2505.05410) — Anthropic's behavioural results
- Yang et al. (EMNLP 2025). [Unveiling Internal Reasoning Modes in LLMs](https://aclanthology.org/2025.emnlp-main.136/) — latent reasoning vs shortcuts
- Hewitt & Liang (2019). [Designing and Interpreting Probes with Control Tasks](https://arxiv.org/abs/1909.03368) — probe selectivity standard
- Belinkov (2022). [Probing Classifiers: Promises, Shortcomings, and Advances](https://arxiv.org/abs/2102.12452) — probe validity critique
- Hubinger et al. (2019). [Risks from Learned Optimization](https://arxiv.org/abs/1906.01820) — deceptive alignment framing

## Author

Victor Ashioya (Jotham) — [ashioyajotham.github.io](https://ashioyajotham.github.io) · [GitHub](https://github.com/ashioyajotham)

Bluedot Impact Technical AI Safety Programme · MsingiAI

## License

[MIT](LICENSE)
