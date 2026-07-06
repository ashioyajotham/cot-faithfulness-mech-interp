# Mechanistic Analysis of Chain-of-Thought

This project investigates whether chain-of-thought (CoT) reasoning in language models is *faithful* — whether the model's stated reasoning process reflects its actual internal computation. We use mechanistic interpretability techniques (activation patching, linear probes, steering vectors) to identify separable *faithful* vs *shortcut* circuits in transformers, and build detectors that work at the activation level.

The long-term safety case: if models develop separable circuits for "produce a CoT that looks correct" and "compute the actual answer," CoT monitoring gives weaker guarantees than assumed. This work builds detectors that operate below the text surface.

## Project Status

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | GPT-2 Small baseline — circuit discovery, detection probe, dataset | **Complete** (`v1.0.0`) |
| **Phase 2A** | Validate Phase 1 claims — probe selectivity, error analysis, bootstrap CI | **Complete** |
| **Phase 2B** | Scale to Qwen 1.5B, Qwen 7B, Gemma 2 2B — circuit discovery, probes, intervention | **Complete** |

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

Full pipeline ran on Qwen2.5-1.5B-Instruct (T4, 50 min), Qwen2.5-7B-Instruct (A100, 64 min), and Gemma 2 2B (A100, 20 min).

#### Four-Model Comparison

| Finding | GPT-2 (124M) | Qwen 1.5B | Qwen 7B | **Gemma 2 2B** |
|---------|:---:|:---:|:---:|:---:|
| Top layer component | L0 attn (0.721) | L0 attn (1.023) | L0 attn (0.944) | **L20 attn (0.443)** |
| Dual-metric divergence | L7H6 ≠ L0MLP | L20H9 ≠ L20H5 | L23H12 ≠ L23H7 | **L25H1 ≠ L20H7** |
| Signal distribution | DISTRIBUTED | DISTRIBUTED | DISTRIBUTED | **LOCALIZED** |
| Circuit probe AUC | 0.98 | 0.617 | 0.536 | **0.633** |
| Selectivity | 0.110 | 0.069 | 0.037 | **0.154** |
| Best intervention | N/A | 2.6% | 3.4% | **3.4%** |
| Faithful preservation | N/A | 95.3% | 97.8–100% | **88–100%** |
| Unfaithful ratio | ~50% | 60% | 69% | **62%** |
| Valid rate | — | 41% | 62% | **17%** |

#### Key Findings (updated with Gemma)

1. **Dual-metric divergence is universal** — replicates in 4/4 models across two architecture families (GPT-2, Qwen, Gemma)
2. **Layer 0 bottleneck is architecture-dependent** — #1 in global-attention models (GPT-2, Qwen), but **NOT** in Gemma 2 (which uses sliding window attention, so L0 can't attend to the full input; the bottleneck shifts to L20)
3. **Linear probes degrade within architecture** — AUC: 0.98 → 0.617 → 0.536 (GPT-2 → Qwen scaling), but Gemma 2B bucks the trend (0.633) with highest selectivity (0.154)
4. **Signal localisation is architecture-dependent** — distributed in GPT-2/Qwen, but **localized** in Gemma (circuit probe beats random layers)
5. **Shortcut circuits consolidate differently** — Qwen 7B concentrates in L23; Gemma distributes across L6–L25
6. **Larger models are more unfaithful** — within Qwen: 60% → 69%

#### Probe Architecture Comparison

| Probe Type | Qwen 1.5B | Qwen 7B | Gemma 2B |
|-----------|:---------:|:-------:|:--------:|
| Circuit linear | 0.617 | 0.536 | **0.633** |
| Circuit MLP | 0.532 | 0.527 | 0.570 |
| Full-stream linear | **0.660** | 0.579 | **0.680** |
| Full-stream MLP | 0.561 | **0.616** | 0.574 |

**Gemma insight**: Full-stream linear is the best probe across all models. Signal localisation in Gemma makes circuit probes more effective than in Qwen at comparable scale.

Phase 2B results are at `phase2/2b_scaling/results/{qwen25-math-1.5b,qwen25-math-7b,gemma2-2b}/`.

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

### Phase 2B — Scaling & Cross-Architecture (complete)

| RQ | Question | Answer |
|----|----------|--------|
| **RQ4** | Does the same circuit structure emerge across architectures? | **Partially** — Dual-metric divergence replicates in 4/4 models. But Layer 0 bottleneck is architecture-dependent: #1 in global-attention models (GPT-2, Qwen), NOT in Gemma 2 (sliding window attention shifts bottleneck to L20). |
| **RQ5** | Does ablating shortcut heads change model behaviour? | **Yes, weakly** — max 3.4% across both Qwen 7B and Gemma 2B. Shortcut is distributed regardless of architecture. |
| **RQ6** | Is faithful reasoning preserved under shortcut ablation? | **Yes** — 88–100% across all models. Dual-circuit separation holds cross-architecture. |
| **RQ7** | Do linear probes scale? | **Within-family: no** (Qwen AUC: 0.617 → 0.536). **Cross-architecture: architecture matters more than scale** — Gemma 2B (0.633) beats Qwen 7B (0.536) despite being smaller. |
| **RQ8** | Is the faithfulness signal localized or distributed? | **Architecture-dependent** — distributed in GPT-2/Qwen, but **localized** in Gemma 2B (circuit beats random layers, gap=+0.071). |

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
