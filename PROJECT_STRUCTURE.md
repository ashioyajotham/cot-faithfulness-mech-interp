# Project Structure & Technical Roadmap
## When Models Lie to Please — Mechanistic Detection of Unfaithful Chain-of-Thought

> **Reorganisation rationale:** Phase 1 (GPT-2 Small baseline) is complete and archival.
> It moves into `phase1/` as a self-contained unit. Phase 2 gets a dedicated root folder
> with two sequential subprojects — `2a_validation/` (repair the Phase 1 claims) and
> `2b_scaling/` (scale to capable models). Shared code is extracted into `shared/` so
> it never duplicates. The dataset is a standalone contribution and lives at the root.

---

## Repository Layout

```
cot-faithfulness-mech-interp/
│
├── phase1/                              # GPT-2 Small baseline — FROZEN
│   ├── experiments/
│   │   ├── circuit_discovery/
│   │   │   ├── phase1_circuit_discovery.ipynb       # Layer-level patching (Stage 1A + 1B)
│   │   │   └── phase1_5_head_level.ipynb            # Per-head patching via hook_z
│   │   └── faithfulness_detection/
│   │       └── phase2_faithfulness_detector.ipynb   # Linear probe, steering vector, hybrid
│   ├── src/
│   │   ├── analysis/
│   │   │   ├── attribution_graphs.py
│   │   │   └── faithfulness_detector.py
│   │   ├── models/
│   │   │   └── gpt2_wrapper.py
│   │   ├── interventions/
│   │   │   └── targeted_interventions.py
│   │   ├── data/
│   │   │   └── data_generation.py
│   │   └── visualization/
│   │       └── interactive_plots.py
│   ├── config/
│   │   ├── model_config.yaml
│   │   └── experiment_config.yaml
│   ├── results/                         # Phase 1 outputs — do not modify
│   └── README.md                        # Phase 1 standalone documentation
│
├── phase2/
│   │
│   ├── 2a_validation/                   # Repair and validate Phase 1 claims
│   │   ├── experiments/
│   │   │   ├── 01_probe_selectivity.ipynb        # Hewitt-Liang control tasks + MDL probe
│   │   │   ├── 02_false_negative_analysis.ipynb  # High-confidence error taxonomy
│   │   │   ├── 03_bootstrap_significance.ipynb   # CIs for L7H6, restoration score rankings
│   │   │   └── 04_gpt2_prompting_formats.ipynb   # Completion-style arithmetic test
│   │   ├── src/
│   │   │   ├── selectivity.py           # Control task generation, selectivity metric
│   │   │   ├── error_analysis.py        # False negative clustering utilities
│   │   │   └── bootstrap.py             # Bootstrap CI, rank stability tests
│   │   ├── config/
│   │   │   └── validation_config.yaml
│   │   └── results/
│   │
│   └── 2b_scaling/                      # Scale validated methodology to capable models
│       ├── experiments/
│       │   ├── qwen/
│       │   │   ├── 01_circuit_discovery.ipynb    # Qwen2.5-Math-7B patching pipeline
│       │   │   ├── 02_detection.ipynb             # Probe training + cross-task eval
│       │   │   └── 03_interventions.ipynb         # Shortcut head ablation experiments
│       │   └── gemma/
│       │       ├── 01_circuit_discovery.ipynb    # Gemma 3 12B IT patching pipeline
│       │       ├── 02_sae_interpretation.ipynb   # Gemma Scope 2 SAE analysis
│       │       └── 03_cross_model_comparison.ipynb
│       ├── src/
│       │   ├── model_registry.py        # HookedTransformer loader for Qwen + Gemma
│       │   ├── efficient_patching.py    # Two-pass patching (layer → head)
│       │   ├── sae_utils.py             # Gemma Scope 2 SAE loading + feature search
│       │   └── cross_model.py           # Head-position alignment across architectures
│       ├── config/
│       │   ├── qwen_config.yaml
│       │   └── gemma_config.yaml
│       └── results/
│
├── shared/                              # Model-agnostic utilities used by both phases
│   ├── patching/
│   │   ├── __init__.py
│   │   ├── hooks.py                     # hook_z, hook_attn_out, hook_mlp_out wrappers
│   │   ├── restoration.py               # Restoration score computation
│   │   └── contrastive.py               # Clean/corrupted pair runner
│   ├── probing/
│   │   ├── __init__.py
│   │   ├── linear_probe.py              # Logistic regression probe with selectivity
│   │   ├── steering_vector.py           # Diff-of-means, projection, threshold search
│   │   └── control_tasks.py             # Hewitt-Liang random label generation
│   ├── data/
│   │   ├── __init__.py
│   │   ├── pair_generator.py            # Contrastive pair construction (arithmetic, logic)
│   │   ├── tokenization.py              # Verified tokenization (handles space-prefix)
│   │   └── loaders.py                   # Dataset loading + stratified splits
│   └── visualization/
│       ├── __init__.py
│       ├── heatmaps.py                  # Head restoration heatmaps
│       ├── roc_curves.py                # ROC + AUC plots
│       └── circuit_graphs.py            # NetworkX circuit graph export
│
├── datasets/                            # Standalone data contribution
│   ├── gpt2_arithmetic/
│   │   ├── faithful_600.jsonl           # 300 faithful + 300 unfaithful, GPT-2 format
│   │   ├── metadata.json                # Carry flags, operand ranges, corruption severity
│   │   └── README.md
│   ├── qwen_arithmetic/                 # Generated in Phase 2B
│   │   └── .gitkeep
│   └── dataset_card.md                  # HuggingFace dataset card
│
├── modal_jobs/                          # Remote compute entry points
│   ├── phase2a_runner.py
│   ├── phase2b_qwen_runner.py
│   └── phase2b_gemma_runner.py
│
├── tests/
│   ├── test_patching.py
│   ├── test_probing.py
│   └── test_data_generation.py
│
├── docs/
│   ├── phase1_findings.md               # Frozen Phase 1 write-up
│   ├── phase2_proposal.md               # Research proposal (this project)
│   └── PROJECT_STRUCTURE.md             # This file
│
├── .github/
│   └── workflows/
│       └── tests.yml                    # CI: run tests on push
│
├── pyproject.toml                       # Package definition + dependencies
├── requirements.txt                     # Pinned for reproducibility
└── README.md                            # Top-level project overview
```

---

## Design Decisions

### Why `phase1/` is frozen

Phase 1 is a completed, peer-reviewed artefact. It should reproduce cleanly from a fresh clone without any dependency on Phase 2 code. Keeping it as a self-contained directory with its own `src/`, `config/`, and `results/` means the Demo Day submission and Bluedot report are permanently reproducible. Nothing in `phase1/` is modified from here forward.

### Why `shared/` exists

The patching logic (`hook_z`, restoration scores, contrastive runners), the probing logic (linear probe, steering vector), and the data generation pipeline are model-agnostic. Without a shared layer, those ~500 lines of carefully-debugged code duplicate across GPT-2, Qwen, and Gemma. `shared/` is the only code that is imported across phases; both `phase1/src/` and `phase2/*/src/` can depend on it.

### Why `2a_validation/` and `2b_scaling/` are siblings, not nested

They are sequential but independently reviewable. A collaborator who only cares about the probe selectivity results (2A) does not need to touch the Qwen pipeline (2B). Separating them as siblings makes the CI setup and compute footprint easier to reason about — 2A runs on CPU in minutes, 2B needs GPU hours on Modal.

### Why `datasets/` is at the root

The contrastive pair dataset is a standalone research contribution independent of any model or phase. It will be released on HuggingFace under a separate card. Keeping it at the root, versioned separately, means it can be cited independently of the broader project and consumed without cloning the full repo.

---

## Phase 2A — Validation

### Goal
Earn the right to make Phase 1 claims with statistical confidence, and characterise
the probe's limitations before scaling.

### Experiments and Sequencing

```
01_probe_selectivity  →  02_false_negative_analysis  →  03_bootstrap_significance
                                                                    ↓
                                               04_gpt2_prompting_formats (parallel)
```

| Notebook | Core question | Method | Runtime |
|----------|--------------|--------|---------|
| 01 | Is the probe selective? | Hewitt-Liang control tasks + MDL probe + scramble baseline | ~2h CPU |
| 02 | What are the 11 confident errors? | Cluster by carry, magnitude, corruption type | ~30min CPU |
| 03 | Is L7H6 statistically dominant? | Bootstrap CI (n=1000), cross-pair-type rank test | ~1h CPU |
| 04 | Can GPT-2 do arithmetic at all? | Completion-style vs instruction-style formatting | ~1h CPU |

### Key Metrics

```python
# 01 — Probe selectivity
selectivity = linguistic_task_accuracy - control_task_accuracy
# Target: selectivity > 0.15 (probe reflects structure, not surface features)

# 03 — L7H6 bootstrap CI
ci_lower, ci_upper = bootstrap_rank_ci(restoration_scores, n_iterations=1000, alpha=0.05)
# Target: L7H6 rank 1 in >95% of bootstrap samples

# 03 — Cross-pair-type stability  
rank_correlation = scipy.stats.spearmanr(ranks_faithfulness_pairs, ranks_shortcut_pairs)
# Target: ρ > 0.7 across all three pair types
```

### Tools

| Tool | Version | Purpose |
|------|---------|---------|
| TransformerLens | ≥2.0 | Hook infrastructure, GPT-2 loading |
| scikit-learn | ≥1.4 | LogisticRegression, train/test split |
| scipy | ≥1.12 | Bootstrap CI, Spearman correlation |
| numpy | ≥1.26 | Array ops |
| matplotlib / seaborn | latest | Visualisation |
| W&B | ≥0.16 | Experiment tracking |

### Configuration (`validation_config.yaml`)

```yaml
model:
  name: gpt2
  device: cpu  # 2A runs on CPU — no GPU needed

probe:
  n_bootstrap_iterations: 1000
  alpha: 0.05
  selectivity_threshold: 0.15

control_tasks:
  n_random_labels: 5           # Run 5 random label seeds, average selectivity
  stratify_by: prompt_number   # Match Hewitt-Liang protocol exactly

false_negatives:
  confidence_threshold: 0.99   # Analyse all examples with p(unfaithful) > 0.99
  cluster_features:
    - carry_required
    - sum_magnitude
    - corruption_severity
    - cot_corruption_type
```

---

## Phase 2B — Scaling

### Goal
Replicate the Phase 1.5 methodology on arithmetic-capable models. Run the intervention
experiments that GPT-2 could not support. Produce a cross-architecture shortcut head
comparison grounded by SAE-level interpretation.

### Model Targets

| Model | Parameters | Heads | Layers | MATH accuracy | Justification |
|-------|-----------|-------|--------|--------------|--------------|
| Qwen2.5-Math-7B | 7B | 28 heads/layer | 32 | 85% MATH | Primary target; HookedTransformer-compatible; trained for arithmetic |
| Gemma 3 12B IT | 12B | 8 heads/layer | 48 | 79% MATH | Cross-architecture check; Gemma Scope 2 SAEs available |

### Experiment Pipeline (Qwen)

```
Dataset construction (2k pairs, 4 arithmetic types)
         ↓
Layer-level patching (hook_attn_out) — identify top 5 layers  [Pass 1]
         ↓
Per-head patching (hook_z) within top 5 layers only           [Pass 2]
         ↓
Component classification (faithful / shortcut / harmful)
         ↓
Detection: probe + steering vector + hybrid (within-task)
         ↓
Cross-task evaluation (train 2-digit → test 3-digit + subtraction)
         ↓
Intervention: ablate top shortcut heads → measure shift rate
```

### Efficient Two-Pass Patching

With 896 heads in Qwen2.5-Math-7B, exhaustive per-head patching is expensive. The
two-pass approach reduces the search space from 896 to ~168 runs:

```python
# Pass 1: layer-level (32 layers × 2 types = 64 runs per pair)
layer_scores = run_layer_level_patching(pair, model)
top_k_layers = get_top_k_layers(layer_scores, k=6)

# Pass 2: per-head within top layers only (6 layers × 28 heads = 168 runs per pair)
head_scores = run_head_level_patching(pair, model, layers=top_k_layers)
```

This is implemented in `shared/patching/efficient_patching.py` and respects the
finding from Phase 1.5 that the faithful components (L0H1, L0MLP) and shortcut
components (L7H6) cluster in specific layer ranges. The pass-1 layer filter
ensures those regions are always included.

### Dataset Extension

Phase 1 used 9 contrastive pairs. Phase 2B targets 2,000:

| Type | Count | Description |
|------|-------|-------------|
| 2-digit addition (no carry) | 400 | Replication of Phase 1 on Qwen |
| 2-digit addition (with carry) | 400 | Targets the carry-circuit gap from Phase 1 |
| 3-digit addition | 400 | Harder computation |
| 2-digit subtraction | 400 | Tests pathway generality |
| Mixed word problems | 400 | Out-of-distribution — wraps arithmetic in natural language |

Each pair has metadata stored in `datasets/qwen_arithmetic/metadata.json`:

```json
{
  "pair_id": "q_0042",
  "a": 347,
  "b": 186,
  "correct": 533,
  "carry_required": true,
  "carry_position": "ones",
  "corruption_type": "step_value",
  "corruption_severity": 0.4,
  "cot_length_faithful": 23,
  "cot_length_unfaithful": 23,
  "task_type": "3digit_addition"
}
```

### SAE Integration (Gemma)

Gemma Scope 2 SAEs are available for Gemma 3 models. For each shortcut head identified
in Gemma, run:

```python
# Find SAE features active on unfaithful examples but not faithful
faithful_features   = sae.encode(faithful_residual_stream)    # shape (n, d_sae)
unfaithful_features = sae.encode(unfaithful_residual_stream)

# Differential activation
delta_features = unfaithful_features.mean(0) - faithful_features.mean(0)
top_features   = delta_features.topk(20).indices  # Most differentially active

# Lookup interpretations
for feat_idx in top_features:
    print(sae.feature_labels[feat_idx])  # Human-readable feature description
```

This produces the first interpretable description of *what* a shortcut head is
attending to — not just that it matters causally.

### Intervention Metrics

```python
# Primary: shift rate
# Did ablating shortcut heads change model output from correct → CoT-consistent?
shift_rate = n_shifted / n_unfaithful_examples
# Target: shift_rate > 0.20

# Secondary: faithful accuracy preservation  
# Did ablation harm the faithful reasoning pathway?
faithful_acc_with_ablation    = eval(faithful_examples, ablated_model)
faithful_acc_without_ablation = eval(faithful_examples, full_model)
accuracy_preservation = faithful_acc_with_ablation / faithful_acc_without_ablation
# Target: > 0.85 (ablation should not significantly harm faithful cases)

# Tertiary: cascade curve
# Does ablating more shortcut heads monotonically increase shift rate?
shift_rates_by_n_ablated = [
    eval_shift_rate(ablate_top_n_shortcut_heads(model, n))
    for n in range(1, 6)
]
```

### Tools

| Tool | Version | Purpose |
|------|---------|---------|
| TransformerLens | ≥2.0 | Hook infrastructure — must support Qwen2.5 and Gemma 3 |
| transformer_lens.utilities | latest | Model loading helpers |
| gemma_scope | latest | SAE loading for Gemma 3 |
| Modal | latest | Remote GPU execution (A100/H100) |
| W&B | ≥0.16 | Experiment tracking + sweep management |
| torch | ≥2.2 | Tensor operations |
| einops | latest | Tensor reshaping (hook_z manipulation) |
| datasets (HuggingFace) | latest | Dataset loading + upload |
| pytest | ≥7.0 | Test suite |

### Model Configuration (`qwen_config.yaml`)

```yaml
model:
  name: Qwen/Qwen2.5-Math-7B-Instruct
  device: cuda
  dtype: float16
  fold_ln: false
  center_writing_weights: false
  center_unembed: false

patching:
  strategy: two_pass
  pass1_components: [hook_attn_out, hook_mlp_out]
  pass2_top_k_layers: 6
  pass2_component: hook_z
  logit_diff_position: last_token

probe:
  feature_dim: null    # Auto-computed from component list
  test_size: 0.2
  random_state: 42
  class_weight: balanced
  cross_task_eval: true

intervention:
  n_heads_to_ablate: [1, 2, 3, 4, 5]
  measure_faithful_preservation: true

compute:
  provider: modal
  gpu: A100
  memory_gb: 40
  timeout_minutes: 90
```

---

## Shared Library API

`shared/` is the only code that both phases import. It must be model-agnostic and
thoroughly tested. The key abstractions:

### `shared/patching/hooks.py`

```python
def make_head_patch_hook(
    clean_cache: Dict[str, torch.Tensor],
    layer: int,
    head_idx: int
) -> Callable:
    """Patch a single attention head's hook_z output with clean activations."""

def make_mlp_patch_hook(
    clean_cache: Dict[str, torch.Tensor],
    layer: int
) -> Callable:
    """Patch a full MLP layer output with clean activations."""

def make_zero_ablation_hook(layer: int, head_idx: Optional[int] = None) -> Callable:
    """Zero out a component. If head_idx is None, ablates entire attention layer."""
```

### `shared/patching/restoration.py`

```python
def compute_restoration_score(
    model: HookedTransformer,
    pair: ContrastivePair,
    hook_name: str,
    clean_cache: Dict,
    corrupted_tokens: torch.Tensor,
    correct_id: int,
    incorrect_id: int
) -> float:
    """
    Restoration = (patched_diff - corrupted_diff) / (clean_diff - corrupted_diff)
    Returns NaN if gap < threshold (pair is degenerate).
    """

def run_full_patching_sweep(
    model: HookedTransformer,
    pairs: List[ContrastivePair],
    strategy: Literal["exhaustive", "two_pass"] = "two_pass",
    top_k_layers: int = 6
) -> Dict[str, float]:
    """Run patching over all components, return averaged restoration scores."""
```

### `shared/probing/linear_probe.py`

```python
class FaithfulnessProbe:
    """
    Linear probe with built-in selectivity testing.
    Wraps LogisticRegression and exposes:
      - fit(X, y)
      - predict(X)
      - selectivity(X, y, n_control_seeds=5) → float
      - bootstrap_ci(X, y, n_iterations=1000) → (lower, upper)
    """
    def selectivity(self, X: np.ndarray, y: np.ndarray, n_seeds: int = 5) -> float:
        """
        Hewitt-Liang selectivity: linguistic_acc - mean(control_task_acc over n_seeds).
        Control task: shuffle labels by word type (here: shuffle by pair_id mod vocab_size).
        """

    def mdl_score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Minimum description length compression score (Voita & Titov 2020).
        Lower = probe is encoding more; higher = probe is memorising.
        """
```

### `shared/data/pair_generator.py`

```python
@dataclass
class ContrastivePair:
    pair_id: str
    clean_prompt: str
    corrupted_prompt: str
    correct_token: str
    incorrect_token: str
    pair_type: str
    metadata: Dict[str, Any]      # carry_required, corruption_severity, etc.

def generate_arithmetic_dataset(
    n_pairs: int,
    task_types: List[str],        # ["2digit_add", "3digit_add", "2digit_sub", ...]
    seed: int = 42
) -> Tuple[List[ContrastivePair], List[ContrastivePair]]:
    """Returns (faithful_examples, unfaithful_examples)."""
```

---

## Experiment Naming Convention

All results files follow: `{phase}_{experiment_id}_{model}_{timestamp}.json`

Examples:
- `2a_01_probe_selectivity_gpt2_20260501.json`
- `2b_01_circuit_discovery_qwen25math7b_20260601.json`
- `2b_03_interventions_qwen25math7b_20260615.json`

W&B project naming:
- `cot-faithfulness-phase2a` — all validation experiments
- `cot-faithfulness-phase2b-qwen` — Qwen scaling
- `cot-faithfulness-phase2b-gemma` — Gemma scaling

---

## Dependencies and Environment

### `pyproject.toml` (key sections)

```toml
[project]
name = "cot-faithfulness"
version = "2.0.0"
python = ">=3.10"

[project.optional-dependencies]
phase1 = [
    "transformer-lens>=2.0",
    "torch>=2.0",
    "transformers>=4.40,<4.46",   # Pin: TRANSFORMERS_CACHE deprecation in >=4.46
    "scikit-learn>=1.4",
    "einops",
    "jaxtyping",
    "matplotlib",
    "networkx",
    "wandb>=0.16",
]
phase2a = [
    "cot-faithfulness[phase1]",
    "scipy>=1.12",
    "seaborn",
]
phase2b = [
    "cot-faithfulness[phase2a]",
    "modal",
    "datasets",                   # HuggingFace datasets
    # gemma_scope installed from source (see docs/setup.md)
]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
```

### Environment setup

```bash
# Clone and set up
git clone https://github.com/ashioyajotham/cot-faithfulness-mech-interp
cd cot-faithfulness-mech-interp

# Phase 2A (CPU only, fast)
pip install -e ".[phase2a]"

# Phase 2B (GPU, full stack)
pip install -e ".[phase2b]"

# Run tests
pytest tests/ -v

# Run Phase 2A locally
jupyter lab phase2/2a_validation/experiments/01_probe_selectivity.ipynb

# Run Phase 2B on Modal
modal run modal_jobs/phase2b_qwen_runner.py
```

---

## Compute and W&B Tracking

### Modal Job Structure (`modal_jobs/phase2b_qwen_runner.py`)

```python
import modal

app = modal.App("cot-faithfulness-phase2b-qwen")

VOLUME = modal.Volume.from_name("cot-faithfulness-checkpoints", create_if_missing=True)

@app.function(
    gpu="A100",
    memory=40960,
    timeout=5400,
    volumes={"/checkpoints": VOLUME},
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def run_circuit_discovery(experiment_id: str, config_override: dict = None):
    """Entry point for Qwen circuit discovery on Modal."""
    import wandb
    from phase2.src.model_registry import load_model
    from shared.patching.restoration import run_full_patching_sweep

    wandb.init(project="cot-faithfulness-phase2b-qwen", name=experiment_id)
    model = load_model("Qwen/Qwen2.5-Math-7B-Instruct")
    # ... experiment logic ...
    VOLUME.commit()
```

### W&B Logging Conventions

Every experiment logs:
```python
wandb.log({
    # Core metrics
    "probe/accuracy": float,
    "probe/roc_auc": float,
    "probe/selectivity": float,   # Phase 2A only
    
    # Circuit metrics
    "circuit/n_faithful_heads": int,
    "circuit/n_shortcut_heads": int,
    "circuit/top_shortcut_head": str,     # e.g. "L7H6"
    "circuit/top_shortcut_restoration": float,
    
    # Intervention metrics (2B only)
    "intervention/shift_rate": float,
    "intervention/faithful_preservation": float,
    
    # Metadata
    "model": str,
    "n_pairs": int,
    "task_types": List[str],
})
```

---

## Migration from Current Structure

The current repo has `experiments/01_circuit_discovery/` etc. at the root. Here is
the migration sequence — do not move files until all are ready to move at once to
avoid breaking notebook imports.

```bash
# Step 1: Create new directories
mkdir -p phase1/experiments/circuit_discovery
mkdir -p phase1/experiments/faithfulness_detection
mkdir -p phase1/src phase1/config phase1/results
mkdir -p phase2/2a_validation/experiments
mkdir -p phase2/2a_validation/src phase2/2a_validation/config phase2/2a_validation/results
mkdir -p phase2/2b_scaling/experiments/qwen
mkdir -p phase2/2b_scaling/experiments/gemma
mkdir -p phase2/2b_scaling/src phase2/2b_scaling/config phase2/2b_scaling/results
mkdir -p shared/patching shared/probing shared/data shared/visualization
mkdir -p datasets/gpt2_arithmetic datasets/qwen_arithmetic
mkdir -p modal_jobs tests

# Step 2: Move Phase 1 notebooks
mv experiments/01_circuit_discovery/phase1_circuit_discovery.ipynb \
   phase1/experiments/circuit_discovery/
mv experiments/01_circuit_discovery/phase1_5_head_level.ipynb \
   phase1/experiments/circuit_discovery/
mv experiments/02_faithfulness_detection/phase2_faithfulness_detector.ipynb \
   phase1/experiments/faithfulness_detection/

# Step 3: Move Phase 1 src
mv src/ phase1/src/
mv config/ phase1/config/
mv results/ phase1/results/

# Step 4: Move dataset
mv data/ datasets/gpt2_arithmetic/   # rename and move existing data files

# Step 5: Initialise shared/ with extracted common code
# (done manually — extract from phase1/src/analysis/ + phase1/src/interventions/)

# Step 6: Update imports in phase1 notebooks
# Change: from src.analysis import ... 
# To:     from phase1.src.analysis import ... OR sys.path manipulation in notebook header

# Step 7: Add phase1/README.md noting this is frozen
# Step 8: Add top-level README.md with new structure overview
```

---

## Git Conventions for Phase 2

```
Branch naming:
  phase2a/probe-selectivity
  phase2a/error-analysis
  phase2a/bootstrap-ci
  phase2b/qwen-circuit-discovery
  phase2b/qwen-interventions
  phase2b/gemma-sae

Commit format:
  [2A] add Hewitt-Liang control task implementation
  [2B/qwen] two-pass patching: 64 → 168 component runs
  [shared] extract restoration score computation into shared/patching/
  [data] add 3-digit addition pairs to qwen dataset
  [fix] tokenization: handle space-prefixed answer tokens in Qwen

Tags:
  v1.0.0  — Phase 1 frozen (tag this before migration)
  v2.0.0  — Phase 2A complete + validated claims
  v2.1.0  — Phase 2B Qwen results
  v2.2.0  — Phase 2B Gemma + cross-model comparison
```

---

## Success Gates

| Gate | Condition | What it unlocks |
|------|-----------|-----------------|
| 2A-G1 | Selectivity > 0 on at least one baseline | Phase 1 detection claim holds; proceed to 2B |
| 2A-G2 | L7H6 rank 1 in >90% of bootstrap samples | Single-head dominance claim holds |
| 2A-G3 | False negative cluster identified | Dataset augmentation prescription ready |
| 2B-G1 | Qwen cross-task probe accuracy > 70% | Methodology generalises beyond GPT-2 |
| 2B-G2 | Intervention shift rate > 20% | Causal shortcut circuit confirmed on capable model |
| 2B-G3 | Gemma shortcut head SAE interpretation > 0 features | First mechanistic interpretation of shortcut routing |

If 2A-G1 fails (probe not selective), the Phase 1 claim is reframed as "circuit activations
encode prompt-construction patterns correlated with faithfulness" — weaker but publishable,
and the Phase 2B probe trains on selectivity-aware features only.

---

*Last updated: April 2026*
*Maintainer: Victor Ashioya — ashioyajotham.github.io*
