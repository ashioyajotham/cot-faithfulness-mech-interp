# When Models Lie to Please: Mechanistic Detection of Unfaithful Chain-of-Thought

This project investigates whether chain-of-thought (CoT) reasoning in language models is *faithful* — whether the model's stated reasoning process reflects its actual internal computation. We use mechanistic interpretability techniques (activation patching, linear probes, steering vectors) to identify separable *faithful* vs *shortcut* circuits in transformers, and build detectors that work at the activation level.

## Project Status

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | GPT-2 Small baseline — circuit discovery, detection probe, dataset | **Complete** (v1.0.0) |
| **Phase 2A** | Validate Phase 1 claims — probe selectivity, error analysis, bootstrap CI | **In progress** |
| **Phase 2B** | Scale to Qwen2.5-Math-7B and Gemma 3 12B IT — intervention experiments | Planned |

## Key Results (Phase 1)

- **23 causally-verified circuit components** identified via activation patching
- **L7H6** identified as the dominant shortcut head (restoration score −0.329)
- **88.1% detection accuracy** (ROC-AUC 0.949) via linear probe on circuit activations
- **Separable faithful/shortcut circuits** confirmed: early-layer faithful heads (L0H1, L0MLP), mid-to-late shortcut heads (L7H6, L5H9)

## Repository Structure

```
cot-faithfulness-mech-interp/
├── phase1/                     # GPT-2 Small baseline — FROZEN (v1.0.0)
├── phase2/
│   ├── 2a_validation/          # Probe selectivity, error analysis, bootstrap CI
│   └── 2b_scaling/             # Qwen2.5-Math-7B + Gemma 3 12B IT pipelines
├── shared/                     # Model-agnostic patching, probing, data utilities
├── datasets/                   # Standalone contrastive pair datasets
├── modal_jobs/                 # Remote GPU execution entry points
├── tests/                      # Test suite
├── docs/                       # Research proposal, project structure, troubleshooting
├── pyproject.toml              # Package definition + optional dependency groups
└── requirements.txt            # Pinned dependencies for reproducibility
```

## Quick Start

```bash
# Clone
git clone https://github.com/ashioyajotham/cot-faithfulness-mech-interp
cd cot-faithfulness-mech-interp

# Phase 2A (CPU only, fast)
pip install -e ".[phase2a]"
pytest tests/ -v

# Phase 2B (GPU, full stack)
pip install -e ".[phase2b]"
modal run modal_jobs/phase2b_qwen_runner.py
```

## Research Questions (Phase 2)

1. **Probe validity (RQ1):** Does the linear probe satisfy the Hewitt-Liang selectivity criterion?
2. **Failure modes (RQ2):** What distinguishes the 11 high-confidence false negatives?
3. **Statistical robustness (RQ3):** Is L7H6's dominance stable under bootstrap resampling?
4. **Scaling (RQ4):** Does the same circuit structure emerge in Qwen2.5-Math-7B and Gemma 3 12B IT?
5. **Intervention (RQ5):** Does ablating shortcut heads shift model behaviour on unfaithful examples?

## Related Work

- Wang et al. (2022). [Interpretability in the Wild](https://arxiv.org/abs/2211.00593) — path patching methodology
- Turpin et al. (2023). [Language Models Don't Always Say What They Think](https://arxiv.org/abs/2305.04388) — CoT unfaithfulness evidence
- Chen et al. (2025). [Reasoning Models Don't Always Say What They Think](https://arxiv.org/abs/2505.05410) — Anthropic's behavioural results
- Yang et al. (EMNLP 2025). [Unveiling Internal Reasoning Modes in LLMs](https://aclanthology.org/2025.emnlp-main.136/) — latent reasoning vs shortcuts
- Hewitt & Liang (2019). [Designing and Interpreting Probes with Control Tasks](https://arxiv.org/abs/1909.03368) — probe selectivity standard

## Author

Victor Ashioya (Jotham) — [ashioyajotham.github.io](https://ashioyajotham.github.io) · [GitHub](https://github.com/ashioyajotham)

Bluedot Impact Technical AI Safety Programme · MsingiAI

## License

[MIT](LICENSE)
