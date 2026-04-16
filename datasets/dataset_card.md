---
license: cc-by-4.0
task_categories:
  - text-classification
language:
  - en
tags:
  - mechanistic-interpretability
  - chain-of-thought
  - faithfulness
  - ai-safety
size_categories:
  - 1K<n<10K
---

# CoT Faithfulness Contrastive Dataset

A contrastive dataset for detecting Chain-of-Thought faithfulness via
mechanistic interpretability.  Each example is a paired clean/corrupted
arithmetic prompt designed to activate either faithful or shortcut
circuits in a transformer.

## Sub-datasets

| Directory | Model | Pairs | Task types |
|-----------|-------|-------|------------|
| `gpt2_arithmetic/` | GPT-2 Small | 500 | 2-digit addition |
| `qwen_arithmetic/` | Qwen2.5-Math-7B | 2,000 (planned) | 2-digit, 3-digit, subtraction, word problems |

## Schema

Each example contains:
- `prompt` — full prompt text
- `label` — 0 (faithful) or 1 (unfaithful)
- `correct_answer` / `cot_answer` — ground truth vs CoT-implied answer
- `requires_carry` — whether the arithmetic requires carrying
- `corruption_type` / `corruption_magnitude` — how the CoT was corrupted

## Citation

```bibtex
@misc{ashioya2026cotfaithfulness,
  title={When Models Lie to Please: Scaling Mechanistic Detection of Unfaithful Chain-of-Thought},
  author={Ashioya, Victor},
  year={2026},
  url={https://github.com/ashioyajotham/cot-faithfulness-mech-interp}
}
```
