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

# CoT Faithfulness Dataset

A contrastive dataset for detecting Chain-of-Thought faithfulness via mechanistic interpretability.

## Dataset Description

This dataset contains pairs of arithmetic problems where:
- **Faithful** examples have correct Chain-of-Thought reasoning
- **Unfaithful** examples have incorrect CoT, designed to test if models follow their stated reasoning or take shortcuts

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("ashioyajotham/cot-faithfulness-arithmetic")

# Access examples
for example in dataset["train"]:
    print(f"Prompt: {example['prompt']}")
    print(f"Label: {example['label_name']}")
```

## Citation

```bibtex
@misc{ashioya2025cotfaithfulness,
  title={Detecting Unfaithful Chain-of-Thought via Mechanistic Interpretability},
  author={Ashioya, Victor},
  year={2025},
  url={https://github.com/ashioyajotham/cot-faithfulness-mech-interp}
}
```
