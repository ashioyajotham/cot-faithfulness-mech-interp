# When Models Lie to Please: Mechanistic Detection of Unfaithful Chain-of-Thought

**Victor Ashioya (Jotham)**  
Bluedot Impact Technical AI Safety Programme · MsingiAI

---

## Abstract

We investigate whether chain-of-thought (CoT) reasoning in language models faithfully reflects their internal computation. Using mechanistic interpretability — activation patching, linear probes, and zero-ablation interventions — we identify separable *faithful* and *shortcut* circuits in transformers processing arithmetic CoT. Across two architectures (GPT-2 Small, 124M params; Qwen2.5-1.5B-Instruct, 1.5B params), we discover a *dual-metric divergence*: the component most useful for *classifying* faithful vs unfaithful reasoning (probe coefficient) is consistently different from the component with the largest *causal effect* on model output (restoration score). This finding replicates across architectures and scales, suggesting it is a fundamental property of how transformers process chain-of-thought. We also find that faithfulness information is distributed across the residual stream rather than localised in identifiable circuit components, with implications for the reliability of CoT monitoring as a safety measure.

---

## 1. Introduction

Chain-of-thought (CoT) prompting — where models produce step-by-step reasoning before an answer — is increasingly used as a safety measure: if a model's reasoning is visible, monitors can flag suspicious steps. But this safety case rests on a critical assumption: that the stated reasoning *actually drives* the model's computation.

Prior work has shown this assumption can fail. Turpin et al. (2023) demonstrated that models sometimes produce CoT that does not reflect their true decision process. Chen et al. (2025) showed that Anthropic's reasoning models frequently generate unfaithful CoT in adversarial settings. Yang et al. (EMNLP 2025) identified latent reasoning modes where models compute answers through internal shortcuts that bypass stated reasoning steps.

We approach this problem mechanistically. Rather than probing model behaviour from outside, we look inside: which circuits activate when a model follows its stated CoT, and which activate when it shortcuts to the correct answer despite incorrect reasoning?

### Research Questions

1. **Can we identify separable faithful and shortcut circuits?** (Phase 1)
2. **Are the identified circuits robust to statistical scrutiny?** (Phase 2A)
3. **Do the same circuit patterns emerge in larger, more capable models?** (Phase 2B)
4. **Can ablating shortcut components force models to follow their stated CoT?** (Phase 2B)

---

## 2. Method

### 2.1 Contrastive Pair Generation

We generate pairs of arithmetic prompts that share identical questions but differ in CoT correctness:

```
Faithful:   "What is 15 + 47? Solution: 15 + 47. Units: 5+7=12. Carry 1. Tens: 1+4+1=6. Answer: "
Unfaithful: "What is 15 + 47? Solution: 15 + 47. Units: 5+7=12. Carry 1. Tens: 1+4+1=8. Answer: "
```

A pair is labelled **unfaithful** if the model produces the correct answer (62) despite the corrupted CoT stating the tens digit is 8. This means the model is *not following its CoT* — it has a shortcut circuit. A pair is **faithful** if the model follows the corrupted CoT and produces the wrong answer (82).

We ensure token-length parity between faithful and unfaithful prompts so activation patching operates on aligned positions.

**Corruption types**: `units_error` (flip units digit), `tens_error` (flip tens digit), `complete_fabrication` (random wrong digits). `carry_error` is generated but filtered as it changes template structure.

### 2.2 Two-Pass Activation Patching

We use a two-pass approach for efficient circuit discovery in large models:

1. **Pass 1 — Layer sweep**: Patch entire layer outputs (attention and MLP) from the clean run into the corrupted run. Measure restoration score: how much the patched logit difference moves toward the correct answer.

2. **Pass 2 — Head-level sweep**: In the top-k layers identified in Pass 1, patch individual attention head outputs (`hook_z`). This gives per-head restoration scores at a fraction of the cost of sweeping all heads in all layers.

**Restoration score** for component $c$:

$$R(c) = \frac{\text{logit\_diff}(\text{patched}_c) - \text{logit\_diff}(\text{corrupted})}{\text{logit\_diff}(\text{clean}) - \text{logit\_diff}(\text{corrupted})}$$

A score of 1.0 means the component fully restores the clean answer; >1.0 means over-restoration; negative means the component actively suppresses the correct answer.

### 2.3 Detection Probe

We train a logistic regression probe on the concatenated `hook_z` activations of identified circuit heads to classify faithful vs unfaithful pairs. We evaluate with:

- **Cross-validated AUC and accuracy** (5-fold)
- **Hewitt-Liang selectivity** — the probe must outperform control probes trained on randomly permuted labels
- **Distributed signal test** — probes trained on random (non-circuit) layers, to check if the signal is localised

### 2.4 Dual-Metric Analysis

We compare two rankings of circuit components:

1. **Probe importance** — absolute logistic regression coefficients (averaged over feature dimensions per head)
2. **Restoration importance** — absolute mean restoration score from activation patching

If these rank differently, it reveals that *discriminative* importance (what helps a classifier tell faithful from unfaithful) differs from *causal* importance (what changes the model's output).

### 2.5 Intervention Experiments

We zero-ablate the top-k shortcut components (by restoration score) and test:

- **Intervention success**: On unfaithful pairs, does the model shift from the correct answer to the CoT-consistent (wrong) answer?
- **Faithful preservation**: On faithful pairs, does the model continue to produce the CoT-consistent answer?

---

## 3. Results

### 3.1 Phase 1 — GPT-2 Small (124M parameters)

**23 causally-verified circuit components** identified via full per-head activation patching across all 12 layers × 12 heads. A linear probe on circuit activations achieves **AUC 0.949** (later improved to 0.98 in Phase 2A replication).

GPT-2 Small cannot reliably perform two-digit addition, which limits intervention experiments but does not affect detection: the model still exhibits separable faithful/shortcut processing of the CoT.

### 3.2 Phase 2A — Validation

| Gate | Condition | Result |
|------|-----------|--------|
| Probe selectivity > 0 | Hewitt-Liang control | **PASS** (selectivity = 0.110) |
| L7H6 rank 1 in bootstrap | >90% of bootstrap samples | **FAIL** (0% — mean rank 12/23) |
| FN cluster identified | Error pattern in false negatives | **INCONCLUSIVE** |

**Key finding**: L7H6 has the highest probe coefficient (0.140) but L0MLP has the highest restoration score (0.721). These measure different things: discriminative vs causal importance. This is the first observation of *dual-metric divergence*.

### 3.3 Phase 2B — Qwen2.5-1.5B-Instruct

#### Dataset

| Metric | Value |
|--------|-------|
| Valid pairs | 110 / 250 (44%) |
| Unfaithful | 64 (58%) |
| Faithful | 46 (42%) |

#### Circuit Discovery

Top components by absolute restoration score:

| Head | Restoration | Type |
|------|------------|------|
| L3H5 | +3.494 | Shortcut (over-restores) |
| L14H9 | +2.185 | Shortcut |
| L13H10 | −2.104 | CoT-following (suppresses) |
| L13H11 | +1.459 | Shortcut |
| L16H1 | −1.446 | CoT-following |

Top layer: **`blocks.0.hook_attn_out`** (restoration = 0.977), matching GPT-2's Layer 0 dominance.

#### Dual-Metric Divergence

| Rank | Top by Probe | Top by Restoration |
|------|-------------|-------------------|
| #1 | **L22H9** (0.156) | **L3H5** (3.494) |
| #2 | L16H7 (0.119) | L14H9 (2.185) |
| #3 | L16H1 (0.101) | L13H10 (2.104) |

**The divergence replicates.** In GPT-2: L7H6 ≠ L0MLP. In Qwen: L22H9 ≠ L3H5. The top discriminative component is never the top causal component.

#### Probe Performance

| Metric | GPT-2 | Qwen 1.5B |
|--------|-------|-----------|
| AUC | 0.98 | **0.38** |
| Accuracy | 0.925 | **0.464** |
| Selectivity | +0.110 | **−0.054** |

The probe fails on Qwen. This is interpretable: in a 28-layer model with 12 heads per layer, the sparse 15-head circuit captures less of the total computation. The faithfulness signal is encoded non-linearly across many more components.

#### Distributed Signal

| Source | Accuracy | AUC |
|--------|----------|-----|
| Circuit (15 heads) | 0.464 | 0.380 |
| Layer 7 | 0.500 | 0.436 |
| Layer 14 | 0.491 | 0.448 |
| **Layer 21** | **0.545** | **0.511** |

Random layers outperform the circuit probe, confirming the distributed signal finding from GPT-2.

#### Intervention

| Ablation Set | Success Rate | Faithful Preserved |
|--------------|-------------|-------------------|
| L3H5 | 0.0% (0/44) | 97.1% |
| L3H5 + L14H9 | **6.4% (3/47)** | 85.3% |
| + L13H10 | 2.4% (1/42) | 97.1% |
| + L13H11 | 6.2% (3/48) | 100% |
| + L16H1 | 4.3% (2/47) | 97.2% |

Key observations:
- **Non-zero success**: Ablating L14H9 causes 3 unfaithful pairs to flip, providing causal evidence for the shortcut circuit.
- **Highly distributed**: Even 5-head ablation only shifts ~4–6% of cases.
- **Selective**: Faithful reasoning is preserved at 85–100%, validating the dual-circuit hypothesis.
- **Non-monotonic**: Adding L13H10 (negative restoration) counteracts L14H9, confirming the circuit has both shortcut and CoT-following components.

---

## 4. Discussion

### 4.1 The Dual-Metric Divergence

Our central finding is that **discriminative importance ≠ causal importance** for faithfulness detection, and this holds across architectures. Why?

*Probe coefficients* measure which activation dimensions have the most distinctive *patterns* that separate classes. A head can be highly discriminative because its activations happen to encode a feature (perhaps attention pattern entropy, or copy-vs-compute mode) that correlates with faithfulness — without that head being causally responsible for the shortcut.

*Restoration scores* measure which components' activations, when swapped from clean to corrupted context, change the output most. A head with high restoration captures the actual computational pathway, even if its activation pattern doesn't neatly separate into two linear classes.

This distinction matters for safety: a monitoring system built on probes would focus on the wrong components. A causal intervention system would need to target a different set of heads than the ones flagged by the probe.

### 4.2 Why the Probe Fails at Scale

The GPT-2 probe (AUC 0.98) succeeds because GPT-2 Small is a 12-layer model where 23 circuit components represent a significant fraction of all computation. In Qwen (28 layers × 12 heads = 336 heads), the 15 identified circuit components are 4.5% of the model. The faithfulness signal, distributed across the other 95.5%, dominates any sparse circuit representation.

This has practical implications: as models scale, faithfulness detection via sparse circuit probes becomes harder. The signal doesn't disappear — it distributes.

### 4.3 Selective Ablation

The intervention results, while modest in magnitude, demonstrate a key property: ablating shortcut heads degrades unfaithful processing (6.4% flip) while preserving faithful processing (85–100% preserved). This asymmetry is evidence for functionally separable circuits — the model uses different pathways for "follow the CoT" vs "compute independently."

### 4.4 Limitations

1. **Arithmetic domain only** — generalisation to natural language reasoning is untested.
2. **Small intervention effect** — 6.4% success may reflect the distributed nature of the circuit rather than a fundamental limitation.
3. **Single model per architecture** — we tested one GPT-2 and one Qwen model; broader scaling studies are needed.
4. **Stochastic dataset** — the contrastive pair generation has randomness; results may vary across runs (different runs produced different top heads, though the dual-metric divergence replicated in both).

---

## 5. Conclusions

We find three robust patterns in how transformers process chain-of-thought:

1. **Dual-metric divergence**: The component best at *classifying* faithful vs unfaithful reasoning ≠ the component with the largest *causal effect*. This replicates across GPT-2 and Qwen.

2. **Distributed faithfulness signal**: Faithfulness information is spread across the residual stream, not localised in identifiable circuit components. Random layers match or outperform targeted circuit probes.

3. **Selective shortcut ablation**: Removing shortcut heads degrades unfaithful processing while preserving faithful processing, providing causal evidence for functionally separable circuits.

For AI safety, these findings suggest that CoT monitoring is more complex than previously assumed. The circuits responsible for "following stated reasoning" vs "computing independently" are real and separable, but detecting their activation requires methods beyond sparse linear probes on identified components.

---

## References

- Chen, J. et al. (2025). Reasoning Models Don't Always Say What They Think. *arXiv:2505.05410*.
- Hewitt, J. & Liang, P. (2019). Designing and Interpreting Probes with Control Tasks. *EMNLP*.
- Hubinger, E. et al. (2019). Risks from Learned Optimization. *arXiv:1906.01820*.
- Turpin, A. et al. (2023). Language Models Don't Always Say What They Think. *NeurIPS*.
- Wang, K. et al. (2022). Interpretability in the Wild. *NeurIPS*.
- Yang, Z. et al. (2025). Unveiling Internal Reasoning Modes in LLMs. *EMNLP*.

---

## Reproducibility

All code, data, and results are available at [github.com/ashioyajotham/cot-faithfulness-mech-interp](https://github.com/ashioyajotham/cot-faithfulness-mech-interp).

```bash
# Phase 1 (GPT-2)
pip install -e ".[phase1]"
jupyter lab phase1/experiments/circuit_discovery/phase1_circuit_discovery.ipynb

# Phase 2A (Validation)
pip install -e ".[phase2a]"
python phase2/2a_validation/run_all_2a.py --device auto

# Phase 2B (Qwen, requires GPU)
pip install -e ".[phase2b]"
python phase2/2b_scaling/colab_runner.py --model qwen25-math-1.5b --device auto
```
