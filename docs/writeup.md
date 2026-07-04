# When Models Lie to Please: Mechanistic Detection of Unfaithful Chain-of-Thought

**Victor Ashioya (Jotham)**  
Bluedot Impact Technical AI Safety Programme · MsingiAI

---

## Abstract

We investigate whether chain-of-thought (CoT) reasoning in language models faithfully reflects their internal computation. Using mechanistic interpretability — activation patching, linear probes, and zero-ablation interventions — we identify separable *faithful* and *shortcut* circuits in transformers processing arithmetic CoT. Across three model scales (GPT-2 Small, 124M; Qwen2.5-1.5B-Instruct; Qwen2.5-7B-Instruct), we discover a *dual-metric divergence*: the component most useful for *classifying* faithful vs unfaithful reasoning (probe coefficient) is consistently different from the component with the largest *causal effect* on model output (restoration score). This finding replicates across all three scales. We also identify six scaling laws: (1) Layer 0 attention is the top layer-level component universally; (2) linear probe AUC degrades monotonically with scale (0.98 → 0.617 → 0.536); (3) larger models are more unfaithful (69% at 7B vs 60% at 1.5B); (4) shortcut circuits consolidate into dedicated layers at scale; (5) faithful preservation under ablation improves with scale (97.8–100% at 7B); and (6) non-linear probes on the full residual stream are the only detection method that improves at scale.

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

### 3.3 Phase 2B — Qwen2.5-1.5B-Instruct (500 pairs)

#### Dataset

| Metric | Value |
|--------|-------|
| Valid pairs | 205 / 500 (41%) |
| Unfaithful | 122 (60%) |
| Faithful | 83 (40%) |

#### Circuit Discovery

Two-pass patching on 200 valid pairs. Top components by absolute restoration score:

| Head | Restoration | Type |
|------|------------|------|
| L20H5 | +1.566 | Shortcut |
| L19H4 | +1.050 | Shortcut |
| L19H11 | +0.972 | Shortcut |
| L3H5 | −0.972 | CoT-following |
| L26H9 | −0.953 | CoT-following |

Top layer: **`blocks.0.hook_attn_out`** (restoration = 1.023), matching GPT-2's Layer 0 dominance.

#### Dual-Metric Divergence

| Rank | Top by Probe | Top by Restoration |
|------|-------------|-------------------|
| #1 | **L20H9** (0.160) | **L20H5** (1.566) |
| #2 | L20H5 (0.153) | L19H4 (1.050) |
| #3 | L22H11 (0.153) | L19H11 (0.972) |

**The divergence replicates.** In GPT-2: L7H6 ≠ L0MLP. In Qwen 1.5B: L20H9 ≠ L20H5.

#### Probe Performance

| Metric | GPT-2 | Qwen 1.5B |
|--------|-------|-----------|
| Circuit linear AUC | 0.98 | 0.617 |
| Circuit MLP AUC | — | 0.532 |
| Full-stream linear AUC | — | **0.660** |
| Full-stream MLP AUC | — | 0.561 |
| Selectivity | +0.110 | +0.069 |

The circuit linear probe is weaker than GPT-2 but passes selectivity. The full-stream linear probe (using residual stream from layers 0, 9, 18, 27) achieves the best AUC (0.660), confirming that the signal exists but is distributed beyond the circuit.

#### Intervention

| Ablation Set | Success Rate | Faithful Preserved |
|--------------|-------------|-------------------|
| L20H5 | 1.1% (1/88) | 98.5% |
| +L19H4 | **2.6% (2/78)** | 95.3% |
| +L19H11 | 1.1% (1/87) | 98.4% |
| +L3H5 | 0.0% (0/78) | 97.0% |
| +L26H9 | 1.1% (1/89) | 94.9% |

### 3.4 Phase 2B — Qwen2.5-7B-Instruct (500 pairs)

#### Dataset

| Metric | Value |
|--------|-------|
| Valid pairs | 309 / 500 (62%) |
| Unfaithful | 214 (69%) |
| Faithful | 95 (31%) |

The 7B model has a substantially higher valid rate (62% vs 41%) and higher unfaithful ratio (69% vs 60%), indicating stronger shortcut circuits.

#### Circuit Discovery

Two-pass patching on 100 valid pairs (28 layers × 28 heads = 784 total heads). A striking result: **all 15 top heads are in Layer 23**.

| Head | Restoration |
|------|------------|
| L23H7 | +1.900 |
| L23H15 | +1.700 |
| L23H16 | +1.700 |
| L23H9 | +1.600 |
| L23H25 | +1.450 |

Top layer: **`blocks.0.hook_attn_out`** (restoration = 0.944), continuing the universal Layer 0 pattern.

The Layer 23 concentration suggests that the 7B model has enough capacity to dedicate an entire layer to shortcut computation — a qualitatively different circuit topology from the 1.5B model, where top heads span layers 0–26.

#### Dual-Metric Divergence

| Rank | Top by Probe | Top by Restoration |
|------|-------------|-------------------|
| #1 | **L23H12** (0.186) | **L23H7** (1.900) |
| #2 | L23H15 (0.178) | L23H15 (1.700) |
| #3 | L23H25 (0.172) | L23H16 (1.700) |

**The divergence replicates at 7B.** L23H12 is the top discriminator; L23H7 is the top causal component. Three for three across model scales.

#### Probe Performance

| Probe Type | 1.5B AUC | 7B AUC |
|-----------|----------|--------|
| Circuit linear | 0.617 | 0.536 |
| Circuit MLP | 0.532 | 0.527 |
| Full-stream linear | **0.660** | 0.579 |
| Full-stream MLP | 0.561 | **0.616** |

Full-stream MLP is the *only* probe that improves from 1.5B → 7B. The faithfulness signal at 7B is non-linearly encoded in the residual stream — circuit-sparse linear probes are insufficient.

#### Intervention

| Ablation Set | Success Rate | Faithful Preserved |
|--------------|-------------|-------------------|
| L23H7 | 0.6% (1/162) | 100% |
| +L23H15 | 1.8% (3/166) | 100% |
| +L23H16 | 3.1% (5/163) | 100% |
| +L23H9 | 2.9% (5/171) | 98.8% |
| +L23H25 | **3.4% (6/174)** | 97.8% |

Intervention is monotonically increasing (unlike 1.5B's non-monotonic pattern), because all top 7B heads are shortcut heads with positive restoration — no conflicting CoT-following heads in the top 5. Faithful preservation is 97.8–100%, cleaner than the 1.5B model.

### 3.5 Cross-Scale Comparison

| Finding | GPT-2 (124M) | Qwen 1.5B | Qwen 7B |
|---------|:------------:|:---------:|:-------:|
| Layer 0 attn restoration | 0.721 (#1) | 1.023 (#1) | 0.944 (#1) |
| Dual-metric diverges | L7H6 ≠ L0MLP | L20H9 ≠ L20H5 | L23H12 ≠ L23H7 |
| Circuit probe AUC | 0.98 | 0.617 | 0.536 |
| Best probe AUC | 0.98 (circuit linear) | 0.660 (full-stream linear) | 0.616 (full-stream MLP) |
| Selectivity | 0.110 | 0.069 | 0.037 |
| Unfaithful ratio | ~50% | 60% | 69% |
| Best intervention | N/A | 2.6% | 3.4% |
| Faithful preservation | N/A | 95.3% | 97.8–100% |
| Signal distribution | DISTRIBUTED | DISTRIBUTED | DISTRIBUTED |
| Circuit head layers | Span L0–L9 | Span L0–L26 | **All in L23** |

---

## 4. Discussion

### 4.1 The Dual-Metric Divergence

Our central finding is that **discriminative importance ≠ causal importance** for faithfulness detection, and this holds across all three model scales. Why?

*Probe coefficients* measure which activation dimensions have the most distinctive *patterns* that separate classes. A head can be highly discriminative because its activations happen to encode a feature (perhaps attention pattern entropy, or copy-vs-compute mode) that correlates with faithfulness — without that head being causally responsible for the shortcut.

*Restoration scores* measure which components' activations, when swapped from clean to corrupted context, change the output most. A head with high restoration captures the actual computational pathway, even if its activation pattern doesn't neatly separate into two linear classes.

This distinction matters for safety: a monitoring system built on probes would focus on the wrong components. A causal intervention system would need to target a different set of heads than the ones flagged by the probe.

### 4.2 Probe Degradation as a Scaling Law

Circuit linear probe AUC follows a clear monotonic decline: 0.98 (GPT-2, 124M) → 0.617 (Qwen 1.5B) → 0.536 (Qwen 7B). The explanation is geometric: at GPT-2 scale, 23 circuit heads represent 16% of all heads (23/144); at 1.5B, 15 heads represent 4.5% (15/336); at 7B, 15 heads represent 1.9% (15/784). As the circuit becomes a smaller fraction of the model, it captures less of the total faithfulness signal.

Critically, the signal does not disappear — it distributes. Full-stream probes (using all residual stream activations from 4 strategically spaced layers) achieve better AUC than circuit probes at both 1.5B (0.660 vs 0.617) and 7B (0.616 vs 0.536). The best probe type also shifts with scale: linear dominates at 1.5B, but MLP dominates at 7B (0.616 vs 0.579), indicating that the faithfulness encoding becomes more non-linear as models grow.

### 4.3 Layer 23 Concentration in the 7B Model

The most striking architectural finding is that **all 15 top circuit heads in the 7B model are in Layer 23** (of 28). In contrast, 1.5B top heads span layers 0–26. This suggests that larger models have enough capacity to dedicate an entire layer to shortcut computation — a qualitatively different topology.

However, concentration does not mean vulnerability. Ablating 5 of 28 heads in Layer 23 only shifts 3.4% of unfaithful pairs, because the remaining 23 heads in Layer 23 compensate. The shortcut is distributed *within* its dedicated layer.

### 4.4 Selective Ablation Improves with Scale

Faithful preservation under ablation improves monotonically: 85–100% at 1.5B → 97.8–100% at 7B. This means the dual-circuit separation becomes *cleaner* as models scale. The shortcut circuit and the CoT-following circuit are more functionally independent in the 7B model, making targeted intervention safer — less collateral damage to correct reasoning.

### 4.5 Larger Models Are More Unfaithful

The 7B model produces unfaithful outputs on 69% of valid corrupted CoT pairs, compared to 60% at 1.5B and ~50% at GPT-2. More capable models are better at ignoring bad CoT to compute the correct answer. This creates a paradox for AI safety: as models become more capable and more widely deployed, they become *more* likely to shortcut past their stated reasoning, precisely when CoT monitoring becomes *less* reliable (per the probe degradation finding).

### 4.6 Limitations

1. **Arithmetic domain only** — generalisation to natural language reasoning is untested.
2. **Small intervention effect** — 3.4% success reflects the distributed nature of the circuit.
3. **Same architecture family for scaling** — both 1.5B and 7B are Qwen2.5-Instruct; cross-family scaling (e.g., Gemma, Llama) would strengthen the claims.
4. **Stochastic dataset** — different runs produce different top heads, though the dual-metric divergence replicated across all runs.
5. **Two-pass patching approximation** — we sweep top-k layers then per-head within those layers, potentially missing important heads in non-top layers.

---

## 5. Conclusions

We find six robust patterns in how transformers process chain-of-thought, tested across three model scales:

1. **Dual-metric divergence**: The component best at *classifying* faithful vs unfaithful reasoning ≠ the component with the largest *causal effect*. This replicates across GPT-2 (124M), Qwen 1.5B, and Qwen 7B.

2. **Layer 0 attention is a universal shortcut bottleneck**: The top layer-level component in all three models, suggesting shortcut circuits begin by capturing input structure at the earliest attention layer.

3. **Linear probe AUC degrades monotonically with scale**: 0.98 → 0.617 → 0.536. Sparse circuit probes become less reliable as models grow.

4. **Shortcut circuits consolidate at scale**: The 7B model concentrates all top shortcut heads in a single layer (L23), unlike the distributed topology at 1.5B.

5. **Larger models are more unfaithful**: Unfaithful ratio increases from ~50% to 60% to 69% across scales.

6. **Non-linear full-stream probes are the most scalable detection method**: The only probe type whose AUC improves from 1.5B to 7B.

For AI safety, these findings reveal a fundamental tension: as models scale, they become better at shortcutting past stated reasoning (finding 5) while the standard detection methods — sparse circuit probes — become less reliable (finding 3). The path forward requires either full-stream non-linear probes (finding 6) or entirely new detection paradigms that operate on the distributed faithfulness signal rather than on identified circuit components.

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

# Phase 2B — Qwen 1.5B (Colab T4)
pip install -e ".[phase2b]"
python phase2/2b_scaling/colab_runner.py --model qwen25-math-1.5b --device auto --n-pairs 500

# Phase 2B — Qwen 7B (Colab A100)
python phase2/2b_scaling/colab_runner.py --model qwen25-math-7b --device auto --n-pairs 500
```
