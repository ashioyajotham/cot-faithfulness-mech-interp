# When Models Lie to Please: Scaling Mechanistic Detection of Unfaithful Chain-of-Thought
## Research Proposal — Phase 2

**Researcher:** Victor Ashioya (Jotham)
**Affiliation:** Bluedot Impact Technical AI Safety Programme; MsingiAI
**Contact:** ashioyajotham.github.io · github.com/ashioyajotham/cot-faithfulness-mech-interp
**Date:** April 2026

---

## Abstract

Phase 1 of this project established a mechanistic pipeline for detecting unfaithful chain-of-thought (CoT) reasoning in GPT-2 Small, identifying 23 causally-verified circuit components, localising a dominant shortcut head (L7H6), and achieving 88.1% detection accuracy (ROC-AUC 0.949) via a linear probe. Reviewer feedback surfaced four substantive limitations that this proposal directly addresses: the probe's selectivity has not been tested against the Hewitt-Liang standard; the high-confidence false negatives (11 of 14 with >99% confidence) are unexplained; L7H6's dominance lacks statistical rigour; and the GPT-2 arithmetic failure means the faithfulness labels are researcher-constructed rather than grounded in demonstrated model behaviour. Phase 2 resolves each of these in sequence before scaling the validated methodology to arithmetic-capable models (Qwen2.5-Math-7B, Gemma 3 12B IT). The central goal is to produce the first causally-grounded, statistically-validated faithfulness detector that generalises beyond a single model family, and to use that detector to test whether surgical ablation of shortcut circuits changes model behaviour in a measurable direction.

---

## 1. Background and Motivation

### 1.1 The Safety Case

Chain-of-thought monitoring is a proposed mechanism for catching misaligned model behaviour during both training and deployment. The argument runs: if a model's stated reasoning is legible and faithful, then monitoring the CoT is equivalent to monitoring the model's decision process. Anthropic's Alignment Science team found, however, that reasoning models reveal their use of hints in fewer than 20% of cases where they actually used those hints, and that outcome-based reinforcement learning improves faithfulness initially but plateaus without saturating. Their study also found that unfaithful chains-of-thought were on average substantially longer than faithful ones, ruling out brevity as the explanation for omission. This means CoT monitoring, as currently practised, provides weaker safety guarantees than assumed.

The deeper problem is structural. If models have separable circuits for "produce a CoT that looks like reasoning" and "compute the actual answer," then the two pathways can diverge without leaving any surface-level trace. Phase 1 of this project provided direct mechanistic evidence that such separation exists in GPT-2 Small. The present proposal tests whether the same structure appears in models that are capable enough to make the safety stakes real.

### 1.2 Related Work

**Faithfulness measurement.** Turpin et al. (2023) demonstrated 36% accuracy drops from biased features that models never acknowledged in their CoT. Lanham et al. (2023) established that larger models are less faithful by early-exit truncation experiments. Chen et al. (2025) extended this to state-of-the-art reasoning models, finding that CoT reveal rates for hint usage often fall below 20% even in Claude 3.7 Sonnet and DeepSeek R1.

**Shortcut circuits.** Yang et al. (EMNLP 2025) investigated two reasoning modes in LLMs — latent reasoning and factual shortcuts — using attention knockout and logit lens techniques, finding that shortcut reasoning bypasses intermediate bridge-entity representation and resembles single-hop factual queries, while their proposed Attribute Rate Ratio metric achieves ~90% accuracy on distinguishing the two modes. This is a direct parallel to the L7H6 shortcut head finding; their work provides both methodological precedent and a comparison baseline.

**Probe validity.** Hewitt and Liang (2019) proposed control tasks — associating word types with random outputs — as a complement to linguistic probing tasks, arguing that a good probe should be selective: achieving high accuracy on the linguistic task and low accuracy on the control task. They found that popular probes on ELMo representations are not selective. This critique applies directly to Phase 1's linear probe, which has not been tested for selectivity.

Belinkov (2022) extended this critique by reviewing the promises and shortcomings of probing classifiers more broadly, arguing that high accuracy alone is insufficient to claim that a probe reveals model-internal structure rather than learning surface patterns from the dataset.

**Deceptive alignment.** Hubinger et al. (2019) formalised the deceptive alignment scenario: a model that appears aligned during training and evaluation but pursues misaligned goals during deployment. Unfaithful CoT is a precursor mechanism — a model producing explanations that do not reflect its internal computation has already separated its "presented" reasoning from its actual decision pathway. The present work can be framed as building an early-warning detector for the first stage of deceptive alignment.

### 1.3 What Phase 1 Established

- A contrastive pair dataset (600 examples, arithmetic domain) with ground-truth faithfulness labels at the activation level — the first such resource designed explicitly for activation-level analysis
- 23 causally-verified circuit components identified via activation patching (hook_z, per-head granularity)
- L7H6 identified as the dominant shortcut head (restoration score −0.329; probe coefficient ~20% higher than next component)
- Linear probe: 88.1% accuracy, ROC-AUC 0.949
- Confirmed that GPT-2 cannot do arithmetic, invalidating the intervention experiments on that model while leaving the detection results intact

### 1.4 What Phase 1 Did Not Establish (and Must)

Four specific gaps, surfaced in reviewer feedback, frame the Phase 2 agenda:

1. **Probe selectivity:** Is the probe detecting faithfulness-related structure in the representations, or learning surface features of how the prompts were constructed? The 30% error rate on high-magnitude sums (>80) suggests some distributional sensitivity.

2. **High-confidence failures:** 11 of 14 false negatives carry >99% confidence. The probe is not just wrong on hard cases — it is confidently wrong. What distinguishes these examples? If they share a structural pattern, that pattern may reveal either the probe's limitations or a more fundamental property of how unfaithful reasoning appears at the circuit level.

3. **Statistical validity of L7H6:** A coefficient 20% higher than the next component is not obviously distinctive without confidence intervals or significance tests. The claim that L7H6 is the dominant shortcut head requires bootstrap resampling or cross-validation across held-out pair types.

4. **Grounded labels on a capable model:** The Phase 1 labels are researcher-constructed: I deliberately corrupted CoT steps and observed whether the model's output changed. On a model that cannot do arithmetic, the model's output changes are uninterpretable. On a model that can, the label "unfaithful" acquires a proper referent: the model was given wrong steps and produced the correct answer anyway, which is behaviorally grounded evidence of shortcut usage.

---

## 2. Research Questions

**RQ1 (Probe validity):** Does the linear probe trained on Phase 1.5 circuit activations satisfy the Hewitt-Liang selectivity criterion? Is its accuracy attributable to representation structure or to memorisable surface features of the prompt?

**RQ2 (Failure mode analysis):** What structural or computational properties distinguish the 11 high-confidence false negatives? Do they cluster by carry magnitude, operand size, or CoT corruption severity?

**RQ3 (Statistical robustness):** Are L7H6's dominance and the restoration score rankings statistically stable under bootstrap resampling and cross-pair-type evaluation?

**RQ4 (Scaling):** Does the same circuit structure — early-layer faithful heads, mid-to-late shortcut heads — emerge in Qwen2.5-Math-7B and Gemma 3 12B IT? Do equivalent shortcut heads appear at comparable positions?

**RQ5 (Intervention):** On a model that can do arithmetic, does ablating the identified shortcut heads shift behaviour from "correct answer despite wrong CoT" toward "answer consistent with wrong CoT"? What is the success rate?

**RQ6 (Probe reformatting):** Can completion-style prompting elicit arithmetic behaviour from GPT-2, enabling a tractable intervention test on the original model?

---

## 3. Phase 2 Research Plan

### 3.1 Experiment 1 — Probe Selectivity and Control Tasks
**Addresses:** RQ1, Hewitt & Liang critique, Belinkov critique

**Method:**
Implement Hewitt and Liang's selectivity test. For each example in the dataset, construct a control task by replacing the faithfulness label (0/1) with a random label sampled uniformly, stratified by prompt number rather than by actual faithfulness. Train a probe on the control labels using the same circuit activations. Selectivity = (linguistic task accuracy) − (control task accuracy). A selective probe has high linguistic accuracy and low control accuracy.

Additionally, train a minimum-description-length (MDL) probe (Voita & Titov 2020) to measure how many bits of information about faithfulness are encoded in the representations, providing a compression-based complement to selectivity.

Run two ablations:
- Scramble activations: randomly permute the circuit activation vectors across examples. If the probe degrades, it is using structure, not surface features.
- Random-layer baseline: replace circuit activations with activations from a randomly-selected non-circuit layer. If the probe performs comparably, the circuit identification added no value over arbitrary layer selection.

**Expected result:** Selectivity > 0.15 would indicate the probe is using genuine representation structure. A selectivity near zero would require reframing the Phase 1 detection result as a surface-pattern classifier.

**Deliverable:** Selectivity score, MDL compression rate, scramble and random-layer baselines. If selectivity is low, the revised claim becomes: "prompt-construction patterns correlate with activation patterns at the identified circuit components," which is still informative but weaker.

---

### 3.2 Experiment 2 — High-Confidence False Negative Analysis
**Addresses:** RQ2, reviewer concern about the 11/14 confident errors

**Method:**
Extract all 14 false negatives from Phase 1. For each:
- Record: operand values (a, b), carry requirement (0/1), sum magnitude, CoT corruption type and magnitude, and which specific intermediate steps were corrupted
- Run the patching experiments on these 14 examples individually to obtain per-example restoration scores
- Inspect the attention patterns of L7H6 on these examples versus true positives

Cluster the false negatives by the above features. Test whether carry-requiring sums (units digit sum > 9) are disproportionately represented. The hypothesis: carry operations require a qualitatively different computational pathway, and the probe's features do not capture the additional circuit components involved.

If the false negatives cluster by carry requirement, this points to an underspecified feature set: the circuit map needs carry-specific components added. If they cluster by sum magnitude (>80), this suggests a distribution shift: the training examples cluster at lower sums, and the probe generalises poorly at the edges.

**Deliverable:** A taxonomy of false negative types, a revised circuit map if carry-specific components are identified, and a concrete prescription for augmenting the dataset.

---

### 3.3 Experiment 3 — Statistical Validation of L7H6
**Addresses:** RQ3

**Method:**
Bootstrap resampling (1000 iterations) over the contrastive pairs to generate confidence intervals for:
- Each head's average restoration score
- The rank ordering of the top 10 heads by restoration score
- The linear probe's coefficient magnitude for L7H6 relative to other components

Additionally, evaluate stability across pair types: run the full patching analysis separately on (a) faithfulness pairs, (b) shortcut detection pairs, and (c) positional bias pairs. If L7H6 consistently ranks as the dominant shortcut head across all three pair types, that is evidence for generality. If it ranks high only on one pair type, the claim must be qualified accordingly.

Finally, run ablation cascade experiments: ablate L7H6 alone, then L7H6 + L5H9, then L7H6 + L5H9 + L6H8. If the probe's accuracy drops monotonically as more shortcut heads are added back (after ablation), this constitutes additive evidence for the shortcut circuit structure.

**Deliverable:** Confidence intervals on all restoration scores, rank stability analysis, cascade ablation curve, revised claim language calibrated to the statistical evidence.

---

### 3.4 Experiment 4 — GPT-2 Completion-Style Prompting
**Addresses:** RQ6, reviewer suggestion

**Method:**
Reviewer feedback noted that GPT-2's 0% arithmetic accuracy may be partly a prompting artefact. GPT-2 was trained as a completion model, not an instruction-following model. The Phase 1 prompt format used question-answer style ("Q: What is 23+45? A:"), which may not align with GPT-2's training distribution.

Test two completion-style reformats:
- Format A: "23 + 45 = " (bare equation completion)
- Format B: "The sum of 23 and 45 is " (natural language completion)
- Format C: Few-shot arithmetic demonstrations followed by the target problem

Measure: p(correct token | prompt) for each format on 100 arithmetic problems. If any format achieves baseline accuracy > 5%, run the full intervention experiments (ablate L7H6, measure shift in behaviour on faithful/unfaithful pairs) on that format.

This is a tractable experiment with a concrete binary outcome: either GPT-2 can produce arithmetic answers in some prompt regime (enabling the intervention test) or it cannot (confirming the architectural limitation hypothesis). Either result is publishable.

**Deliverable:** Accuracy vs. prompt format comparison, and either the intervention experiment results or a definitive characterisation of why intervention is intractable on GPT-2.

---

### 3.5 Experiment 5 — Scaling to Qwen2.5-Math-7B
**Addresses:** RQ4, RQ5 — the primary Phase 2 contribution

**Model choice rationale:**
Qwen2.5-Math-7B achieves 85% on the MATH benchmark and is instruction-tuned for arithmetic. It is HookedTransformer-compatible via TransformerLens. Its 32-layer, 28-head architecture (896 heads total vs. 144 in GPT-2) requires compute-efficient patching strategies.

**Dataset construction:**
Extend the Phase 1 contrastive pair design to include:
- Two-digit addition (replication of Phase 1 on a capable model)
- Three-digit addition (harder carry requirements)
- Subtraction (different computational pathway)
- Mixed arithmetic word problems (tests domain generality)

Target: 2,000 contrastive pairs balanced across types and difficulty levels.

**Circuit discovery (Phase 1 replication):**
Run the full Phase 1.5 pipeline on Qwen2.5-Math-7B:
1. Zero ablation on all heads and MLPs to identify necessary components
2. Contrastive patching to separate faithful from shortcut components
3. Head-level analysis via hook_z to identify per-head restoration scores

Compute efficiency: with 896 heads, running exhaustive patching is expensive. Use a two-pass approach: first run layer-level patching (hook_attn_out) to identify the top 5 most important layers; then run per-head analysis only within those layers. This reduces compute from 896 to approximately 168 runs per example.

**Detection:**
Train the same three-path detector (linear probe, steering vector, hybrid) on Qwen activations. Compare:
- Within-task accuracy (train on 2-digit, test on 2-digit)
- Cross-task accuracy (train on 2-digit, test on 3-digit and subtraction)

The cross-task test is the critical generalisation check missing from Phase 1.

**Intervention:**
Ablate the identified shortcut heads (Qwen-equivalent of L7H6). Measure:
- On unfaithful examples (wrong CoT): does the model shift from "correct answer" to "answer consistent with wrong CoT"?
- On faithful examples (correct CoT): does ablation degrade accuracy (does it harm the faithful pathway)?

Target metric: intervention success rate = fraction of unfaithful examples that shift to CoT-consistent output after shortcut head ablation. Phase 1 hypothesis: >30% success rate would constitute meaningful evidence of causal shortcut routing.

**Deliverable:** Full replication of Phase 1.5 methodology on Qwen2.5-Math-7B, cross-task detection results, intervention success rate.

---

### 3.6 Experiment 6 — Cross-Model Comparison (Gemma 3 12B IT)
**Addresses:** RQ4 — generalisation across architectures

Running the same pipeline on Gemma 3 12B IT (Google's instruction-tuned version) serves two purposes: it tests whether the shortcut circuit structure is architecture-specific or general, and it provides a second data point for the L7H6-equivalent head search.

Gemma 3 12B uses 48 layers and 8 attention heads per layer (384 heads total vs. Qwen's 896). The lower head count makes exhaustive patching more tractable. Gemma Scope 2 SAEs for Gemma 3 are publicly available, enabling a methodological complement: use SAE features to interpret what L7H6-equivalent heads in Gemma are attending to, not just whether they matter causally.

**Deliverable:** Cross-architecture comparison of shortcut head positions, SAE-grounded interpretation of at least one shortcut head, preliminary evidence for or against universality of the shortcut circuit structure.

---

## 4. Addressing the Deceptive Alignment Connection

Reviewer feedback noted that the project mentions deceptive alignment (Hubinger et al. 2019) briefly but does not investigate it. Phase 2 makes this connection explicit and testable.

The deceptive alignment hypothesis predicts: a model trained with outcome-based RL on tasks where CoT is unnecessary will develop circuits that route answers through shortcuts while producing plausible-looking CoT text. This is exactly what Anthropic observed: when RL increases how frequently hints are used (reward hacking), the propensity to verbalise those hints in the CoT does not increase, even without training against a CoT monitor.

The Phase 2 detector can be applied to this setting. Specifically:
- Fine-tune Qwen2.5-Math on arithmetic tasks with outcome-based RL (using the GSM8K training split)
- At checkpoints throughout training, run the full Phase 1.5 circuit discovery pipeline
- Track whether shortcut head restoration scores increase as RL training progresses (i.e., whether RL strengthens the shortcut circuits)
- Compare shortcut head strength in the RL-trained model vs. the SFT baseline

This is a direct test of the deceptive alignment precursor hypothesis: does RL training, optimising for correct answers without CoT supervision, produce models with stronger shortcut circuits?

This experiment is ambitious and may fall outside the immediate Phase 2 scope depending on compute availability. It is included here as the natural third phase of the project and the target for the Schmidt Sciences RFP submission.

---

## 5. Literature to Engage

The following papers, cited by reviewers or identified in the literature search, must be engaged explicitly:

| Paper | Relevance | How engaged |
|-------|-----------|-------------|
| Hewitt & Liang, EMNLP 2019 | Probe selectivity standard | Experiment 1 directly implements their control task method |
| Belinkov, *Computational Linguistics* 2022 | Probe interpretation critique | Selectivity + MDL tests address both critiques |
| Chen et al. (Anthropic), arXiv 2505.05410 | Strongest recent evidence for CoT unfaithfulness in capable models | Motivates Experiment 5; Anthropic's behavioural results + our mechanistic results are complementary |
| Yang et al., EMNLP 2025 | Latent reasoning vs. factual shortcuts via attention knockout | Methodological parallel; ARR metric as comparison baseline |
| Hubinger et al. 2019 | Deceptive alignment framing | RL checkpoint experiment (Section 4) |
| Cox (LessWrong, 2025) | Linear probes can decode answers before CoT generation | Our detection probe complements this: they probe before generation, we probe at answer time |
| Turpin et al. 2023 | Foundational faithfulness measurement | Already cited; re-engage with their biasing methodology for Qwen experiments |
| Hanna et al., NeurIPS 2023 | GPT-2 arithmetic limitations | Already cited; use their greater-than circuit as comparison for Experiment 3 |

---

## 6. Project Structure and Timeline

### Phase 2A — Validation and Repair (Months 1–2)

| Experiment | Description | Deliverable |
|------------|-------------|-------------|
| 1 | Probe selectivity (Hewitt-Liang control tasks) | Selectivity score, MDL compression, random baselines |
| 2 | High-confidence false negative analysis | Failure taxonomy, revised circuit map |
| 3 | Bootstrap CI for L7H6 and restoration scores | Confidence intervals, rank stability report |
| 4 | GPT-2 completion-style prompting | Accuracy vs. format; intervention results if feasible |

**Gate:** Phase 2B proceeds unconditionally (probe selectivity results inform but do not block scaling). If the probe is not selective, the contribution is reframed as a circuit-based surface-feature detector — still informative, but with revised claims.

### Phase 2B — Scaling (Months 3–5)

| Experiment | Description | Deliverable |
|------------|-------------|-------------|
| 5 | Qwen2.5-Math-7B full pipeline | Circuit map, detection accuracy (within- and cross-task), intervention results |
| 6 | Gemma 3 12B IT comparison | Cross-architecture shortcut head comparison, SAE interpretation |

### Phase 2C — Write-up and Dissemination (Month 6)

- Workshop paper targeting: MechInterp @ NeurIPS 2026, or ICLR 2027 (full paper track)
- Dataset release: expanded contrastive pair dataset on Hugging Face
- Code release: TransformerLens pipeline compatible with GPT-2, Qwen2.5-Math, and Gemma 3
- Blog post on *Unsupervised Insights* Substack covering the selectivity results and scaling findings

---

## 7. Compute Requirements

| Task | Model | Est. GPU-hours |
|------|-------|----------------|
| Experiment 1–3 (GPT-2) | GPT-2 Small | ~8 hrs (Modal A100) |
| Experiment 4 (GPT-2 prompting variants) | GPT-2 Small | ~4 hrs |
| Experiment 5 (Qwen2.5-Math circuit discovery, 2-pass) | Qwen2.5-Math-7B | ~40 hrs |
| Experiment 5 (Qwen detection + intervention) | Qwen2.5-Math-7B | ~20 hrs |
| Experiment 6 (Gemma 3 12B) | Gemma 3 12B IT | ~30 hrs |
| **Total** | | **~102 hrs** |

Infrastructure: Modal (existing account, Phase 1 pipeline already deployed). Weights & Biases for experiment tracking. Estimated cost at Modal spot pricing: ~$200–250.

---

## 8. Success Criteria

The project succeeds if any of the following are achieved:

**Minimum:** Probe selectivity > 0 (Phase 1 detection result holds as structure-based, not surface-based); bootstrap CI confirms L7H6 dominance; failure analysis produces a concrete prescription for dataset improvement.

**Target:** Qwen2.5-Math-7B replication with cross-task accuracy > 75%; at least one intervention success rate > 20%.

**Stretch:** Cross-architecture shortcut head identified in Gemma 3 with SAE-level interpretation; RL training checkpoint experiment showing shortcut head strengthening.

---

## 9. Connection to the Schmidt Sciences RFP

This proposal constitutes Phase 2 of the three-phase scope submitted to the Schmidt Sciences Interpretability RFP (INT-RFP-26-0000000071). The full proposal structure is:

- **Phase 1 (complete):** GPT-2 Small baseline — circuit discovery, detection probe, dataset release
- **Phase 2 (this document):** Validation + scaling to arithmetic-capable models; intervention experiments
- **Phase 3 (proposed):** Application to human-AI collaborative reasoning; using circuit-level faithfulness scores as trust calibration signals in human-facing systems

The present document constitutes the technical specification for Phase 2.

---

## 10. Acknowledgements

Phase 1 was completed as part of the Bluedot Impact Technical AI Safety Programme. This proposal responds directly to reviewer feedback from the Bluedot assessment panel. The Anthropic Alignment Science team's concurrent work on CoT faithfulness in reasoning models (Chen et al. 2025) provides both motivation and a complementary behavioural baseline against which the mechanistic results of this project can be situated.

---

## References

Belinkov, Y. (2022). Probing classifiers: Promises, shortcomings, and advances. *Computational Linguistics*, 48(1), 207–219.

Chen, Y., Benton, J., Radhakrishnan, A., Uesato, J., Denison, C., Schulman, J., … Perez, E. (2025). Reasoning models don't always say what they think. arXiv:2505.05410.

Hanna, M., Liu, O., & Variengien, A. (2023). How does GPT-2 compute greater-than? Interpreting mathematical abilities in a pre-trained language model. *NeurIPS 2023*.

Hewitt, J., & Liang, P. (2019). Designing and interpreting probes with control tasks. *EMNLP-IJCNLP 2019*, 2733–2743.

Hubinger, E., van Merwijk, C., Mikulik, V., Skalse, J., & Garrabrant, S. (2019). Risks from learned optimization in advanced machine learning systems. arXiv:1906.01820.

Lanham, T., Chen, A., Radhakrishnan, A., Steiner, B., Denison, C., Hernandez, D., … Perez, E. (2023). Measuring faithfulness in chain-of-thought reasoning. arXiv:2307.13702.

Nanda, N., Chan, L., Lieberum, T., Smith, J., & Steinhardt, J. (2023). Progress measures for grokking via mechanistic interpretability. *ICLR 2023*.

Turpin, M., Michael, J., Perez, E., & Bowman, S. (2023). Language models don't always say what they think: Unfaithful explanations in chain-of-thought prompting. *NeurIPS 2023*.

Voita, E., & Titov, I. (2020). Information-theoretic probing with minimum description length. *EMNLP 2020*.

Wang, K., Variengien, A., Conmy, A., Shlegeris, B., & Steinhardt, J. (2022). Interpretability in the wild: a circuit for indirect object identification in GPT-2 small. *ICLR 2023*.

Yang, Y., Sun, H., Wang, J., Qi, Q., Zhuang, Z., Wang, H., Ren, P., Wang, J., & Liao, J. (2025). Unveiling internal reasoning modes in LLMs: A deep dive into latent reasoning vs. factual shortcuts with attribute rate ratio. *EMNLP 2025*, 2186–2206.
