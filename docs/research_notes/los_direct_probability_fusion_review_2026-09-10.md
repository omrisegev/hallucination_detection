# LOS-style direct probability fusion for reasoning localization

**Date:** 2026-09-10  
**Scope:** A focused literature and code review for using the sorted next-token distribution directly inside the project's answer-local fusion method.

## Decision

The proposed experiment is justified and is materially different from the token-feature fusion already tested in Step 334.

The primary input must be the sorted probability/log-probability vector itself. It should not first be reduced to entropy components or other uncertainty summaries. Such a reduction would discard the distribution shape that LOS-Net was designed to retain.

The frozen candidate should use **K=10**. We should not run a K sweep in the first experiment. The LOS-Net ablation found that K=10 captured 99.49% of probability mass on Mistral-7B/HotpotQA and achieved 71.82 AUC, compared with 72.92 at K=1000. The paper further reports that K=10 captured more than 91% of mass in every evaluated model/dataset combination. This supports K=10 as a low-cost, pre-registered starting point, but does not prove it is optimal for ProcessBench.

## What LOS-Net actually uses

For a response of T tokens and vocabulary size V, LOS defines a token-distribution sequence with one full next-token distribution per position. It sorts every row independently and keeps its top K values. Sorting removes token identity and gives a fixed rank coordinate system that transfers across vocabularies.

LOS also retains two facts about the generated/observed token:

1. its own probability or log-probability;
2. its exact rank in the complete vocabulary distribution.

This matters because a chosen-token probability of 0.1 has different meaning when it is rank 1 and when it is rank 50.

The paper describes a learned projection across the K probability ranks, concatenation with the chosen-token/rank encoding, and an encoder Transformer across token positions. It is trained with response labels. The official implementation is more specific:

- it calculates `log_softmax` over the full vocabulary;
- it sorts those log-probabilities;
- it appends the chosen-token log-probability and its exact rank;
- preprocessing standardizes each token's sorted full-vocabulary log-probability vector before retaining the top K entries;
- LOS-Net applies a linear layer over the K rank coordinates, followed by a Transformer over the token sequence.

This is still direct use of the distribution. Standardization changes its coordinate scale; it does not collapse the row to one statistic such as entropy.

Primary sources: [AAAI paper](https://ojs.aaai.org/index.php/AAAI/article/view/40254), [extended arXiv version](https://arxiv.org/abs/2503.14043), [official repository](https://github.com/BarSGuy/Beyond-next-token-probabilities), [official preprocessing](https://raw.githubusercontent.com/BarSGuy/Beyond-next-token-probabilities/main/utils/dataset_preprocess.py), [official architecture](https://raw.githubusercontent.com/BarSGuy/Beyond-next-token-probabilities/main/utils/Architectures.py), and [official log-probability extraction](https://raw.githubusercontent.com/BarSGuy/Beyond-next-token-probabilities/main/utils/logits_handler.py).

## What the paper proves, and what it does not

The useful positive result is the TDS ablation. LOS-Net consistently beat learned baselines that received only the actual-token probability and rank. This is evidence that the other probability ranks contain information beyond the probability of the selected token.

The result does not directly establish our method because LOS-Net differs from our setting in three ways:

- it is supervised with labels from many other answers;
- it predicts whether a complete response is correct;
- its nonlinear Transformer can learn interactions across ranks and token positions.

Our target is answer-local, label-free fitting and first-error localization. Therefore the paper supports the **representation hypothesis**, not the expected ProcessBench gain.

The paper's transfer result is also a caution. For hallucination detection, zero-shot transfer was not sufficient to beat simple probability baselines; fine-tuning was needed. Our unsupervised fusion must therefore be tested directly rather than justified through LOS-Net's supervised AUC.

## Why this is new relative to our latest experiments

Step 334 compared token entropy, token varentropy, and answer-local IU-PCR over nine already-reduced token streams. Token entropy scored 35.44 ProcessBench / 0.7301 PRMBench within-answer AUC; token varentropy scored 35.68 / 0.7425; fusion of the nine reduced streams tied token entropy. Those results show that combining summaries did not help.

They do not test the current hypothesis. Entropy and varentropy are many-to-one summaries: different ranked probability profiles can produce the same scalar. The new experiment preserves the K rank coordinates before fusion.

## Frozen proposed representation

For every generated token t, obtain the full final-layer logits, align them to the token they predict, and form:

1. `tds_rank_1 ... tds_rank_10`: the ten largest log-probabilities, sorted descending and standardized as in the official LOS preprocessing;
2. `atp`: the selected token's standardized log-probability;
3. `atp_rank_encoded`: the selected token's standardized log-probability multiplied by the LOS scaled-rank term `1 - 2 * rank / V`.

This produces a matrix with tokens as rows and 12 direct distribution coordinates as columns. Token IDs may be stored for audit, but they must not enter fusion. The exact chosen-token rank should be computed over the full vocabulary, even when it falls below the saved top 10.

Because standardizing log-probabilities by their unweighted full-vocabulary mean and standard deviation is invariant to the shared `logsumexp` offset, the same standardized rank profile can be computed from raw logits. The raw top-10 log-probabilities should nevertheless also be stored so that the representation can be audited and alternative normalization can be evaluated only if this frozen test reveals a normalization failure.

## Frozen pipeline

The candidate remains a development of the project's fusion method:

`direct top-10 distribution signature -> answer-local Joint/LW fusion over rank columns -> one risk score per token -> existing top-10 token mean per reasoning step -> existing mean-entropy q=0.3 no-error gate`

The gate and readout must remain unchanged. This isolates the value of the new fusion axis. Raw mean entropy remains appropriate for the no-error gate because answer-local standardization intentionally removes much of the answer-level absolute scale.

The primary comparison should contain only:

1. frozen token entropy;
2. frozen token varentropy, reported as the prior post-hoc best single stream;
3. the prior nine-summary token IU result;
4. actual-token probability plus rank under the same new fusion implementation;
5. top-10 TDS plus actual-token probability and rank under Joint/LW fusion, the primary candidate.

Comparison 4 versus 5 tests the paper's central TDS claim inside our estimator. Comparison 1 versus 5 tests whether the resulting method improves the current localization pipeline.

Use the complete matched ProcessBench and PRMBench development population. Report ProcessBench all-eight macro F1, Q4/Q8, every cell, exact/early/late localization, clean-answer accuracy, coverage and paired uncertainty. Report PRMBench within-answer AUC and PRMScore; pooled AUC may remain descriptive. A small subset may be used only to smoke-test alignment and finite values.

Promote the candidate only if it improves ProcessBench over token entropy in the pre-specified paired comparison without a material PRMBench regression. If it improves within-answer ranking but not ProcessBench, inspect step aggregation and the gate rather than opening a rank/K search. If it does not beat the ATP+rank ablation, close direct rank fusion under the current answer-local estimator; that outcome would not refute supervised LOS-Net.

## Capture change that should be made before the queued GPU pass

The current localization layer-view driver already performs the expensive teacher-forced forward pass and has final-layer logits available. Before that job is submitted, its per-row output should also record:

- top-10 normalized log-probabilities for every generated token;
- top-10 token IDs for audit;
- selected-token log-probability;
- selected-token exact rank over the vocabulary;
- full-vocabulary log-probability/logit mean and standard deviation used by LOS normalization;
- full entropy, to reproduce the existing telemetry gate.

This adds little storage compared with the planned layer-by-token field and avoids a second complete GPU pass. The alignment test is load-bearing: logits at position i predict token i+1. The saved rank profile, selected-token score, and existing token entropy must all refer to the same predicted token.

## Information from papers that cite or extend this line

The citation chain is still sparse because the AAAI paper is recent. No verified citing work directly solves label-free first-error localization from output distributions.

Three later structure-aware works provide design guidance:

- [ACT-ViT](https://arxiv.org/abs/2510.00296) explicitly cites LOS-Net and treats the layer-by-token activation tensor as structured data. It supports preserving the natural axes instead of reducing them early, but it is supervised and white-box.
- [CHARM](https://arxiv.org/abs/2509.24770) represents tokens and attention flows as an attributed graph. It supports combining complementary computational traces, but it does not provide evidence for probability-rank fusion; our own graph controls also make this a lower-priority direction.
- [UniProbe](https://arxiv.org/abs/2608.10835) explicitly cites LOS-Net and moves structure-aware detection to token-level localization. Its useful lesson is to train or score the token target directly and preserve response order. It concerns supervised multimodal hallucination localization with hidden states and attention, so its performance cannot be transferred to ProcessBench.

The closest fixed, unsupervised analytic control is [Min-K%++](https://arxiv.org/abs/2404.02936), which LOS-Net cites. It contextualizes the selected token's log-probability using the full vocabulary distribution. It supports retaining the selected-token rank/relative score alongside the top-K profile, but its data-contamination result does not establish hallucination localization performance.

An informal LOS-Net reproduction reports that a local temporal convolution improved response-level AUC. No peer-reviewed paper or auditable code for that result was found. It should not enter the first experiment. Our previous onset, innovation and BOCPD tests already warn that temporal-change machinery can add complexity without improving first-error localization.

## Research claim if the experiment succeeds

A defensible framing would be:

> LOS-Net shows that full next-token distributions contain useful supervised response-level information. We show that a compact sorted output signature can instead be fused without labels, using only tokens from the answer being localized, to improve first-error localization in reasoning traces.

This keeps fusion at the center of the contribution and makes the relation to LOS-Net precise: we borrow the data representation, replace cross-answer supervised learning with answer-local fusion, and change the target from answer correctness to the first erroneous reasoning step.
