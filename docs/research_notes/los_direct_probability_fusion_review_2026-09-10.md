# LOS-style direct probability fusion for reasoning localization

**Date:** 2026-09-10  
**Scope:** A focused literature and code review for using the sorted next-token distribution directly inside the project's answer-local fusion method.

## Decision

The proposed experiment is justified and is materially different from the token-feature fusion already tested in Step 334.

The primary input must be the sorted probability/log-probability vector itself. It should not first be reduced to entropy components or other uncertainty summaries. Such a reduction would discard the distribution shape that LOS-Net was designed to retain.

The cached-data candidate uses **K=15**. We should not run a K sweep in the first experiment. The
reason is benchmark continuity: the saved token entropy was calculated from 15 retained
probabilities, while every audited cache contains at least 50. K=15 therefore gives the new method
and Token Entropy the same retained-support budget. LOS-Net's K=10 ablation supports the broader
claim that a compact sorted head can retain useful information, but it does not determine K for
ProcessBench.

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

## Frozen cached-data representation

The current caches do not contain the complete vocabulary vector, its mean and standard deviation,
or the chosen token's exact full-vocabulary rank. They do contain sorted top-50 log-probabilities,
the existing top-15 entropy, the scored token ID, and its negative log-probability in
`token_spilled_energies`. The earlier statement that the selected-token probability was unavailable
was incorrect; the v2 audit corrected it by matching this field exactly against the saved Top-50
entry whenever the selected token appears there. Therefore v1 formed:

1. `rank_1 ... rank_15`: the 15 largest saved log-probabilities, sorted descending;
2. direct probabilities `exp(log p)`, without top-K renormalization;
3. risk orientation `1-p1` for rank 1 and `p_rank` for ranks 2 through 15;
4. column standardization only across the fitting observations used by the project's fusion
   method.

This produces a matrix with tokens as rows and 15 direct distribution coordinates as columns. It
uses the probability values directly and does not collapse each row to entropy. It is inspired by
LOS-Net's sorted-distribution representation, but it is not an exact LOS-Net input reproduction.

V2 adds `selected-token surprisal = -log p(selected token)` and an explicit residual Top-15 mass.
This isolates the available ATP contribution. An exact LOS-style rank encoding and full-vocabulary
standardization still require a future capture. They must not be reconstructed from unavailable
fields or described as present in the cached run.

This is a new matrix-level combination, not the first use of either quantity in the project.
Step334's nine primitive token streams already included `spilled_series` and
`topk_tail_mass_series`; the historical mixed-v2 feature pools also contain engineered summaries
derived from spilled probability and tail mass. V2 asks whether their raw token-level coordinates
help specifically when appended to the direct Top-15 rank matrix under the same fusion estimator.

## Other conclusions from LOS-Net that matter here

- **LOS has two necessary parts in the paper's definition.** TDS records the sorted output
  distribution at every sequence position; ATP records the probability of the token that actually
  appears. Sorting the TDS removes token identity, so ATP must be supplied separately. The paper
  further encodes the ATP's exact rank to distinguish, for example, a probability of 0.1 at rank 1
  from the same probability at rank 50.
- **Distribution shape added signal beyond ATP and rank.** Across hallucination detection and data
  contamination detection, the full LOS-Net generally beat learned MLP and Transformer baselines
  that received only ATP plus rank. This supports keeping the 15 direct rank coordinates in our
  fusion rather than replacing them with selected-token surprisal.
- **A compact head can work, but K=10 was not the paper's default.** Main experiments used
  K=1000. In the K ablation over 10, 50, 100, 500 and 1000, performance improved weakly or stayed
  close as K grew, with diminishing returns after about K=100 in the highlighted case. Across all
  reported hallucination-detection model/dataset combinations, even K=10 captured more than 91%
  of probability mass. This justifies testing a compact head; it does not identify K=15 as optimal
  for our benchmarks.
- **Hallucination transfer needed adaptation.** Zero-shot transfer across an unseen HD model or
  dataset did not beat the simple probability baselines. Fine-tuning then beat those baselines in
  15/18 cross-model cases and 15/18 cross-dataset cases. We should therefore treat the paper as
  evidence for the representation, not as evidence that an unsupervised answer-local estimator will
  transfer automatically.
- **The target and supervision differ from ours.** LOS-Net learns a response-level binary label from
  many labeled examples and uses an encoder Transformer across sequence positions. It does not
  identify the first erroneous reasoning step and does not learn only from the answer being scored.
- **The theorem is about expressive capacity.** The paper proves that the architecture can
  approximate a broad class of LOS scoring functions, including common aggregation rules. This does
  not prove that the full distribution always improves accuracy, or that IU-PCR is the best way to
  fuse it.
- **Its speed claim begins after the output signature exists.** The reported detector forward pass is
  around 1e-5 seconds and the model has about one million parameters. This comparison does not
  include the cost of producing or transferring the LLM logits.
- **The authors identify extensions rather than tested localization gains.** They mention generated-
  text detection, exact-token flags and multiple prompting as future uses. These do not enter our
  frozen gray-box v2 experiment.

## Frozen pipeline

The candidate remains a development of the project's fusion method:

`direct top-15 distribution signature -> answer-local IU-PCR or Joint/LW fusion over rank columns -> one risk score per token -> existing top-10 token mean per reasoning step -> existing mean-entropy q=0.3 no-error gate`

The gate and readout must remain unchanged. This isolates the value of the new fusion axis. Raw mean entropy remains appropriate for the no-error gate because answer-local standardization intentionally removes much of the answer-level absolute scale.

The primary cached-data comparison contains:

1. frozen token entropy;
2. frozen token varentropy, reported as the prior post-hoc best single stream;
3. the prior nine-summary token IU result;
4. direct top-15 probability fusion with equal weights;
5. direct top-15 probability fusion with IU-PCR, the primary candidate;
6. direct top-15 probability fusion with Joint/Ledoit-Wolf shrinkage, a supporting candidate.

The completed v1 cached run tested whether retaining the sorted head helps our estimator. The v2
experiment isolates the saved selected-token probability plus an explicit tail-mass summary. It still
cannot test the paper's exact selected-token rank because that field was not saved.

Use the complete matched ProcessBench and PRMBench development population. Report ProcessBench all-eight macro F1, Q4/Q8, every cell, exact/early/late localization, clean-answer accuracy, coverage and paired uncertainty. Report PRMBench within-answer AUC and PRMScore; pooled AUC may remain descriptive. A small subset may be used only to smoke-test alignment and finite values.

Promote the candidate only if it improves ProcessBench over token entropy in the pre-specified paired comparison without a material PRMBench regression. If it improves within-answer ranking but not ProcessBench, inspect step aggregation and the gate rather than opening a rank/K search. If it does not beat the ATP+rank ablation, close direct rank fusion under the current answer-local estimator; that outcome would not refute supervised LOS-Net.

## Fields to save in a future GPU capture

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
