# PRM boosting positioning: research record for peer review (2026-10-23)

**Who and why.** Claude, on Omri's request (2026-10-22/23). Omri asked whether the BR2-PRM paper
(ACL 2026) blocks publication. He then asked for a review of the positioning page
(<https://claude.ai/artifact/BiZC1yYTyMvGkosaXFPCRz>), a deeper read of the competing papers, and an
update of the page. This file is the record for an independent peer review on the other computer.

**What is in this folder.**
- `PRM_BOOSTING_POSITIONING.html`: standalone copy of version 4 of the published page. It opens offline;
  only Google Fonts loads from the network.
- `PEER_REVIEW_HANDOFF.md`: this file.

**No experiment was run.** Everything below is reading papers and re-checking existing result tables on
branch `claude/prm-boosting`. All PRMBench numbers are development data.

---

## 1. Short answer

1. **BR2-PRM does not block publication.** It is a different task (choosing the best of N whole answers),
   a different modality (multimodal), and it is supervised (naive Bayes fitted by counting labels). It does
   not cite Nadler. Cite it as related work.
2. **FUSE (ICML 2026, Candès group) is the closest method to ours**, and the old page filed it under
   "framing only". FUSE builds directly on Nadler's line (Parisi 2014; Jaffe 2015, 2016). It is label-free,
   but it works only at the response level. Our novelty is the level and the views (step level, within one
   answer, views from the PRM's own forward pass), not the estimator.
3. **CARE (ICML 2026)** is a second close estimator: dependency graph, then three conditionally independent
   groups, then tensor CP. It is not step-level and does not cite Nadler.
4. **The old page overstated two claims.**
   - The CROP margin (22.3% vs 21.8%) is about a quarter of our seed SD (1.9), so it is a tie.
   - The PRISM-level PRMScore comes mostly from the GAP flag rule. Fusion adds +2.39 under the same rule.
5. **Nothing found does label-free step-level fusion of a PRM with its own telemetry.** This rests on
   searches, not proof (Section 6).

---

## 2. Papers read and how

| Paper | How it was read | Coverage |
|---|---|---|
| BR2-PRM / PRMBench-V (ACL 2026) | Official ACL PDF, text extracted with `pypdf` (31 pages), grepped and read by me | full main text, App. D, reference list |
| FUSE (arXiv 2604.18547) | Local extraction `papers/extracted/fuse-ensembling-verifiers-with-zero-labeled-data.md`, read by me | Sec. 1-3, related work, discussion, App. A-B and the references grep |
| CARE (arXiv 2603.00039 v2) | Sub-agent read the arXiv abstract and HTML | Sec. 1-7 and references verbatim; parts of App. C-E only as a model summary |
| Unsupervised Multimodal Reward Modeling (arXiv 2610.12707v1) | Sub-agent read the arXiv HTML | about 78% verbatim; last ~22% of references summarized |
| Mind the Gap (ICML 2026), PRISM, CROP, ARES, SLIDER, R-PRM | Existing primer `docs/research_notes/competitor_primer_2026-10-22/` (written by other agents on 2026-10-22) and `papers/digests/crop-conformal-clean-prefix.md` | not re-read by me; numbers taken from the primer |
| Ising aggregation (2601.22336), Weaver, ScalePRM, uPRM, Optimal Aggregation | `docs/research_notes/literature_sweep_2026-10-21/` and `competitor_registry_2026-10-21/REGISTRY.md` | not re-read by me |

Two quick checks a reviewer should repeat: the BR2-PRM bibliography grep, and the missing Nadler references
in CARE. The commands are in Section 7.

---

## 3. Findings per paper

### 3.1 BR2-PRM (Chan, Zhu, Jiang, Ji, Dai, Xue, Han, Guo; HKUST and PKU; ACL 2026, pp. 44672-44702)

- **Main contribution** is the PRMBench-V benchmark: 907 multimodal queries × 9 error types = 8,163 cases,
  step-level labels. The best judge reaches about 0.34 strict accuracy. BR2-PRM is Sec. 4.6 plus App. D.
- **Task.** Best-of-N selection (N = 2-16) on MMK12 with Gemma-3-12B-it and LLaVA-1.5-7B policies. It scores
  whole answers and does not localize steps.
- **Input.** One multimodal judge generation per candidate: an integer 1-5 score for each of nine error types.
  No token probabilities.
- **Algorithm.**
  - For each type k: z_k ~ Bernoulli(π_k), s_k | z ~ Categorical(θ_{k,z}), with Dirichlet priors.
  - Fit (Alg. 1): add counts of (score, known label) on PRMBench-V to the prior; π_k is the empirical rate.
    **No EM, so it is supervised naive Bayes**, despite the "Dawid-Skene" framing.
  - Score: S = Π_k P(z_k = 0 | s_k). Choose the argmax.
  - An informal O(1/√N) selection bound.
- **Result.** Up to +4.8 points over plain averaging of the judge's scores, and 1.2-1.8 over a
  reliability-weighted average (text). The primer measured 0.27-1.67 off Fig. 6.
- **Citations.** Dawid & Skene (1979) is the only reliability-model reference. A grep for
  Nadler, Parisi, Jaffe, Kluger, "spectral", "unsupervised", "label-free", "telemetry", "logit" and
  "hidden state" finds 0 hits.
- **Inconsistencies** (from the primer, consistent with my read):
  - the 1-5 score scale vs 0-5 in the prompt;
  - Eq. 11 drops the prevalence terms;
  - MMK12 is both a calibration source and the test set.

### 3.2 FUSE (Lee, Ma, Zhao, Nair, Spector, Cohen, Candès; Stanford and Google; arXiv 2604.18547, ICML 2026)

- **Task.** Best-of-N response selection. An N × m matrix of scores from m verifiers (reward models and
  LLM judges, up to 33) over N responses to one question.
- **Labels.** None. It runs per question ("query-conditional") or pooled ("batched").
- **Algorithm.**
  1. Transform each verifier's scores; the default is a per-verifier threshold. The thresholds minimize a
     label-free statistic of triplet conditional-independence (TCI) violation, Prop. 2.4, which uses
     ratios of the third- to second-order covariance tensors.
  2. Apply the Jaffe et al. (2015) method of moments to get each verifier's sensitivity and specificity.
  3. Use those estimates to build pseudo-labels, then fit an aggregator (for example a logistic regression)
     on them. This avoids the joint conditional-independence assumption.
- **Builds on Nadler:** Parisi, Strino, Nadler, Kluger (2014); Jaffe, Nadler, Kluger (2015); Jaffe, Fetaya,
  Nadler, Jiang, Kluger (2016); Shaham et al. (2016); Dawid & Skene (1979).
- **Results.**
  - It matches Weaver, which uses labels for 5% of questions (GPQA Diamond 64.4 vs 64.1, 70B setting).
  - It wins 27 of 40 comparisons against supervised baselines.
  - Datasets: GPQA, GPQA Diamond, MATH500, MMLU, MMLU-Pro (Saad-Falcon et al. data), HLE (649 questions),
    and IMO Shortlist (123).
- **Key finding for us.** The plain Jaffe (2015) closed-form ensemble "under-performs a simple naive
  ensemble in 7 out of 10 settings" (Fig. 3). That mirrors our result: on the corrected-template PRMBench
  banks, the plain mean matches or beats L-SML on 7 of 10 banks (`PROGRESS.md` 2026-10-21, Step 468).
- **Step level.** None. Its stated limitation is that it needs samples from several distinct verifier models.

### 3.3 CARE (Zhao, Shin, Huang, Namburi GNVV, Sala; UW-Madison; arXiv 2603.00039, ICML 2026)

- **Task.** Aggregate LLM-judge scores per item: real-valued ratings (MAE) or classification/preference
  (accuracy).
- **Algorithm.**
  1. Estimate the precision matrix.
  2. Split it into sparse + low-rank (penalized least squares, γ_n and τ).
  3. **SVD variant** (Gaussian): the low-rank part gives the latent quality loadings.
     **Tensor variant** (mixture): the sparse graph splits the judges into three conditionally independent
     groups, then a third-order CP decomposition gives the class means and priors.
  4. Break the sign/label symmetry with the leading eigenvector (default) or a human anchor set.
  5. Score with the posterior E[Q | J] or P(Q = 1 | J).
- **Labels.**
  - The core is label-free.
  - Hyperparameters are tuned on a 15% labeled split, and the programmatic-judge choice uses 10% labeled data.
  - App. E.8 says the best-factor choice "uses the test set" and is optimistic.
- **Results.** 12 benchmarks; the 26.8% error reduction is UltraFeedback MAE 0.851 (majority vote) → 0.623.
  Best on 5 of 6 classification/preference sets. No step-level reasoning data.
- **Citations.** Anandkumar et al. (2014), Chandrasekaran et al. (2012), and weak supervision (Ratner, Bach,
  Fu, Shin). **No Nadler, Parisi or Jaffe, and no FUSE.**
- **Code.** github.com/SprocketLab/CARE.
- **Unverified:** how the three-group partition is searched; the exact CARE-SVD weighting formula.

### 3.4 Mind the Gap (Chen, Chen, Yue, Li; ICML 2026, PMLR v306); taken from the 2026-10-22 primer

- One channel: drops in the negative top-20 entropy, EMA-smoothed; one teacher-forced pass; no PRM.
- The score is label-free; thresholds are set on a labeled half of the test data.
- ProcessBench exact first-error accuracy, e.g. MATH Qwen3-8B 32.90 vs 24.62 for average entropy. No PRMBench.
- The released code covers final-answer detection only and uses top-1 log-prob, not top-20 entropy.

### 3.5 Unsupervised Multimodal Reward Modeling (Li, Fan, Brantley, Kakade, Zhang; arXiv 2610.12707v1, 8 Oct 2026)

- Response-level. Trains a bias-free 2,048-parameter linear head on frozen Qwen3-VL hidden states with an
  in-batch Bradley-Terry loss against perturbed negatives.
- No ensembling, no telemetry fusion, no step localization, no PRMBench.
- Cites none of Nadler, Parisi, Jaffe or Weaver.
- **Not a competitor.** Listed here only because it came up in the search.

### 3.6 Others (from the registry and sweep, not re-read)

| Paper | Why it is not a step-level label-free competitor |
|---|---|
| Weaver (NeurIPS 2025) | weak supervision at the response level |
| Ising aggregation (ICML 2026) | EM on binary votes; theory that conditional-independence rules are suboptimal under dependence |
| ScalePRM | aggregates step verdicts only to make training labels |
| uPRM | trains a PRM without labels |
| Optimal Aggregation (ICLR 2026) | labeled calibration, response level |

None reports step-level PRMBench.

---

## 4. Verification of the positioning page against result tables

All checked on branch `claude/prm-boosting`, Qwen2.5-Math-PRM-7B, PRMBench.

| Claim on the page | File | Checked value | Status |
|---|---|---|---|
| PRM native TPR 95.40 / FPR 69.42 / 65.49 | `results/prm_boost_collection_v1/tables/COMP_PRISM_T1_T2.csv` | 95.400 / 69.420 / 65.488 | match |
| PRM under GAP 68.01 | same | 68.008 (Overall, `COMP_PRISM_T7_CATEGORIES.csv`) | match |
| L-SML (CHAIN_N16) under GAP: TPR 88.57 / FPR 45.06 / 70.40 | `COMP_PRISM_T1_T2.csv` | 88.575 / 45.060 / 70.399 | match |
| Matched TPR 87.3: PRM FPR 47.70, L-SML 47.78 | `COMP_PRISM_MATCHED_TPR.csv` | 47.699 / 47.783 | match |
| Label-chosen channels (LSML_CONT) at matched TPR: FPR 44.22 | same | 44.216 | match (not label-free) |
| MS category: native 99.87, PRM+GAP 94.64, L-SML+GAP 88.87 | `COMP_PRISM_T7_CATEGORIES.csv` | 99.866 / 94.64 / 88.867 | match |
| GAP vs constant k=2: −0.35 | `results/error_count_label_free_v1/README.md` line 21 | −0.35 [−0.72, +0.00] | match |
| CROP protocol, L-SML train-only fit 22.3 / 23.2% kept | `CROP_C4_INDUCTIVE.csv` (CHAIN_N16 inductive) | 22.334 / 23.182 | match |
| Seed SD of that row | same, `kept_sd` | 1.946 / 1.737 | **new on page** |
| L-SML fitted on all traces 21.9 / 23.3%, SD | `CROP_C1_MAIN.csv` | 21.895 / 23.257, SD 1.853 / 1.885 | match |
| Label-chosen channels 24.4% | `CROP_C1_MAIN.csv` LSML_CONT fixed101 | 24.401 | match |
| Other seed SDs in the CROP table | `CROP_C1_MAIN.csv` `kept_sd` | Direct PRM 0.27/0.36; format LR 2.05/2.14; format+PRM 2.15/2.32; tele13 2.03/1.99; tele16 2.00/2.11 | **new on page** |
| CROP published Trace+PRM 21.8% / AUROC .834 | `papers/digests/crop-conformal-clean-prefix.md` | 21.8 / .8336 | as digested; CROP's own seed SD is not known |
| Plain mean ≥ L-SML on 7 of 10 banks | `PROGRESS.md` 2026-10-21 entry (Step 468) | as written there | not recomputed |

**Not re-checked:** the other-PRM rows (PURE, RLHFlow, Math-Shepherd, Qwen-72B), the ARES, SLIDER and
Socratic rows, and the leaderboard values. They are unchanged from version 1 of the page.

---

## 5. Changes made to the page (version 1 → version 4, same URL)

1. **Lede rewritten.** "Matches CROP", "reaches PRISM's PRMScore", and the flag rule named as the main
   source. FUSE is named as the same estimator family.
2. **Map table: four rows added.** FUSE, CARE, Mind the Gap, BR2-PRM. The old
   "FUSE, Weaver, ScalePRM, uPRM, Optimal Aggregation" row became
   "Weaver, Ising aggregation, ScalePRM, uPRM, Optimal Aggregation".
3. **PRISM section.**
   - Retitled "reached without retraining, mostly through the flag rule".
   - Added three key figures: 47.70 vs 47.78; 68.01 ≈ 68.0; +2.39.
   - Added a bullet on the rule's built-in prior: the MS category drop and GAP not beating constant k=2.
   - The claim now says not to say "beats PRISM".
4. **CROP section.**
   - Retitled "matched without them".
   - Added a seed-SD column.
   - The Qwen-7B bullet now calls it a tie, and says only the label-chosen variant (24.4%) clears it.
5. **New section "The label-free ensembling family: FUSE and CARE"**:
   - a method summary for each;
   - the 7-of-10 parallel;
   - what is new in our work;
   - candidate fusion rules;
   - the Ising theory.
6. **New section "Two more neighbours":** Mind the Gap and BR2-PRM, each with a "fair test" line.
7. **Claims.**
   - "Not supported" now includes beating PRISM, beating CROP, and a new estimator.
   - "Open before a paper" now lists the FUSE-at-step, CARE, BR2-analogue and Mind the Gap baselines, plus
     a constant-k reference.
8. **Footer.** Sources and paper links added. Date set to 23 October.

Charts and leaderboard tables are unchanged.

---

## 6. Conclusions

- **Publication is not blocked** by any paper found. The supportable novelty statement:

  > Label-free, step-level fusion of a frozen PRM with telemetry from its own forward pass, evaluated with
  > official PRMBench PRMScore. The estimator belongs to the Nadler / Jaffe moment-based family, also used at
  > the response level by FUSE and, with tensor groups, by CARE.

- **Reviewer exposure:**
  1. A reviewer who knows FUSE will ask why we did not run FUSE at the step level. This is CPU-only on
     existing scores.
  2. A reviewer who knows CARE will ask the same about CARE.
  3. The GAP rule's PRMBench prior makes raw leaderboard comparisons unfair to native-threshold PRMs.
     Always pair "L-SML vs its own PRM under the same rule" with a constant-k reference.
- **Useful for the open fusion direction** (Omri, 2026-10-01: replace the plain average with an estimate-based
  combination):
  - FUSE's dependence-reducing transform plus a pseudo-label aggregator, and CARE's dependency-graph grouping,
    are concrete, published candidates.
  - FUSE's 7-of-10 result says the averaging problem is general, not a bug in our pipeline.
- **Search limits.**
  - Semantic Scholar citers of PRMBench (136), Socratic (3) and Hard2Verify (22), from the 2026-10-21 registry.
  - The venue sweep of 2026-10-21.
  - Three web searches on 2026-10-22/23 (Section 8).
  - Absence of a competitor is a search result, not proof.

---

## 7. Questions for the peer reviewer

1. Re-grep the BR2-PRM PDF bibliography for Nadler / Parisi / Jaffe / Kluger. Expected: 0 hits.
   ```bash
   python -I -c "from pypdf import PdfReader as R;import sys;t=''.join(p.extract_text() or '' for p in R(sys.argv[1]).pages);import re;print({k:len(re.findall(k,t,re.I)) for k in ['nadler','parisi','jaffe','kluger','dawid','unsupervised']})" 2026.acl-long.2068.pdf
   ```
2. Confirm in the CARE reference list (arXiv HTML v2) that Nadler, Parisi and Jaffe are absent. Read the
   appendix sections that came only as a summary (C, D, E) to check how the three-group partition is chosen.
3. Re-derive the CROP seed SD: `kept_sd` in `results/prm_boost_collection_v1/tables/CROP_C4_INDUCTIVE.csv`
   for `CHAIN_N16 (inductive, C4)`. Is "a tie" the right word, given that CROP's own seed SD is unknown?
4. Check that "fusion adds +2.39 under the same rule" (68.01 → 70.40) uses the same answers and the same
   evaluator port for both rows (`COMP_PRISM_T1_T2.csv`, the non-control 6,211-answer pool).
5. Is there a step-level, label-free PRM ensemble published after 2026-10-21 that we missed?
   Suggested searches: arXiv listings citing FUSE (2604.18547), and citers of CARE.
6. Does the page now overstate or understate anything? In particular the "Supported" list and the lede.

---

## 8. Sources and citations

**Papers (primary sources).**
- Chan, C.-M., Zhu, H., Jiang, C., Ji, J., Dai, J., Xue, W., Han, S., Guo, Y. (2026). *Benchmarking
  Fine-Grained Error Detection in Multimodal Reasoning.* ACL 2026 (Long Papers), pp. 44672-44702.
  <https://aclanthology.org/2026.acl-long.2068/> · PDF <https://aclanthology.org/2026.acl-long.2068.pdf>
- Lee, J., Ma, V., Zhao, S., Nair, Y., Spector, A., Cohen, R., Candès, E. (2026). *FUSE: Ensembling Verifiers
  with Zero Labeled Data.* arXiv:2604.18547; ICML 2026 (OpenReview LeRgUcmLF3). <https://arxiv.org/abs/2604.18547>
- Zhao, J., Shin, C., Huang, T.-H., Namburi GNVV, S. S. S., Sala, F. (2026). *CARE: Confounder-Aware
  Aggregation for Reliable LLM Evaluation.* arXiv:2603.00039v2; ICML 2026 (PMLR v306, zhao26aq).
  <https://arxiv.org/abs/2603.00039> · code <https://github.com/SprocketLab/CARE>
- *Dependence-Aware Label Aggregation for LLM-as-a-Judge via Ising Models.* arXiv:2601.22336; ICML 2026.
  <https://arxiv.org/abs/2601.22336> (authors not checked here)
- Chen et al. (2026). *Mind the Gap: Catching Hallucinations via Evidence Drop.* ICML 2026, PMLR v306 (chen26n).
  <https://proceedings.mlr.press/v306/chen26n.html> · code <https://github.com/QJ0114/evidence-drop>
- Li, Y., Fan, J., Brantley, K., Kakade, S. M., Zhang, H. (2026). *Unsupervised Multimodal Reward Modeling.*
  arXiv:2610.12707v1. <https://arxiv.org/html/2610.12707v1>
- PRISM, arXiv:2606.09078. CROP (Cheung et al.), arXiv:2605.30085v2. Weaver (Saad-Falcon et al.),
  arXiv:2506.18203, NeurIPS 2025. uPRM (Gadetsky, Kodryan et al.), arXiv:2605.10158.
  <https://arxiv.org/pdf/2605.10158>
- Song et al. (2025). *PRMBench: A Fine-grained and Challenging Benchmark for Process-Level Reward Models.*
  ACL 2025. <https://aclanthology.org/2025.acl-long.1230/>

**Foundational references (the estimator family).**
- Dawid, A. P., Skene, A. M. (1979). Maximum likelihood estimation of observer error-rates using the EM
  algorithm. *JRSS Series C* 28(1):20-28.
- Parisi, F., Strino, F., Nadler, B., Kluger, Y. (2014). Ranking and combining multiple predictors without
  labeled data. *PNAS* 111(4):1253-1258.
- Jaffe, A., Nadler, B., Kluger, Y. (2015). Estimating the accuracies of multiple classifiers without
  labeled data. *AISTATS 2015.*
- Jaffe, A., Fetaya, E., Nadler, B., Jiang, T., Kluger, Y. (2016). Unsupervised ensemble learning with
  dependent classifiers. *AISTATS 2016.*
- Anandkumar, A., Ge, R., Hsu, D., Kakade, S. M., Telgarsky, M. (2014). Tensor decompositions for learning
  latent variable models. *JMLR* 15.

**Search hits seen but not read in full.**
- <https://arxiv.org/pdf/2605.29009> (Cross-Model Entropy, label-free RL reward; no PRMBench)
- <https://arxiv.org/html/2504.16828v3> (ThinkPRM, trained with labels)
- <https://arxiv.org/html/2502.11250v1> (Uncertainty-aware step-wise verification)
- <https://arxiv.org/html/2601.17223v1> (Verifiable PRMs)
- <https://arxiv.org/html/2605.16567> (Unsupervised ensemble outlier model selection)
- <https://arxiv.org/html/2601.20556> (energy-based unsupervised ensemble)

**Project files used.**
- `results/prm_boost_collection_v1/tables/`
  (`COMP_PRISM_T1_T2.csv`, `COMP_PRISM_MATCHED_TPR.csv`, `COMP_PRISM_T7_CATEGORIES.csv`,
  `COMP_PRISM_PAPER_ROWS.csv`, `CROP_C1_MAIN.csv`, `CROP_C4_INDUCTIVE.csv`)
- `results/error_count_label_free_v1/README.md`
- `PROGRESS.md` (2026-10-21 entry)
- `docs/research_notes/competitor_primer_2026-10-22/`
- `docs/research_notes/competitor_registry_2026-10-21/REGISTRY.md`
- `docs/research_notes/literature_sweep_2026-10-21/`
- `papers/extracted/fuse-ensembling-verifiers-with-zero-labeled-data.md`
- `papers/digests/crop-conformal-clean-prefix.md`
