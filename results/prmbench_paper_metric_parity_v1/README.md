# Apples to apples with four PRMBench-citing papers (2026-10-18)

**What this is.** For each of the four papers we position "PRM boosting" against, we checked whether the paper's
metric is the same as ours. Where it was not, we computed the paper's own metric for our arms on existing local
data, and redrew the paper's table or figure with our points added. CPU only. No model inference, no AIRCC job,
no Drive write. Nothing is committed.

**Our arms.** All four are Qwen2.5-Math-PRM-7B on PRMBench (6,969 answers; 6,211 non-control answers enter
PRMScore):

| arm | what it is | labels |
|---|---|---|
| PRM_native | the PRM alone, its native rule (reward ≤ 0.5 means error) | the PRM was trained with labels by its authors |
| LSML_CONT | L-SML of PRM + 3 channels | the 3 channels were **chosen with PRMBench labels** |
| CHAIN_N16 | L-SML CHAIN of PRM + 16 channels | label-free channel list; **contains 3 digit channels** |
| CHAIN_N13 | the same with 13 channels | label-free and digit-free |

Every fusion is fitted once, without labels, on the pooled steps of all 6,969 answers. That is transductive: it
is **not answer-local fitting**. The fused arms use the label-free Q80 rule, which flags the top 20% of
answer-standardized step scores.

**Parity before use (`PARITY.csv`).** The CHAIN per-step scores exist only on AIRCC, so `common.py` refits them from
the local exported channels. All four arms reproduce
`results/prm_boosting_bank_sweep_v1/SWEEP_RESULTS_qwen_prm_7b.csv` exactly (difference 0.0 up to 1e-14):

| arm | PRMScore | within-AUC |
|---|---|---|
| PRM_native | 65.49 | 0.8012 |
| LSML_CONT | 69.48 | 0.8495 |
| CHAIN_N16 | 68.25 | 0.8377 |
| CHAIN_N13 | 67.82 | 0.8288 |

So no arm is a GAP for missing scores.

Every number below is from `METRICS.csv` (216 rows). The underlying values are in `*_RESULTS.json`.

---

## 1. PRISM (arXiv 2606.09078). Re-trains Qwen-PRM-7B with labels. See `PRISM_PARITY.png`.

**Same metric? Yes for PRMScore, the 9 categories, FPR, TPR and TNR.** In PRISM the positive class is a *correct*
step, as in the official PRMBench code. FPR is the share of error steps the PRM passes as correct, which equals
1 − TNR.

Our PRM_native reproduces their baseline on 6,211 answers / 83,371 steps:

| metric | theirs | ours |
|---|---|---|
| PRMScore | 65.5 | 65.49 |
| FPR | 69.34 | 69.42 |
| TPR | 95.36 | 95.40 |
| TNR | 30.66 | 30.58 |

All nine Table 7 categories match within 0.15, except MS (99.7 vs 99.87). MS has no error steps, so its score is the
valid-step F1.

PRISM's **FNR (4.27) and Precision (89.40) do not fit with the rest of its own table**: TPR + FNR ≠ 100. They are
reproduced only if the 758 synthetic all-correct control rows are pooled in (ours: 4.24 / 89.48). We report both
pools.

**How we compare (PRMBench, deployed rules):**

| | FPR ↓ | TPR | PRMScore |
|---|---|---|---|
| PRISM (re-trained PRM) | 47.13 | 87.30 | 68.0 |
| LSML_CONT | **43.55** | 86.90 | **69.48** |
| CHAIN_N16 | 45.68 | 86.40 | 68.25 |
| CHAIN_N13 | 46.54 | 86.27 | 67.82 |

**Important control.** FPR depends strongly on the threshold. We set each arm's global threshold so that its TPR
matches PRISM's 87.30. Picking that point uses labels, so it is a reading, not a deployable rule:

| arm at TPR 87.30 | FPR | PRMScore |
|---|---|---|
| untouched PRM, threshold moved only | 47.70 | 68.26 |
| LSML_CONT | 44.22 | 69.58 |
| CHAIN_N16 | 47.78 | 68.20 |
| CHAIN_N13 | 48.53 | 67.94 |

Moving the threshold alone gets the untouched PRM to roughly PRISM's FPR and PRMScore (47.70 / 68.26 vs
47.13 / 68.0). At matched TPR, only LSML_CONT is clearly better than both, and LSML_CONT's channels were chosen
with labels. The label-free CHAIN arms are no better than the moved-threshold PRM. The category picture
(Table 7) is the same for all three fused arms:
- **Large gains:** NR, NCL and DC (e.g. NR 67.7 vs PRISM-C4 53.5).
- **Losses:** CI, DR and MS. Q80 flags 20% of the steps even in MS, where no step is wrong.

**Still missing:** ProcessBench (Table 6) for the fused arms. This is a **GAP** because no PRM-backbone telemetry
exists on ProcessBench, and collecting it needs new GPU inference. PRM_native matches their baseline on GSM8K
(71.0/96.4/81.8) and MATH (67.0/91.1/77.2). Their Omni-MATH and OlympiadBench cells look transposed: ours are
OlympiadBench 54.8/85.3/66.7 and Omni-MATH 54.7/83.4/66.1. PRISM's PB rows are therefore not used as their claim.

## 2. CROP (arXiv 2605.30085). Conformal clean-prefix certification. See `CROP_PARITY.png`.

**Same metric? No.** CROP reports a **pooled** step AUROC (all test steps together; ours is within-answer) and
the **retained-prefix fraction** at α = 0.05. CROP's threshold is calibrated with labels on a held-out
calibration split, for every risk proxy. That is a different access level from our label-free rule.

**Their protocol.** Read from the paper and their GitHub code:
- the first 5,000 rows of `hitsmy/PRMBench_Preview` in dataset order;
- 10 seeds (2806–2815), with trace-level 60/20/20 splits stratified by has-error;
- a fixed 101-point grid on [0, 1];
- the largest λ with (1 + #contaminated) / (n + 1) ≤ α;
- "kept" is the mean over test traces of M/T; "risk" is the share of test traces whose kept prefix contains an
  error.

**Our reproduction (APPROXIMATE subset).** We used the first 5,000 non-control rows in our loader's Hub order:
67,465 steps vs their 67,445, with an error rate of 15.88% vs their 15.9%. Our loader dropped 5 duplicate rows
that CROP keeps. The splits re-implement their `split_traces`; the exact assignment may differ.

**Check:** our Direct PRM on their fixed grid gives AUROC 0.7999 ± 0.0038, kept 4.8 ± 0.2%, risk 0.6 ± 0.1%.
That is exactly their row (0.8010 / 4.8 / 0.6). Random gives 9.2% kept (theirs 8.7%).

**How we compare** (risk stays ≤ 5.0% for every arm):

| proxy | access | step AUROC (pooled) | kept % | risk % |
|---|---|---|---|---|
| Trace+PRM (theirs) | supervised LR | 0.8336 | 21.8 | 5.0 |
| Direct PRM (theirs) | fixed grid | 0.8010 | 4.8 | 0.6 |
| Direct PRM, same scores, quantile grid (ours, bridge) | — | 0.7999 | 17.2 | 3.9 |
| LSML_CONT | label-free fusion | 0.8018 | **25.1 ± 1.5** | 4.6 |
| CHAIN_N16 | label-free fusion | 0.7943 | 23.3 ± 0.9 | 5.0 |
| CHAIN_N13 | label-free fusion | 0.7815 | 23.0 ± 1.3 | 5.0 |
| Telemetry13+PRM LR (ours, supervised analog of Trace+PRM) | supervised LR | 0.8285 | 24.8 | 4.4 |
| Position/length-only LR (ours, supervised) | supervised LR | 0.7488 | 11.8 | 4.9 |

Two things stand out:
- **Direct PRM's 4.8% is mostly a grid effect.** The same PRM scores on a finer grid keep 17.2%.
- **Our pooled AUROC is lower than our usual within-answer AUC.** On this subset LSML_CONT is 0.840 within-answer
  but 0.802 pooled. Our scores are partly answer-standardized.

All three label-free fused arms keep more prefix than every published row on PRMBench, at the same nominal risk.
CHAIN_N13 is digit-free.

**Protocol differences that remain:**
- Our fused arms cannot use a [0, 1] grid. They use 201 train-split quantiles, which is CROP's own target-domain
  rule.
- The fusion was fitted label-free on all 6,969 answers, including these traces.

**Still missing:** their Token/format and Trace-features proxies. They need the step text (not local) and
Llama-3.1-8B features. Our position/length LR is only a subset of Token/format.

## 3. SLIDER (arXiv 2610.00571). Redundancy detection in PRMBench's redundancy class. See `SLIDER_PARITY.png`.

**Same metric? Unknown, so treated as APPROXIMATE.**

What the paper states:
- positive = steps annotated redundant;
- "around 750 problems";
- the threshold comes from a linear classifier on a held-out set;
- the metric is "accuracy": Step-RRI 61.91%, InfoGain 48.38%, Cosine 51.60%.

What it does not state: the split, whether step 1 is scored, class weighting, or whether the accuracy is
balanced.

Our class has 758 problems / 11,613 steps, of which 13.2% are redundant. Answering "not redundant" every time
already gives 86.8% plain accuracy, so 61.91% cannot be plain accuracy with an unweighted threshold at the
natural base rate. Our closest readings (10 random 50/50 trace splits, held-out 1-D logistic threshold):

| arm | accuracy, balanced LR threshold | balanced accuracy | label-free Q80, balanced accuracy |
|---|---|---|---|
| PRM_native | 50.4 | 60.4 | 50.7 |
| LSML_CONT | 71.2 | 72.3 | 71.8 |
| CHAIN_N16 | 74.1 | 74.3 | 72.8 |
| CHAIN_N13 | 72.7 | 73.7 | 73.3 |

Under every reading we computed, the fused arms are 9–12 points above Step-RRI's 61.91, even without threshold
labels (Q80). Restricting to steps ≥ 2 changes the fused-arm numbers by less than 1 point.

**Still missing:**
- SLIDER's exact split and accuracy definition (not in the paper; the authors would need to confirm them).
- Their Step-RRI scores, so that a matched comparison could be run.

## 4. Optimal Aggregation of LLM and PRM Signals (arXiv 2510.13918). See `OPTAGG_PARITY.png`.

**Same metric? No, and it cannot be computed. Every arm is a GAP.**

Their metric is the final-answer accuracy of Best-of-N / weighted voting over 32 (MATH) or 112 (MATH500) sampled
responses per question. The weights are calibrated on a labeled set of questions. Their Table 1 for
Qwen2.5-PRM-7B, averaged over 5 LLMs:

| method | accuracy |
|---|---|
| Logit WV | 63.3 |
| BoN | 61.8 |
| MV | 57.1 |
| oracle | 66.5 |

PRMBench has one fixed trace per perturbation and no graded final answers.

**Data needed:** a pool of sampled responses per question (e.g. MATH500, N ≥ 32) with graded final answers, plus
the PRM rewards and backbone telemetry from the same forward pass. That means new generation and inference, which
is not authorized here.

`aggregation_gap.py` also reports a **proxy that is not their metric**: inside each of PRMBench's 707 source groups,
pick the trace with the lowest maximum step risk.

| arm | clean trace picked |
|---|---|
| random pick | 13.1% |
| PRM_native | 51.6% |
| LSML_CONT | 26.4% |
| CHAIN_N16 | 20.5% |
| CHAIN_N13 | 16.5% |

The fused scores rank answers against each other worse than the PRM does. This is expected, because they contain
an answer-standardized PRM term. It is a real limitation if boosting were used for Best-of-N.

---

## Files

- **Scripts:** `common.py` (scores + parity), `prism_metrics.py`, `crop_metrics.py`, `slider_metrics.py`,
  `aggregation_gap.py`, `build_table.py`, `plot_papers.py`, `run_all.sh` (whole pipeline, about 30 s).
- **Results:** `METRICS.csv`, `PARITY.csv`, `PRISM_RESULTS.json`, `CROP_RESULTS.json`, `SLIDER_RESULTS.json`,
  `AGGREGATION_PROXY.json`, `FUSION_DIAG.json`.
- **Scores:** `ARM_SCORES.npz` (per-step scores of the four arms, 2.3 MB).
- **Figures:** `PRISM_PARITY.png`, `CROP_PARITY.png`, `SLIDER_PARITY.png`, `OPTAGG_PARITY.png`. Blue is published,
  orange is ours, grey hatched is a GAP.

**Scope.** This is PRMBench development data that has already been inspected. It is not untouched confirmation.
The PRM in every arm was trained with labels, so "label-free" describes only our fusion step.
