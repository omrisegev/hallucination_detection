# Cross-model answer gate: go/no-go check (2026-11-14)

**Verdict: NO-GO.** The answer statistics of different base models are strongly dependent when the label is fixed.
Dawid-Skene cannot identify the error prevalence from them. Data: development data, labels used only for this diagnostic.

## Question

The review of the two-head answer gate (thread "Review two-head answer gate architecture") proposed one answer
statistic per base model, then Dawid-Skene (DS) across 3 base models to estimate the error prevalence pi without labels.
DS needs the 3 statistics to be approximately independent when the label is fixed. This check tests that on existing
standalone telemetry. CPU only. No new inference.

## Definitions

| Quantity | Definition |
|---|---|
| Base models | 7B-Inst = Qwen2.5-Math-7B-Instruct, 72B-Inst = Qwen2.5-Math-72B-Instruct, 7B-base = Qwen2.5-Math-7B, Llama-8B = Llama-3.1-8B-Instruct (standalone pairing of Qwen-PRM-7B, Qwen-PRM-72B, PURE, RLHFlow). The PRM score is not used. |
| Step score s | 16 raw TELE16 channels -> global rank -> N(0,1) over all steps of the benchmark (no per-answer z) -> L-SML fusion, anchor = channel mean (same as GSP-T). Label-free. |
| Step p-value p_i | Share of all steps of the benchmark (same model) with score >= s_i. |
| Simes | T_simes = -log10( min_i n p_(i) / i ) over the n steps of one answer. Corrects for the number of steps. |
| HC | Higher-Criticism: max_i sqrt(n) (i/n - p_(i)) / sqrt(p_(i)(1 - p_(i))). |
| A_mean | Mean of s over the steps of one answer. Used only for the within-model reference. |
| y | 1 if any step of the answer has an error label. |
| Length-stratified AUROC | AUROC of the statistic for y inside 5 bins of n_steps, mean weighted by bin size. |
| Conditional abs corr | abs(Spearman) between two statistics inside y = 0 and inside y = 1, mean weighted by class size. |
| DS pi-hat | Vote of model m = 1 if T_simes_m > its own q-quantile (label-free). 2-class DS EM on 3 votes, 50 restarts. Error class = class with the higher vote rate. Main q = .5; q = .3 and .7 for sensitivity. 90% bootstrap (300). |
| Main triplet | 7B-Inst, 72B-Inst, Llama-8B. Alt triplet: 7B-base in place of 7B-Inst. |

Go rule, fixed before the run: all ProcessBench cross-model conditional abs corr of the main triplet < .20
("clearly below .26") AND ProcessBench abs(pi-hat - pi) <= .10 (main triplet, q = .5).

## Results

Gate of the extraction: answer_z(G16) equals the stored R1 columns (max abs diff 0.0) in 16 of 16 cells that have
stored columns. The 8 CRV cells have no stored R1 columns (no gate). 24 of 24 cells extracted. Labels agree across the
4 models in all cells.

### (1) Answer-level AUROC of T_simes (length-stratified)

| Benchmark | answers | pi | length only | 7B-Inst | 72B-Inst | 7B-base | Llama-8B |
|---|---|---|---|---|---|---|---|
| ProcessBench | 3,400 | .653 | .588 | .785 | .756 | .762 | .750 |
| PRMBench | 6,969 | .866 | .472 | .760 | .746 | .738 | .770 |
| Socratic | 2,995 | .998 | .729 | .341 | .360 | .301 | .447 |
| Hard2Verify | 177 | .768 | .606 | .509 | .525 | .528 | .493 |
| CRV-Arith | 1,500 | .083 | .478 | .759 | .667 | .545 | .686 |
| CRV-GSM8K | 1,319 | .210 | .624 | .747 | .721 | .694 | .717 |

Socratic has 6 clean answers: its AUROC is not informative. All values in AUROC.csv (also HC and A_mean).

### (2) Conditional abs corr given y

| Benchmark | cross-model, main triplet pairs | 7B-Inst vs 7B-base | within-model A_mean vs Simes |
|---|---|---|---|
| ProcessBench | .60 / .60 / .88 | .83 | .87 - .93 |
| PRMBench | .81 / .84 / .86 | .81 | .70 - .73 |
| Socratic | .65 / .75 / .81 | .72 | .75 - .79 |
| Hard2Verify | .95 / .95 / .96 | .97 | .95 - .96 |
| CRV-Arith | .74 / .76 / .77 | .74 | .56 - .81 |
| CRV-GSM8K | .83 / .85 / .90 | .91 | .84 - .88 |

Inside length bins the values drop by at most .12 (COND_CORR.csv, column cond_abs_corr_len).

### (3) DS pi-hat vs true pi (main triplet)

| Benchmark | pi | q = .3 | q = .5 | q = .7 |
|---|---|---|---|---|
| ProcessBench | .653 | .693 | .478 | .286 |
| PRMBench | .866 | .701 | .497 | .306 |
| Socratic | .998 | .661 | .481 | .309 |
| Hard2Verify | .768 | .687 | .504 | .311 |
| CRV-Arith | .083 | .677 | .489 | .310 |
| CRV-GSM8K | .210 | .708 | .500 | .304 |

pi-hat is approximately 1 - q in all 36 fits (2 triplets x 3 q x 6 benchmarks). The latent class that DS finds is
"the answer is high on all models", not "the answer has an error". The ProcessBench q = .3 value (.693, error .04)
is a coincidence of 1 - q = .7 and pi = .65; it is not an estimate.

## Interpretation

- The base models read the same answer text. The step score responds to properties of the text (for example hard or
  number-dense steps) in the same way for each model. This shared component is much larger than the label signal.
- The dependence between models is as high as, or higher than, the dependence between two views of one model.
  Different model families (Llama vs Qwen) lower it only on ProcessBench (.60), not enough.
- 7B-base vs 7B-Inst: .72 - .97. The grouping guess in the review is correct, but it does not matter, because all pairs are dependent.
- Single-model answer signal exists (ProcessBench .75 - .79, PRMBench .74 - .77, above length only). The problem is
  identification of pi and of the threshold, not ranking. This is the same pattern as E1, raw-null, mixture and CP EM.

## Files

| File | Content |
|---|---|
| extract.py | Stage 1 (PC): step score s per base model, R1 gate. Run with HD_ROOT = a checkout with results/c4u_telemetry_only_v1. |
| analyze.py | Stage 2 (any machine): answer statistics, (1)-(3), VERDICT.json |
| figures.py | fig1_auroc.png, fig2_cond_corr.png, fig3_ds_pi.png |
| steps/*.csv.gz | 24 cells: uid, step, y, s |
| EXTRACT_GATE.json, extract.log | Extraction gate and log (PC run, 2026-11-14, Python 3.13.3, HD_ROOT = wt_c4u_teleonly) |
| AUROC.csv, COND_CORR.csv, DS_PI.csv, VERDICT.json | Results |
