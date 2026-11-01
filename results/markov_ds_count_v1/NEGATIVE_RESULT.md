# Negative result — Markov Dawid-Skene error count (exact, multi-voter)

**Hypothesis (one sentence):** A Dawid-Skene model with the arm's L-SML groups as voters, Markov error labels and
an exact likelihood for answer-normalized inputs can estimate how many steps of an answer are wrong better than GAP
and better than a constant share. Everything is label-free.
**Date / step:** 2026-11-01, Step 486
**Benchmark / population:**
- **Judging:** Socratic, all 2,995 answers (2,991 enumerated exactly, n ≤ 16).
- **Descriptive:** Hard2Verify (181 answers) and PRMBench (6,969 answers; exact for n ≤ 12, 60%).
- **Bootstrap:** source-group, from `E.boot_weights`.
- **Grid:** 4 PRMs × 3 arms (AUTO, SUB5, TELE16).

## Arms compared (Socratic, count MAE; lower is better; `RESULTS.csv`)
| arm / PRM | GAP | K2 (constant 2) | MDS (candidate) | SHARE (control) | IND | MDS valid | ŝ vs true .342 (`MECHANISM.csv`) |
|---|---|---|---|---|---|---|---|
| AUTO / Qwen-7B | 1.537 | 1.236 | 1.637 | 1.883 | = GAP (invalid) | yes | .588 |
| AUTO / Qwen-72B | 1.639 | 1.236 | = GAP | — | — | **no** (ŝ .711) | — |
| AUTO / PURE | 1.594 | 1.236 | 1.476 | **1.209** | 1.478 | yes | .237 |
| AUTO / RLHFlow | 1.636 | 1.236 | = GAP | — | — | **no** (ŝ .898) | — |
| SUB5 / Qwen-7B | 1.318 | 1.236 | 1.420 | 1.883 | 1.475 | yes | .508 |
| SUB5 / Qwen-72B | 1.603 | 1.236 | = GAP | — | 1.490 | **no** (ŝ .912) | — |
| SUB5 / PURE | 1.496 | 1.236 | 1.675 | 1.883 | = GAP (invalid) | yes | .580 |
| SUB5 / RLHFlow | 1.560 | 1.236 | = GAP | — | 1.470 | **no** (ŝ .874) | — |
| TELE16 / Qwen-7B | 1.556 | 1.236 | 1.327 | **1.152** | 1.324 | yes | .280 |
| TELE16 / Qwen-72B | 1.636 | 1.236 | = GAP | — | — | **no** (ŝ .887) | — |
| TELE16 / PURE | 1.596 | 1.236 | 1.479 | **1.209** | 1.477 | yes | .237 |
| TELE16 / RLHFlow | 1.621 | 1.236 | = GAP | — | — | **no** (ŝ .900) | — |

**Win/loss record (E1, `VERDICT.json`).** E1 needs ≥ 3 of 4 Socratic PRMs per arm. It passes in **0 of 4** PRMs in
every arm.
- **Valid fits:** MDS is valid in only 2 of 4 PRMs per arm (Qwen-7B and PURE).
- **Against GAP (count MAE CI high < 0):** MDS is better in 3 of the 6 valid cells:
  - TELE16 Qwen-7B: −.229 [−.28, −.18];
  - TELE16 PURE: −.116 [−.16, −.07];
  - AUTO PURE: −.118 [−.17, −.07].
- **Against SHARE in those same 3 cells, MDS is worse:** +.175 [+.14, +.21], +.271 [+.24, +.30], +.267 [+.23, +.30].
  So the gain over GAP is a prevalence effect, not answer-specific counting.
- **Where MDS beats SHARE** (AUTO Qwen-7B, SUB5 Qwen-7B and PURE), the reason is that SHARE inherits MDS's
  overcounted share (.51–.59). In those cells MDS is worse than GAP (+.10 to +.18).
- **Against the constant K2, MDS is worse in all 6 valid Socratic cells:** +.091 to +.440, every CI above 0
  (`CONTRASTS.csv`, MDS − K2).

**Mechanism controls.**
- **E2 (Markov versus independent):** fails. Where both fits are valid, the difference is ±.003, with CIs that include
  0.
- **E3 (several voters versus one):** not testable as declared. SINGLE is invalid in 4 of the 6 valid-MDS Socratic
  cells, so "MDS − SINGLE" there equals "MDS − GAP".

## Partial success: the count raises the official PRMScore above every label-free count rule in 3 cells
The flags are always the arm's own top k̂, so any PRMScore difference comes from the count alone. In 3 cells, MDS
beats GAP, the constant K2 and SHARE on PRMScore, with every 95% CI above 0 (`CONTRASTS.csv`, metric `official`;
Socratic on the ×100 scale):

| cell | MDS | − GAP | − K2 | − SHARE | ORACLE |
|---|---|---|---|---|---|
| Socratic · TELE16 · Qwen-7B | 65.07 | +0.61 [+0.07, +1.15] | +1.44 [+0.93, +1.95] | +0.98 [+0.49, +1.48] | 70.33 |
| PRMBench · AUTO · PURE | 69.70 | +0.64 [+0.43, +0.86] | +0.68 [+0.34, +1.03] | +6.57 [+6.05, +7.10] | 72.84 |
| PRMBench · TELE16 · PURE | 69.38 | +0.67 [+0.44, +0.89] | +0.73 [+0.39, +1.08] | +6.32 [+5.82, +6.84] | 72.49 |

**Hard2Verify, RLHFlow.** MDS also beats GAP (+7.53 and +8.61) and K2 (+7.24 and +7.78) in AUTO and TELE16, but
SHARE is better there (−3.46 and −3.52).

**Model-level successes.**
- **Synthetic data:** the exact model is identifiable and counts best (MAE 0.89, against GAP 1.17 and SHARE 1.03).
- **Voter quality:** on real data, δ tracks voter quality without labels.

**Limits.**
- The pre-registered primary endpoint (count MAE on Socratic) is not met.
- PRMScore was a declared metric, but these cells were singled out after the results were read.
- Socratic is development data.

## Three plausible reasons it failed
1. **Agreement identifies voter quality, not error quantity.** Where the fit is valid, the fitted separations δ_g are
   ordered like the voters' within-AUC: Spearman .6–1.0 in 5 of the 6 valid Socratic cells, and .10 in SUB5 PURE,
   from `MECHANISM.csv`. So the
   model does find which voters are good. But the share it implies is either a constant that SHARE already reproduces
   (.237–.280 against a true .342) or an overcount (.51–.59). This is the thesis claim from the adversarial review.
2. **Label switching / the "upper half" solution.** On Qwen-72B and RLHFlow the fit moves to e ≈ .93–.99 and
   ŝ ≈ .87–.91 (`MECHANISM.csv`). The model then explains the data as "almost every step is an error and the few
   clean ones stand out". With weak voters (RLHFlow voter within-AUC .42–.61), the invariant likelihood cannot tell
   the two orientations apart. The validity bound stops these fits from counting, so the cell falls back to GAP.
3. **Conditional independence between groups does not hold in the real data.** The synthetic position-trend control
   (`tests/test_markov_ds_count.py`) produced exactly this kind of degenerate fit: e → 0, ρ → 1, share .53. On real
   Socratic, the correlation between the posterior and position ranges from −.38 to +.28 across cells. A shared
   nuisance across groups (position, answer style) breaks the DS assumption that the count needs. Ranking survives it
   better than counting does.

## Verdict
- [ ] closes the DIRECTION (the idea cannot work under this access/contract)
- [x] closes this IMPLEMENTATION only (roster, lambda, readout, bank); the idea stays open

The exact model, pooled per cell with the L-SML groups as voters and no anchor, does not count better than a constant
share or K2 on Socratic. The model is identifiable on synthetic data (MAE 0.89 vs GAP 1.17 and SHARE 1.03), so this
evidence is against the real-data assumptions (conditional independence, a shared δ, one orientation). It is not
evidence against the statistics.

## What would reopen it
- **A count anchor that is not the agreement itself:** for example, a calibrated clean-answer gate from part 1, or a
  prior on the count fixed outside the benchmark.
- **Voters whose errors are demonstrably conditionally independent given the label:** for example, a residual check
  per group pair, before fitting.

## Post hoc observation (descriptive, not a result)
In the 4 valid AUTO/TELE16 Socratic cells, the exact posterior marginal P(y_t = 1) ranks errors within an answer
slightly better than the arm's fused score: +.047, +.022, +.024, +.013 within-AUC. In SUB5 it is ±.007. On PRMBench
it is mostly worse: −.02 to −.06 in the valid cells. Source: `POSTHOC_POSTERIOR_WITHIN_AUC.json`. This was computed
after the labels were read and is not a pre-registered endpoint.

**Source files:**
- **Results:** `RESULTS.csv`, `CONTRASTS.csv`, `MECHANISM.csv`, `VERDICT.json`, `REPORT.html`.
- **Fits:** `fits/` (36 of 36 cells; the commands are `run.py fit 9` and `run.py evaluate`).
- **Integrity:** `PROTOCOL.json` (hashes frozen before any label).
- **Independent check:** `independent_check/PARITY.json`. 2,991 of 2,991 Socratic Qwen-7B TELE16 counts match a
  separate implementation; the largest marginal difference is 9.5e-12.
- **Pre-label stop:** `prelabel_synthetic/PRELABEL_STOP.md`.
