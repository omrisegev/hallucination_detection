# Negative result — neighbor-aware readout of the fused step score (neighbor_readout_v1)

**Hypothesis (one sentence):** Errors come in runs, so a step next to a flagged step should also be flagged. A
label-free readout that uses this adjacency should therefore beat GAP. It could do so in which steps are flagged
(P1), in how many steps are flagged (P2), or in how the neighbor weight is chosen without labels (P3).

**Date / step:** 2026-11-01, Step 485

**Benchmark / population:**
- **Primary:** PRMBench, non-control answers, using the official PRMScore pool (`evaluate_collection.load_prmbench`,
  labels from raw 1-based `error_steps`). 6,969 answers and 94,203 steps are scored.
- **Transfer:** Socratic-PRMBench, 2,995 answers.
- **Descriptive only:** Hard2Verify, 181 answers.
- **PRMs:** 4 (Qwen-7B, Qwen-72B, PURE, RLHFlow).
- **Arms:** 2 fused arms. CHAIN_N16 is primary; TELE_N16 is secondary.
- **Bootstrap:** paired, 2,000 draws, by source group on PRMBench and by answer on the external sets.
- **Folds:** none. The fits are label-free and pooled per cell.
- **ProcessBench:** not used. Its labels mark only the first error, and it is out of focus.

## Arms compared

The columns differ from the template. The template's "PB macro-F1" column is replaced by Socratic PRMScore. The values
are CHAIN_N16 for Qwen-7B / Qwen-72B / PURE / RLHFlow. Sources: `RESULTS.csv` for the scores and `CONTRASTS.csv` for the
deltas.

| arm | what it is | PRMB within-AUC | PRMB PRMScore | Socratic PRMScore (x100) | vs reference |
|---|---|---|---|---|---|
| GAP (reference, incumbent) | top-k by score, k at the largest gap (cap ceil(n/2)) | .838 / .843 / .816 / .775 | 70.40 / 71.09 / 69.51 / 66.25 | 66.89 / 57.43 / 64.99 / 57.06 | — |
| K2 (construction-prior reference) | top 2 per answer | (same ranking as GAP) | 70.80 / 71.90 / 69.60 / 67.43 | 65.97 / 57.11 / 64.27 / 57.25 | +.40 / +.81 / +.09 / +1.18 on PRMB |
| ORACLE (label ceiling) | top-k with the true count | (same ranking as GAP) | 75.40 / 76.37 / 73.73 / 70.70 | 73.70 / 63.93 / 70.51 / 62.48 | ceiling |
| H1-NOPOS a (simple control: no position term) | HMM posterior ranking, b = 0, count k_GAP | — | official delta vs GAP −.225 / .000 / +.009 / −.199 | — | 2 significant losses, 2 ties |
| **H1a (candidate, primary)** | 2-state HMM with position entry; posterior ranking, count k_GAP | .840 / .859 / .835 / .764 | 70.16 / 71.23 / 69.54 / 66.00 | falls back to GAP in 3/4; PURE 64.98 | −.242 [−.397, −.098] / +.142 [+.043, +.234] / +.031 [−.058, +.124] / −.247 [−.372, −.135] |
| **H1b (candidate, count)** | the same HMM; flags p > .5, so the model sets the count | .840 / .859 / .835 / .764 | 59.20 / 55.23 / 63.29 / 50.42 | PURE 58.64; others fall back | −11.2 / −15.9 / −6.2 / −15.8, all CIs exclude 0 |
| V1 (candidate) | SYM(w), w from the variogram Wiener weight; count k_GAP | — | 69.75 / 68.07 / 68.07 / 65.56 | falls back in 2/4 | −.649 / −3.019 / −1.438 / −.692, all CIs exclude 0 |
| G1 (candidate) | Laplacian smoothing, λ by pooled GCV (λ = 2.0 in all 4) | — | 67.50 / 68.93 / 66.79 / 62.06 | 66.70 / 57.40 / 62.83 / 55.29 | −2.903 / −2.159 / −2.716 / −4.189, all CIs exclude 0 |

**Win/loss record.**
- **H1a vs GAP**, on PRMBench PRMScore (the primary endpoint, CHAIN_N16): 1 win (Qwen-72B), 2 losses (Qwen-7B, RLHFlow)
  and 1 inconclusive (PURE). The paired CIs above include 0 only for PURE.
- **Pre-registered rules.** P1, P2 and P3 fail on both arms (`VERDICT.json`). Transfer (T) was not triggered.
- **Count.** H1b's count error is 4.12 / 5.00 / 2.94 / 5.17 against GAP's 1.42 / 1.39 / 1.34 / 1.48 (`RESULTS.csv`,
  `count_mae`).

## Three plausible reasons it failed

1. **At a fixed count, the neighbor signal sits below the flag line.**
   - The stage-1 curves (`CURVES.csv`, CHAIN_N16 PRMBench) show the ranking does improve. LIFT's peak within-AUC is
     .850 / .859 / .833 / .785, against .838 / .843 / .816 / .775.
   - Yet the best PRMScore at count k_GAP gains only +.06 / +.08 / +.07 / +.01 over GAP.
   - GAP flags 2.23–2.45 steps per answer (`RESULTS.csv`, `mean_flags`). The reordering happens among the steps it already leaves unflagged, and
     PRMScore depends only on the flagged set. This is why P3's ratio has no defined value in most cells: there is no
     headroom to share.
2. **The label-free HMM finds a "high-score half" state, not the error state.**
   - The implied share of the error state is .474 / .542 / .400 / .552 (`fits/*__CHAIN_N16.json`, `implied_share`).
   - The true step error rate in the PRMBench metric pool is .158. Source: `evaluate_collection.load_prmbench`,
     Qwen-7B, `err[keep].mean()`, computed during this evaluation.
   - Persistence is overstated: ρ = .65–.77 (`fits/prmbench__*CHAIN*.json`), against an observed run persistence of about .42.
   - H1b therefore flags about 2.5 times too many steps. This is the declared risk N1: Gaussian emissions on answer-z
     scores accept a spurious second class, which the protocol's synthetic test showed on pure noise.
3. **The two moment and GCV weight rules both over-smooth.**
   - The variogram λ is .51–.72 on PRMBench CHAIN (`fits/*.json`, `V1.lam`). It gives w = .51–1.11, beyond the best
     stage-1 SYM weight of about .0–.3.
   - GCV picks the grid edge, λ = 2.0, in every PRMBench cell. Answer-z scores of short answers look smooth to GCV, so
     it chooses heavy smoothing.
   - Both were declared risks (protocol section 6).

## Verdict
- [ ] closes the DIRECTION (the idea cannot work under this access/contract)
- [x] closes this IMPLEMENTATION only (fixed-count re-ranking; a Gaussian 2-state HMM on answer-z; variogram/GCV
  weights)

The adjacency signal is real: within-AUC rises in 3/4 PRMBench PRMs at the stage-1 peak. It can only change PRMScore by
changing the COUNT. SYM_RECOUNT, a label-using diagnostic, reaches 70.72 / 71.66 / 69.82 / 66.25 against GAP's
70.40 / 71.09 / 69.51 / 66.25 (`CURVES.csv`). This study's count candidate (H1b) was not a usable way to change it.

## What would reopen it

A label-free count rule that uses adjacency and does not depend on a Gaussian 2-state fit to answer-z scores. One
example is GAP recomputed on a neighbor-lifted score, with the weight fixed in advance or chosen without labels. It
would be pre-registered as a new candidate and judged on Socratic, since PRMBench is now exposed. The SYM_RECOUNT
curve above is development evidence that this headroom exists. It is not a result.

**Source files:**
- `RESULTS.csv`, `CONTRASTS.csv`, `CURVES.csv` and `VERDICT.json`;
- `fits/*.json` and `fits/*.npz`;
- `PROTOCOL.json` (frozen hashes);
- `independent_check/PARITY.json`.

**Commands:** `python run.py fit 8` then `python run.py evaluate`.

**Coverage:**
- 24 / 24 cells fitted and evaluated.
- H1 is valid, with no fallback, in 7 / 8 PRMBench cells and 3 / 8 Socratic cells (`fits/*.json`).
- The independent re-implementation matches H1 / H1-NOPOS flags exactly in the one cell checked: 94,203 / 94,203
  steps.
- No README or REPORT.html existed in this directory when this file was written.

## Addendum (2026-11-01, POST HOC): the in-scope arms are AUTO, SUB5 and TELE16

**Correction from Omri.** CHAIN_N16 is out of scope, because it forces the PRM into its own group. The in-scope arms are
AUTO, SUB5 and TELE16.

**Coverage.**
- TELE16 (TELE_N16) was in the frozen run.
- AUTO and SUB5 (backbone telemetry) were added afterward by `arms_extension.py`. It reruns the frozen `run.py`, after
  the hash check, on the new scores, with no change to any parameter or rule.
- Parity of GAP on those scores with the stored references is exact on PRMBench and Socratic. Hard2Verify uses 181
  answers here, against 177 in the references.
- Standalone telemetry is not on this machine and was not run.

**Results** (`EXT_VERDICT.json`, `VERDICT.json`, `REPORT.html`). P1, P2 and P3 fail on AUTO, SUB5 and TELE16 alike.
Counts are taken per PRM, with nothing averaged. ↑ is a significant gain, ↓ a significant loss, ≈ inconclusive, and
↺ a fallback to GAP.

| arm | H1a on PRMBench | H1b on PRMBench |
|---|---|---|
| AUTO | PURE +.08 ↑; RLHFlow −.21 ↓; Qwen-7B ≈; Qwen-72B ↺ | 3 ↓, 1 ↺ |
| SUB5 | Qwen-7B +.11 ↑; PURE +.12 ↑; Qwen-72B ↺; RLHFlow ↺ | 2 ↓, 2 ↺ |
| TELE16 | RLHFlow −.07 ↓; 2 ≈; 1 ↺ | 3 ↓, 1 ↺ |

**One partial positive.**
- **What it is.** SUB5 H1b gains outside PRMBench in three cells:
  - Socratic Qwen-7B: +1.49 [+.99, +1.98]; the count error falls from 1.32 to 1.15.
  - Hard2Verify Qwen-7B: +5.2.
  - Hard2Verify PURE: +14.0. Here the count error rises, from 3.18 to 3.57.
- **What goes against it.** The same arm loses 19–27 points on Socratic with PURE and RLHFlow.
- **Likely reason.** These sets average 3.0–3.9 true errors per answer, against 2.1 on PRMBench. A count of about
  half the steps is therefore closer there by accident.
- **Status.** POST HOC, single cells, not a method.

**New failure mode.** In every Socratic cell where H1a collapses by 10 to 25 points, the fit has b = −8 to −11 and a
μ gap of .2 to .4 (`fits*/socratic__*.json`). The HMM learned "beginning of the answer" instead of "error". This is the
nuisance-state risk declared in protocol section 6.

**Verdict unchanged.** This closes the implementation, not the direction.
