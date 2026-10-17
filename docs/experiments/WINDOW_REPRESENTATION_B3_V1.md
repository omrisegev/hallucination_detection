# Window representation B3 v1 — measurement first, gated answer-local fusion (item 4) — PRE-REGISTRATION

Written and committed 2026-09-23 before any number was computed. Branch `claude/lsml-ct7-levers-v1`.
Module `spectral_utils/window_moment_bank.py`; stage 1 `scripts/diagnostics/window_pr_measurement_v1.py`;
stage 2 `scripts/experiments/window_answer_local_fusion_v1.py`; config `configs/window_representation_b3_v1.json`.
Authorized by Omri on 2026-09-23 as "measurement plus gated fusion". Development only; nothing is promoted.

## Question (atlas handoff B3, `docs/HANDOFF_localization_2026-09-17_evening.md`)

L-SML has never been tested where it is defined. At step level the banks carry 1.80-2.46 conditionally
independent views (labels for measurement only), an answer has a median of eight steps, and every fitted
rule equals averaging. On 8-token windows an answer has a median of about 49 fit rows and level / sd /
slope are different functionals of a stream, not smoothings of one series. The handoff's "3.4-3.9
effective views" were MARGINAL within-answer participation ranks from small pilots (3.64 on the
58-answer moment pilot; 3.651 for the width-32 STFT bank), not the within-label conditional PR behind
1.80 / 2.46 / 2.83; this is the first conditional measurement of a moment bank, and nothing is assumed
from those figures. The original moment-bank code did not survive on any branch; the bank is rebuilt.

## Bank (declared)

Base token streams: `q15_H1`, `q15_VE1`, `logprob_margin`, `true_tail50`, `energy_innovation` from the
eleven-channel `TOKEN_MATRICES.npz`, and `bocpd_residual`, `chosen_std_excess` (step-0 rule applied)
from item 3's `CT7_TOKEN_MATRICES.npz`. Views, one per measured evidence family, with sd and slope on
the two entropy-family representatives of the eleven-bank (not chosen by per-view AUC):
`q15_H1__{level,sd,slope}`, `q15_VE1__{level,sd}`, `chosen_std_excess__level`, `logprob_margin__level`,
`true_tail50__level`, `energy_innovation__level`, `bocpd_residual__level` (p = 10). Width 8; dense
scoring windows at stride 1 plus a full-width window anchored at the end; the non-overlapping grid
(every 8 tokens) is the fit set. The 33-view (11 x 3) ladder is not run.

## Stage 1 — measurement (labels for measurement only)

1. Anchor: the conditional PR of the seven frozen CT7 profiles on PRMBench steps must be within 0.01 of
   1.80, else the run stops.
2. Conditional PR (within-label centring, correlation eigenvalues, (Σλ)²/Σλ², the
   `stage_b_2x2_v1.conditional_participation_ratio` recipe) of the bank at window level (PRMBench
   windows lying inside one labelled step; straddlers dropped and counted) and at step level after
   two readouts of the window scores mapped to tokens (`windows_to_tokens`): Top10 over the step's
   tokens (primary) and the overlap mean (`tokens_to_official_steps`); step views answer-standardized.
3. Shuffled reference: the same pipeline on tokens permuted within each answer independently per
   channel (seed 20260918). The permutation makes the channels independent, so its PR sits near p;
   it is an UPPER reference. If it does not sit above the real value the contrast is uninformative.
4. Effective sample size of each view's fit-window series (n_eff / n, median over answers with at least
   12 fit windows), and one marginal within-answer participation rank (SVD of the standardized fit
   windows, median) to bridge the old 3.6-type figures.

**Gate**: real step-level PR (Top10 readout) >= 3.0 AND shuffled reference >= real + 0.5. Only then
stage 2 runs. Otherwise the report ends with the measurement and the statement "the window
representation does not clear L-SML's domain on this population".

Predictions, written before scoring: window-level PR above step-level PR; step-level PR of the ten-view
bank between 2 and 3; n_eff/n of the level views well below 1 (adjacent windows are dependent);
the shuffled reference near 10 at every level.

## Stage 2 — answer-local fusion (only if gated in)

Per answer, on its own non-overlapping fit windows only (never the dense rows), standardized by the
fit windows' mean/sd: `equal`; IU-PCR (`upcr.upcr_fit`, `laplacian_upcr.IU_FIT_DEFAULTS`,
abstentions flagged and counted); shrinkage-IU (joint target from `shrinkage_iu`, Ledoit-Wolf alpha
scaled by n / n_eff with n_eff the smallest per-view effective sample size, clipped to 1; a label-free
rule, not an optimum; block/diag targets deferred); continuous L-SML (residual K, guard on, groups
discovered). Minimum fit windows 3p = 30 (240 tokens); below that, and on any failed fit, the answer
falls back to equal and is counted. Weights are oriented to correlate positively with the equal-weight
dense score (label-free). Scores: dense windows → tokens → official steps by Top10 (primary) and by
overlap mean; answer-standardized; argmax; frozen CT7 gate for the common-gate F1. Every learned arm
is reported twice: full coverage with fallback, and `*_native` (valid only where the fit was native).
Diagnostics per arm: distance of the weights from equal (cosine and L2 to 1/p), IPR, alpha, K counts,
abstentions, fallbacks, native coverage, runtime.

Contrasts, one Holm family: primary `window_lsml_top10 − window_equal_top10`; IU and shrink-IU minus
equal; the native-row versions; the overlap-mean versions; `shrink_iu − iu`; each window arm minus
`ct7`; `top10 − overlap_mean` at equal weight; depth-stratum SLA intervals.

Prediction: PRMBench answers (median step 24 tokens) fall back in a large fraction; on ProcessBench
the learned arms are within 1 pp of equal on SLA, and the whole window family is below CT7.

## Decision language

Development rows. If the gate fails, the finding is that the answer-only L-SML objective stays out of
reach on this representation with this bank. If it passes and a learned arm beats equal with an
interval excluding zero, that is recorded as the first configuration in L-SML's domain on this
population; it is not a candidate until frozen and confirmed on untouched data.

## Execution

```
python -B scripts/diagnostics/window_pr_measurement_v1.py --config configs/window_representation_b3_v1.json   # minutes
python -B scripts/experiments/window_answer_local_fusion_v1.py --config configs/window_representation_b3_v1.json  # only if gate_passed
python -B scripts/diagnostics/window_pr_measurement_v1.py --dry-run; python -B scripts/experiments/window_answer_local_fusion_v1.py --dry-run --force
```

Outputs: `results/window_representation_b3_v1/{RUN_FREEZE.json, WINDOW_PR.json}` and, if gated in,
`results/window_representation_b3_v1/fusion/{RUN_FREEZE.json, RESULTS.json, SUMMARY.csv, UNCERTAINTY.json, ...}`.
