# Review of `codex/cumulative-vote-fusion-v2` (Codex, 2026-09-22)

Claude, 2026-09-22, on `claude/token-axis-fusion-sampling-3i9r2u`. Reviewed commits
`0c7d6826` (base snapshot), `da7be4da` (eleven-channel run), `bcf5a4bd` (CT7-profile run).
Read: both protocols, both session reports, HISTORY/PROGRESS additions, `scripts/experiments/
cvf_v2/{core,em,readout,ranking,scoring,ct7,ct7_evaluation,data,runner}.py`, `SUMMARY.csv`,
`PB_METRICS.json`, `PAIRED_CONTRASTS.csv`, `READOUT_SELECTIONS.csv`, `VALIDATION.json`,
`RUN_FREEZE.json`, `PROFILE_VALIDATION.json`, `MINDGAP_REPLAY_MANIFEST.json`. Ran the branch's
unit tests here (10 passed) and one synthetic probe of the tie rule (below). The large inputs
(`TOKEN_MATRICES.npz`, `CT7_DEV_SCORES.npz`, `JOINED.npz`) are not in git, so no number was
recomputed; consistency between the committed files was checked instead.

## 1. What Codex ran

A re-implementation of the Step 423/424 design as a frozen, full-population protocol on the
Qwen cells: 6,800 ProcessBench answers (4,442 erroneous) and 6,969 PRMBench answers, v3 labels,
source-group folds v2, per-scorer fits, nested inner folds for PRMScore thresholds, 10,000 paired
source bootstraps with one Holm family. Nine arms: binary equal / spectral / binary L-SML /
continuous-L-SML bridge / Dawid-Skene / hierarchical EM, and soft equal / spectral / continuous
L-SML (τ = 1). Three rosters: fixed top5, training-selected readout per channel, and a
within-answer token shuffle control; plus an error-only training control, single channels,
longest-step and relative-position controls, CT7, the Step 422 token fusion, and a float64
Mind-the-Gap adapter replay. A second run replaces the eleven channels by CT7's seven frozen step
profiles.

Headline (gate-free PB SLA, macro over eight cells / common-CT7-gate F1 / PRMB within-AUROC):

| arm | SLA | F1 | within-AUC |
|---|---:|---:|---:|
| CT7 (incumbent) | 39.89 | 41.19 | .7724 |
| Step 422 token L-SML | 35.92 | 37.32 | .7532 |
| eleven, selected readouts, soft continuous L-SML | 36.18 | 38.31 | .7615 |
| eleven, selected readouts, soft equal | 36.03 | 38.13 | .7605 |
| eleven, top5, soft equal | 32.49 | 35.24 | .7532 |
| eleven, top5, binary DS | 32.98 | 34.93 | .7556 |
| eleven, top5, binary equal | 21.25 | 25.94 | .7350 |
| CT7 profiles, soft continuous L-SML | 40.00 | 41.26 | .7686 |
| CT7 profiles, soft equal | 39.83 | 41.09 | .7724 |
| CT7 profiles, binary DS | 34.63 | 36.43 | .7456 |
| longest-step control | 31.78 | 35.14 | .6181 |

Codex's own reading: no arm beats CT7; on the eleven-channel bank learned fusion adds +0.15 pp
over soft equal (CI includes zero); on CT7 profiles soft L-SML is +0.11 pp [−0.60, +0.83] on SLA
and −0.0038 [−0.0049, −0.0027] on within-AUC. No promotion. I agree with that reading.

## 2. Implementation: sound, with one inherited defect

- Fold discipline, freezing, anchors, nested thresholds, EM monotonicity checks, saved-model
  replays: all present and internally consistent; the ten tests pass here.
- Binary L-SML uses raw ±1 votes and the canonical Jaffe score matrix / residual from
  `fusion_utils`; spectral weights keep their sign (no clipping); orientation is endpoint-based
  and fit-only; PAVA and earliest-mode readout match the design.
- The Mind-the-Gap replay is the registered adapter (adjusted EMA span 5, worst negative flux
  per step) rebuilt from raw top-k in float64; its SLA on Qwen (22.25) is far below the paper's
  39.27, which strengthens Step 423's Llama finding that the common-protocol replay is not the
  paper's number.
- **Inherited defect: the earliest-wins tie rule turns discrete channels into step-0 voters.**
  `profiles()` takes `argmax` with no tie perturbation (the protocol says so explicitly; my
  Step 424 script has the same rule). On a synthetic answer with a channel taking values in
  {0, ½, 1}, the argmax lands on step 0 in 96% of draws (300 replicates, 8 steps of 25 tokens),
  because the top5 mean of every step saturates at the same value. On the real bank this is
  visible in `PB_METRICS.json` for the single channels: `chosen_surprisal` early 0.86 / late
  0.00, `top15_turnover` 0.86 / 0.00, `top50_js` 0.83 / 0.03, `dominant_freq16` 0.83 / 0.01 —
  numerically the profile of the "always step 0" control (0.87 / 0.00). Four or five of the
  eleven binary voters therefore always say "step 0".

## 3. What the defect explains

- **The binary-versus-soft gap (21.25 vs 32.49 for top5 equal, +11.2 pp [+9.4, +13.1]).** With
  four or five constant step-0 voters out of eleven, the median of positions is dragged early
  (binary equal: early 0.70, late 0.09). The soft encoding spreads tied mass over steps and has
  no such bias (early 0.31). So most of the "softness" gain is tie handling, not information
  preserved by not binarizing. The clean test is binary votes with tied maxima split (each tied
  step gets 1/k) or abstained; it needs no refit of anything else.
- **Why Dawid-Skene "fixes" binary (+11.3 pp over binary equal) but hurts on CT7 profiles
  (34.6 vs 37.8 for binary equal).** DS learns η ≈ low for the step-0 voters and mutes them; on
  CT7's seven informative profiles there is nothing to mute and the EM only adds variance. This
  is consistent with Step 423 (DS = SML when all voters are of one kind) and says EM earns its
  place only when the roster contains systematically biased voters.
- **Why learned soft weights lose to soft equal on the top5 roster (−6.5 pp [−8.0, −5.1]).** The
  soft instances are cumulative curves. An uninformative channel's curve is close to the uniform
  ramp, and all such channels' ramps are highly correlated with each other, so the leading
  eigenvector of the curve covariance loads on the junk channels: `top5 soft continuous L-SML`
  has early 0.59, the signature of the step-0 voters again. Fitting the weights on the jumps
  (the pmf) or on curves with the uniform ramp subtracted removes this; the selected-readout
  roster masks it (there the junk channels get CUSUM readouts and stop tying).

None of this changes the conclusion that nothing beats CT7, but it changes what the table
means: the eleven-channel binary rows are not a measurement of binarization, and the soft
learned rows are not a measurement of learned weighting.

## 4. Long chains, the original question

| arm | short cells | long cells | drop | late on long |
|---|---:|---:|---:|---:|
| CT7 | 42.1 | 37.7 | 4.4 | 0.34 |
| Step 422 token L-SML | 41.1 | 30.7 | 10.4 | 0.45 |
| eleven, selected, soft L-SML | 40.2 | 32.2 | 8.0 | 0.38 |
| eleven, top5, soft equal | 37.7 | 27.2 | 10.5 | 0.37 |
| longest-step control | 37.1 | 26.4 | 10.7 | 0.41 |

The cumulative-vote fusion does not close the long-chain gap; CT7 remains the most
length-robust arm by a wide margin. The one anti-late readout, `onset80`, was chosen on the
training folds for `energy_level` only and was excluded from PRMB, so "all channels read by
onset" was never a roster in this run. The shuffle control (29.7 on soft equal, above the
longest-step 31.8 minus 2) confirms Step 427's point that much of any PB SLA is step length.

## 5. Smaller points

- PRMB binary votes use "strictly above the within-answer median", i.e. a 50% positive rate per
  channel against a 16% error prevalence; the protocol disclaims the assumption but the encoding
  imposes it. Top-k per answer (k matched to prevalence) is the tighter choice.
- The CT7 BOCPD profile is recovered algebraically from `six_equal` (max difference 7.2e-15) and
  labelled as such; fine, and honestly recorded.
- Reproducibility from git alone is not possible: inputs are absolute Windows paths and the
  large arrays are local. Hashes and source snapshots are in place, which is the project norm.
- HISTORY: Codex's block has no Step number and sits on the master lineage; my Steps 423/424
  and the local Claude session's Steps 423–427 (`claude/token-probability-fusion-v1`) collide.
  Tag, never renumber, per CLAUDE.md.

## 6. Recommended next, in order

1. Re-run the binary arms with tie-split votes (cheap, no other change) and report the
   binary-versus-soft contrast again; that is the honest version of the softness question.
2. Fit the soft weights on jumps rather than cumulative curves and re-report learned versus
   equal on the top5 roster.
3. Run the fixed roster "every channel read by onset80" as one more arm on the eleven-channel
   bank; it is the only untested part of the anti-late hypothesis.
4. Drop `energy_level` (Step 425) or, better, run the bank at seven informative channels so the
   step-0 voters are out of the roster and the fusion question is asked on a clean bank.
