# CT7 family-equal v1 — partition, then equal within and across (item 2) — PRE-REGISTRATION

Written and committed 2026-09-23 before any score was computed. Branch `claude/lsml-ct7-levers-v1`.
Driver `scripts/experiments/ct7_family_equal_v1.py`, config `configs/ct7_family_equal_v1.json`,
module `spectral_utils/family_equal_readout.py`, shared plumbing `scripts/experiments/ct7_levers_common.py`.
Development only: the frozen 13,769-answer population (v3 labels, source folds v2, frozen CT7 gate).
Nothing is promoted; every arm is a development row.

## Question

CT7 averages its seven answer-standardized step views with weight 1/7, so the entropy-level family
(H0lim, ve0, ve0.75, ve1, all conditionally correlated .75-.97) holds 5/7 of the vote and the two
temporal views and the chosen-token view 1/7 each. The level views are late-biased (late .39-.42),
the temporal views early-biased (early .44 / .38); CT7's early .27 / late .34 balance is what equal
weight happens to produce. The Step 399 addendum found, on other rosters, that "mean within group +
equal weight per standardized group" carries the Joint result and that a cross-group eigen-solve
costs ~3 pp. Does giving each declared family one vote change CT7's localization, and in particular
its late misses on long chains?

## Fixed design

- Input: the frozen `profiles.npy` (145,597 x 7, sha `d564ba43…ff674` per `PROFILE_VALIDATION.json`),
  whose mean replays CT7 exactly (asserted); the frozen CT7 gate and scores (`CT7_DEV_SCORES.npz`,
  sha `9d10d2ff…b430`); CT7's macro-F1 .41188745848863717 and within-AUC .7723966352864217 replayed
  before any new row (the same asserts as `cvf_v2/data.py::prepare`).
- View order: H0lim, ve0, ve0.75, ve1, H0lim_prefix_innovation, bocpd_residual, chosen_token_z_despiked.
- Partitions, declared: **4/2/1** = A {0,1,2,3} distribution-shape level, B {4,5} temporal (prefix
  residual and BOCPD residual; the innovation is a prefix residual and behaves like BOCPD), C {6}
  provided-token evidence. Sensitivity **5/1/1** = A {0..4}, B {5}, C {6}.
- Arms (argmax per answer; the frozen gate for the common-gate F1):
  - `ct7` (mean of 7; replay) and `ct7_z` (the same after one within-answer standardization).
  - `fam421_raw`, `fam511_raw`: fixed weights 1/(K|g|) — "partition only", no re-standardization.
  - `fam421_answer`, `fam511_answer`: each family mean standardized within the answer (mean/sd, a
    constant family is zeroed), then the mean of the three. A fixed rule with an answer-adaptive
    scale: on a 2-step answer each family is ±1 and the fused score is a majority of three signs.
    Zeroed families and 2-step answers are counted.
  - `fam421_fold`, `fam511_fold`: family means divided by their training-fold sd (the ladder's
    `ceq` rule), five source folds. Pooled-donor access, label-free.
  - `fam421_eigen`, `fam511_eigen`: unguarded cross-family SML eigen-solve on the three family
    virtuals, fitted per fold, oriented to correlate positively with the equal mean (label-free).
    The comparator the method card says loses ~3 pp.
  - `fam_auto_fold`, `fam_auto_answer`: one label-free discovery route only — the residual-affinity
    (|C − vvᵀ|) spectral partition with fold-deletion stability (the Step 413 ladder route), K in
    {2, 3, 4}, minimum group size 1, selected by median ARI, then minimum ARI, then smaller K; the
    partition per fold is recorded; the same two scalings applied.
  - Controls: `six_equal` (views 0-5), `level_only` (A), `temporal_only` (B), `ct7_single__*`,
    `control__longest_step` (when `step_lengths.npy` is available).
- Every arm's scores are within-answer standardized before the pooled endpoints (PRMScore, pooled
  step AUROC), so that an answer-level scale difference cannot reorder steps across answers; the
  argmax and the within-answer AUROC are invariant (asserted).

## Endpoints and contrasts

- **Primary**: exact SLA on the 11+ stratum, `fam421_answer − ct7`, paired source-question
  bootstrap (10,000 draws, micro within the stratum because several cells have few 11+ erroneous
  answers). The frozen `cvf_v2.uncertainty.bootstrap` gives the three-endpoint intervals; the
  stratum intervals come from the same unit and seed in `ct7_levers_common.stratum_sla_intervals`.
- Secondary, one Holm family (`planned_contrasts_extra`): macro8 gate-free SLA, common-gate macro-F1,
  PRMB within-answer AUROC for every family arm minus `ct7`; `fam421_{answer,fold} − fam421_raw`;
  `fam421_* − fam511_*`; `fam_auto_* − fam421_*`; `fam421_eigen − fam421_answer`;
  `fam421_answer − six_equal`; `level_only − ct7`, `temporal_only − ct7`.
- Descriptive: early/late fractions by stratum and on the four long cells, tolerance-one, MAE,
  PRMScore (fit-free branch, on the standardized scores; secondary because it is scale-matched only
  by that standardization), coverage counts, runtime per stage.

## Predictions, written before scoring

1. `fam421_answer` lowers the late fraction on the 11+ stratum relative to CT7 and raises the early
   fraction; macro8 SLA within ±0.5 pp of CT7.
2. `fam421_raw` is closer to CT7 than `fam421_answer` (the partition alone moves less than the
   answer-adaptive scale).
3. `fam421_eigen` is below `fam421_answer` (the unguarded eigen-solve on three virtuals).
4. The discovered partition does not reproduce 4/2/1; several K are perfectly stable and the
   tie-break picks K = 2 with the level family against the rest (the route's documented behaviour;
   the module self-test reproduces it on a Joint-structured plant).
5. `fam511_*` trails `fam421_*` on the 11+ stratum (the innovation belongs with the temporal family).

## Decision language

Development rows, recorded. A favourable outcome is "above CT7 on 11+ SLA with an interval excluding
zero, and no primary endpoint interval excluding zero on the negative side". An unfavourable outcome
is recorded as "the temporal family is not the lever for late misses". Neither outcome promotes,
freezes, or replaces CT7. Untouched confirmation is a separate requirement.

## Execution

```
python -B scripts/experiments/ct7_family_equal_v1.py --dry-run                                   # synthetic
python -B scripts/experiments/ct7_family_equal_v1.py --config configs/ct7_family_equal_v1.json  # data
```

Outputs in `results/ct7_family_equal_v1/`: `RUN_FREEZE.json` (own sources + inputs hashed; a
changed freeze refuses the directory), `RESULTS.json`, `SUMMARY.csv`, `UNCERTAINTY.json`,
`BOOTSTRAP_PRIMARY_DRAWS.npz`, `PRMSCORE.json` (when the PRMBench metadata pickle is present).
Runtime: seconds for the arms, minutes for the bootstrap. No labels enter any fit.
