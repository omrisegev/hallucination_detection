# LR feature-choice versus weighting diagnostic (task 7)

This is a **new internal source-group OOF diagnostic**, distinct from the
historical cached LR panel. The historical split provenance cannot currently
be certified; none of these new predictions will be renamed as a replay of it.

## Locked experiment

All ten banks use the authoritative PRMBench outer and per-outer inner maps in
`inputs/FOLDS_V2.json`. Entire canonical source groups stay together. Outer test
groups do not enter a scaler, grouping, weight fit, mask, or hyperparameter
choice. The inner maps are separate five-fold maps, not the outer fold numbers.

The supervised rank selector is balanced LR with **fixed C=1**, chosen before
running this diagnostic. It is freshly fitted on each inner/outer training
partition. The LR fuser's C is independently tuned on the seven historical C
values by mean class-balanced inner log-loss. No outer-test labels choose C,
polarity, masks, or grouping.

In the one half-bank crossover, every mask keeps `ceil(P/2)` columns. The
ranking quantity is absolute contribution per training SD on the same input
scale for all three methods: effective weight on answer-z V times the training
SD of V. Ties use bank order. The LR mask is applied to LR and to each matching
label-free fuser; each label-free mask is applied to its own fuser and LR.
Full-bank fits are controls. **A label-free fuser on an LR-selected mask is a
supervised-selected diagnostic**, not a new label-free contribution.

R1/R2/R3 preserve the historical threshold/contrast-pair/bottom-quartile
definitions, but now derive them from a C=1 training-only selector inside every
inner and outer partition. This deliberately replaces the leaky global mean
of supervised outer-fold coefficients. It is a corrected diagnostic protocol,
not a claim of numerical identity to the historical restricted estimates.

L-SML uses the actual continuous entry point and fails explicitly if fewer
than three groups are selected. IU-PCR uses the pinned two-component L2
defaults, no exclusion or fallback, and training-only normalization. L-SML
and IU-PCR retain their native label-free sign rules; evaluation labels never
flip their signs. All fitted affine weights reproduce their native training
scores to a maximum absolute error of 1e-9 or fail the run.

N16/N56 drop-one and family-omission probes refit balanced LR at the **full
LR's train-selected C for each outer fold**. These are supervised, exploratory
conditional-omission diagnostics; correlated coefficients and omission gains
are not causal feature importance.

## Outputs and interpretation

- `checkpoints/<bank>/outer_<k>/` contains fitted coefficients, train/test group
  manifests, actual masks in every validation partition, C losses, raw test
  scores, failures and source hashes.
- `OOF_<bank>.npz` and `PER_ANSWER_<bank>.npz` retain step-level and answer-level
  raw material. `METRICS.csv` is the new panel; incomplete arms have explicit
  coverage and cannot be treated as full-population results.
- Weights are exported both on V (for affine replay) and per training SD (for
  comparison), with L1-normalized signed contributions for figures. Fold
  stability, selection frequency, family mass and inter-method agreement are
  descriptive summaries.
- Four-way outcomes compare top1-on-error against the new full-bank LR, by
  actual error class, first-error relative-position quintile, answer step-count
  bin and, where available, raw answer token count. First-error position is an
  outcome-side diagnostic, not a feature fed to any model.
- Paired intervals use 5,000 bootstrap draws of the **canonical source group**,
  seed 20261011, conditional on the fitted predictions and choices. They do not
  refit selection in each bootstrap and are exploratory pointwise intervals.

Ranking is evaluated within each answer, then averaged. An answer is entirely
in one outer fold, so OOF calibration shifts cannot affect its ordering. No
global AUC over concatenated OOF steps is used. Results are internal validation
on already inspected development data; they do not certify a fresh external
test or select a new champion.

## Reproduce

From the project root, using the shared main virtual environment:

```sh
/Users/osegev/Desktop/hallucination_detection/.venv/bin/python results/advisor_completion_2026-10-11/lr/reproduce.py
```

The inputs are prepared by the v11 population/bank adapter. Each bank/outer
checkpoint is atomic and resumable. The reproducibility wrapper **first rejects
changed input/script/helper identities** rather than silently reusing old
checkpoints, then rebuilds tables/figures, native replay and acceptance checks.
Detailed logs are retained in `logs/`. The underlying driver accepts
`--bank N16 --outer 0` for one partition and `--aggregate` for saved-score-only
aggregation; use the wrapper for an identity-checked complete replay.

The runnable helpers live in `spectral_utils/advisor_lr.py`; there are no
notebook-only fusion helpers, package installs, GPU jobs or Git actions.
