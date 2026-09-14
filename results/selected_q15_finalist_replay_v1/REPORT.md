# Selected q15 finalist replay v1

Status: **COMPLETE / REVIEW PASS** on 13,769 answers and 145,597 official
steps. This is a deterministic replay of the development-selected method, not
untouched confirmation.

## Finalist specification

The single method used unchanged in every benchmark is the four-view q15 bank
`{H0lim, VE0, VE0.75, VE1}` with frozen label-free orientation, Top10 token
mean computed separately for each view, and an equal step-level average in the
views' natural units. It uses no centering, scale normalization, supervised
simplex, q50 duplication, or position-varying coefficient fit. The frozen
mean-entropy q=.3 ProcessBench gate and nested PRMScore q=.8 calibration remain
unchanged.

The standalone reconstruction is bitwise identical to Experiment 1B's selected
`q15_raw_equal` vector (maximum absolute error 0).

## Absolute results

| Method | PB all-8 | PB raw exact | PRMB within | Fold pooled | OOF pooled | PRMScore |
|---|---:|---:|---:|---:|---:|---:|
| **Selected q15 raw, per-view Top10** | **36.6201%** | **33.1607%** | **.753436** | **.722708** | **.722305** | .634412 |
| Original static fusion-before-Top10 | 36.1674% | 32.6655% | .751115 | .721815 | .721407 | **.634805** |
| Original position experiment, equal z | 36.2041% | 32.9356% | .746602 | .675240 | .675153 | .589883 |
| Original local shrinkage + position | 35.9189% | 32.7330% | .747381 | .671646 | .671582 | .587410 |

## Fixed paired contrasts

Intervals are family-wise 98.333% whole-source-group bootstrap intervals over
the three comparisons. PB values are percentage points.

| Finalist minus comparison | PB delta [interval] | PRMB-within delta [interval] | PRMScore delta |
|---|---:|---:|---:|
| Original static fusion-before-Top10 | +0.453 [-0.181, +1.122] | +.002321 [.001322, .003367] | -.000392 |
| Original position equal z | +0.416 [-0.710, +1.505] | +.006834 [.003937, .009762] | +.044529 |
| Original local shrinkage + position | +0.701 [-0.462, +1.823] | +.006055 [.003317, .008850] | +.047003 |

The finalist's PRMB-within improvement is positive against all three starting
points under the corrected intervals. Its PB point estimate is also higher in
all comparisons, but each PB interval crosses zero. Against the strongest
original static score, cross-answer calibration is essentially unchanged:
PRMScore is lower by .000392. The large pooled/PRMScore recovery relative to the
position methods comes primarily from retaining absolute answer level rather
than centering every answer.

## Decisions relative to the original position-temporal version

1. Keep the same four q15 Rényi/varentropy definitions; H1 and H-infinity are
   not yet part of the finalist.
2. Apply Top10 to every feature independently before fusion, instead of fusing
   token scores and only then selecting the Top10 tokens.
3. Keep natural feature units and the answer-level offset. Remove within-answer
   centering and scaling from the deployed score.
4. Use one raw equal arithmetic mean. In standardized-coordinate terms this
   preserves the empirically useful low-alpha-heavy natural scale ratios; it is
   not interpreted as four equal effective contributions.
5. Use q15 only. Do not add q50 or the static eight-view q15+q50 duplication.
6. Use the same method in ProcessBench and PRMBench; do not choose a support or
   feature set by benchmark.
7. Reject the tested supervised global step-BCE simplex.
8. Do not promote the tested position-IU or local shrinkage-position mechanism
   into the current finalist. Its early/late mechanism result remains useful,
   but its score is not the leading locator.
9. Leave the mean-entropy q=.3 no-error gate, earliest-argmax tie rule, and
   PRMScore q=.8 calibration unchanged. Gate replacement is the next separate
   experiment, not a hidden part of this result.

## Decision

`FREEZE_Q15_RAW_PER_VIEW_TOP10_AS_CURRENT_DEVELOPMENT_FINALIST`.

This is the configuration to carry into the next gate experiment. It still
requires confirmation on a new model/response population before a
generalization claim. A future temporal selector must beat this uncentered
finalist and preserve an explicit answer-level channel; the completed
position-varying z-score method is not retained by default.

## Integrity record

- Source position and uniform result reviews both PASS.
- Four mechanism tests PASS; source and replay arrays are finite and aligned.
- Exact population: 13,769 answers, 145,597 steps.
- Finalist score equals the prior selected score bitwise.
- Score archive was frozen before replay evaluation at SHA256
  `3bd5c5b95474d75366b97b012b018d168cacafb3ccea178d26170207e220c7e6`.
- Existing exclusion-safe calibration thresholds were reused unchanged.
- No fitting, package installation, GPU/cluster action, Drive mutation, commit
  or push was performed.

