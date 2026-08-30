# Reasoning Localization 0.3662 — Conditional Contribution Ablation V1

Status: `PLANNED`; design amendment only; no result has been opened under this
contract.

## 1. Scientific question

The Phase-2 atomic arms test compact formulations around an entropy anchor.
They do not establish that a weak or harmful equal-weight atomic formulation
has no conditional value inside family6. A signal can be marginally weak yet
reduce errors that the other provenance families make.

This branch therefore separates two conclusions:

- `FORMULATION_REJECTED`: the exact atomic construction, reducer, and fusion
  that was run did not pass its gate.
- `SIGNAL_EXCLUDED`: the underlying signal also failed a frozen conditional
  contribution test inside family6.

An atomic hard failure applies to that exact formulation. It does not by
itself assign `SIGNAL_EXCLUDED`.

## 2. Placement and execution boundary

Call this branch `P2_CONDITIONAL_ABLATION`. It must finish after the frozen
C1--C8 atomic roster and before Phase-2R-B temporal transforms, Phase 3
fusion, PRMBench transfer, or early-detection transfer can use a survivor
roster.

Some atomic outputs already existed when this amendment was requested. The
ablation roster below is therefore fixed structurally from the complete
family6 provenance inventory, not selected from observed atomic F1. Evidence
from the current ProcessBench population is `DEVELOPMENT` and requires fresh
confirmation before a universal claim.

The entire execution registry, variant order, hashes, comparison family, and
tie breaks must be frozen before the first P2 conditional-ablation score is
evaluated. Run one row, rebuild the living report, and discuss it before the
next row.

## 3. Exact parent

`P2C_F6_TOP10_REFERENCE` reconstructs the current-common-population family6
representation from the frozen 28 broad token views:

1. donor/calibration-only mixed-v2 robust standardization;
2. equal mean inside each of the five non-structural provenance families used
   by the exact current R2 implementation; the structural context stream is
   retained in preparation but has zero local weight;
3. the same frozen family6 local fitting rule and answer-level detector used
   by the current R2 contract;
4. the selection-opened top-ten step reducer;
5. a per-arm deterministic grouped five-fold threshold fitted only after the
   complete label-free score freeze.

The broad inventory contains six families:

- singleton `entropy_level` (one view);
- `entropy_dynamics` (three views);
- `structural` (one retained context stream in the executable current R2
  preparation; zero local weight, so it is not a member of the R2 parent);
- `sampled_energy` (four views);
- `partition_energy` (four views);
- `topk_distribution` (six views).

The executable parent therefore averages five local families. The historical
name `family6` is retained only for lineage continuity; it must not be used to
claim that current R2 gives structural features nonzero weight. The parent is
a new common-protocol calibration reference. It may not inherit
the historical 0.3662 score, the historical detector, or the historical
population.

## 4. Frozen ablation roster

### 4.1 Family-level leave-one-out

Run all five rows corresponding to nonzero-weight parent families, regardless
of prior atomic results:

- `P2C_F6_MINUS_ENTROPY_LEVEL`
- `P2C_F6_MINUS_ENTROPY_DYNAMICS`
- `P2C_F6_MINUS_SAMPLED_ENERGY`
- `P2C_F6_MINUS_PARTITION_ENERGY`
- `P2C_F6_MINUS_TOPK_DISTRIBUTION`

Each row removes exactly one frozen family and refits only the label-free
calibration-side local scorer allowed by the parent contract. It cannot alter
the remaining views, standardization, reducer, answer detector, population,
folds, or bootstrap groups.

`P2C_F6_PLUS_STRUCTURAL_CONTROL` is the separately signed negative-control
candidate. It adds the retained structural context stream as a sixth equal-mass
family. Its contrast is candidate minus the exact five-family parent. This is
an insertion test, not a leave-one-out contribution estimate.

### 4.2 Targeted within-family view leave-one-out

These four rows answer whether atomic source signals contribute conditionally
without removing their whole provenance family:

- `P2C_F6_MINUS_ENTROPY_SWVAR16_VIEW`
- `P2C_F6_MINUS_ENTROPY_CUSUM_VIEW`
- `P2C_F6_MINUS_SAMPLED_LEVEL_VIEW`
- `P2C_F6_MINUS_PARTITION_LEVEL_VIEW`

The exact frozen view names are respectively
`entropy_sw_var_series`, `entropy_cusum_abs_series`, `spilled_series`, and
`energy_series`. After removing one view, the surviving members of that
family retain an equal within-family mean. No other family mass changes.

### 4.3 C1 formulation swap

`P2C_F6_SWAP_C1_SWVAR16` replaces only the existing
`entropy_sw_var_series` member with the exact C1 per-response-reset,
available-prefix population-variance curve. It does not add a seventh family
and does not change the other two entropy-dynamics members.

This row is necessary because the reconstructed family6 SWVar uses the frozen
historical causal-window alignment and prefix backfill, whereas C1 uses an
available-prefix warm-up. The leave-one-out row tests whether the historical
SWVar member contributes; the swap tests whether the exact C1 curve is a
better conditional member.

No adaptive window, learned mixture weight, conditional gate, or second swap
may be introduced after outcomes open.

### 4.4 Frozen C7/C8 insertion diagnostics

Two user-requested bridges test whether the uncertain atomic signals have
conditional value in the full reference without pretending they are ordinary
interchangeable scalars:

- `P2C_F6_PLUS_C7_EDIS_VIEW` inserts the exact frozen C7 onset curve as one
  additional member of `entropy_dynamics`, then re-normalizes only that
  family's equal mean. It does not create a seventh family.
- `P2C_F6_PLUS_C8_OUTER_EXPERT` retains the exact five-family parent and
  combines its token-risk curve with the frozen C8 token-risk curve as two
  equal empirical-rank outer experts. C8's 58-coordinate fitted IU system is
  not described or counted as a single primitive family6 feature.

Neither row changes the top-ten reducer, answer detector, population, folds,
or threshold contract. No C7+C8 pair is opened in Phase 2C. Such a pair is
eligible for Phase 3 only if both individual insertion premises survive.

## 5. Contrasts and interpretation

For leave-one-out row `A`, define conditional contribution as:

\[
\Delta_{\mathrm{conditional}}(A)
= F1(\mathrm{family6\ full})-F1(\mathrm{family6\ minus\ }A).
\]

A positive value means the component helps in context. For the C1 swap, use
the ordinary candidate-minus-parent direction:

\[
\Delta_{\mathrm{swap}}
=F1(\mathrm{family6\ with\ C1\ SWVar})-F1(\mathrm{family6\ parent}).
\]

Report raw scores and both directions explicitly so a sign convention cannot
reverse the conclusion.

The thirteen primary macro-F1 contrasts (five family leave-one-outs, four view
leave-one-outs, one SWVar swap, structural insertion control, C7 insertion,
and C8 outer-expert insertion) form one closed Bonferroni family.
Use 20,000 paired whole-source-question draws on the exact eight-Qwen common
population. Required secondary outputs are exact-error, within-one, clean
abstention, cell and family W/T/L, worst cell, worst error family, prediction
flips, and short/medium/long step-span strata.

Use `+0.003` as the minimal conditional-contribution benefit and `-0.005` as
the practical-harm boundary. A component is conditionally supported only if:

- the multiplicity-valid lower bound exceeds `+0.003`;
- at least six of eight scorer cells are nonnegative;
- the worst cell is at least `-0.020`;
- exact-error and clean-abstention each regress by no more than `0.010`.

An interval crossing zero is `INCONCLUSIVE`, not rejection. A hard technical,
leakage, suffix-invariance, provenance, or population failure stops the exact
row. A worst-cell hard boundary of `-0.030` rejects the exact formulation but
does not rewrite another row's signal-level conclusion.

## 6. Eligibility for later fusion

A signal or family may enter the bounded Phase-3 roster by either of two
routes:

1. `ATOMIC_SURVIVOR`: its exact atomic formulation passes the Phase-2 atomic
   gate.
2. `CONDITIONAL_ONLY`: its atomic formulation does not pass, but its frozen
   leave-one-out conditional contribution is supported under this contract.

`CONDITIONAL_ONLY` preserves only the exact family placement and formulation
that the ablation supports. It does not authorize a free-standing channel,
arbitrary weight search, or new cross-family combination. A failed equal-rank
fusion such as C1 remains rejected even if the underlying SWVar view later
earns conditional-only eligibility.

C7 can earn conditional-only eligibility only in the registered
`entropy_dynamics` placement. C8 can earn it only as the registered outer
expert; it cannot be unpacked into a post-result-selected subset of its 58
coordinates.

## 7. Bounded Phase-3 fusion ladder

Phase 3 is not an all-method-by-all-family grid. After the Phase-2C survivor
set freezes, it proceeds in the following order:

1. matched equal-family and ordinary IU-PCR parents;
2. hierarchical family experts, where singleton `entropy_level` passes
   through and eligible multi-view families may use one preregistered U-PCR,
   IU-PCR, L-SML, or DUFS-LIU formulation selected only by nested
   calibration/stability evidence;
3. SU-PCR only when its sparse-support and identifiability premise gates pass;
4. at most one B3 conditional nonlinear arm after a simpler exact parent
   survives;
5. one isolated `gate -> SU-PCR` mechanism study, if opened, comparing plain
   SU on frozen inputs against a donor-only size-matched input gate followed by
   unchanged SU.

The `gate -> SU-PCR` study must include permuted-gate and size-matched ranking
controls, and must feed the same selected inputs to IU/LIU/L-SML controls so
that input selection is separated from the SU fusion rule. A gate may optimize
only a frozen donor-side SU structural diagnostic, never ProcessBench or
PRMBench F1. The older pseudo-label DUFS gate used continuous L-SML output and
did not establish that a learned-input SU-PCR beat DUFS-LIU. Separately, the
corrected final-answer `STG_SU_STABLE` experiment did learn fold-stable sparse
support for SU-PCR and recovered canonical SU-PCR to near IU/DUFS-LIU parity,
but its small advantages over IU and matched random support were not
statistically supported. It is therefore a genuine Phase-3 mechanism premise,
not evidence that STG already improves localization. The bounded transfer
contract is in `REASONING_LOCALIZATION_03662_STG_GRAPH_TRANSFER_V1.md`.

If neither route passes, the signal is `SIGNAL_EXCLUDED` for Phase 3. An
inconclusive conditional result may be retained for fresh confirmation but is
not rankable or promotion-eligible on the current population.

## 8. Transfer and reporting

ProcessBench and PRMBench remain separate estimands. The P2 ablation uses only
ProcessBench for development. Any later PRMBench evaluation must freeze the
full parent, ablated roster, orientations, scaling, and weights before labels
open; no shared average is permitted.

The living HTML must add:

- a family6 composition diagram;
- a full-versus-leave-one-out paired-delta forest;
- a family-by-cell contribution heatmap;
- exact-error versus clean-abstention component plot;
- a ledger distinguishing `FORMULATION_REJECTED`, `ATOMIC_SURVIVOR`,
  `CONDITIONAL_ONLY`, `INCONCLUSIVE`, and `SIGNAL_EXCLUDED`.

Historical family6 and the 0.3662 anchor remain context-only and may not enter
this common-protocol leaderboard.
