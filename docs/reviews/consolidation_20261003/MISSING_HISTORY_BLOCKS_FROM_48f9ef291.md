# Missing historical blocks for review — not yet applied to consolidated HISTORY

Exact text from HISTORY.md at `48f9ef291`, in source order. These eight level-two step blocks were omitted by the restoration at `b4e97ed72`. Preserve original numbers and line identity when integrating; do not overwrite same-numbered steps of other lines.

## Step 339 [localization] — H3 is above 0.3662 numerically, but not confirmed

The exact Stage-4 historical evaluator was checksum-replayed before candidate
audit results opened: entropy/top-five is `0.3614213584` and the historical
finalist is `0.3662328342`. H0/H2/H3 predictions were frozen for the identical
eight cells, fixed 40% calibration and 20% audit roles, 1,270 scorer rows and
635 paired source questions.

H0 scores `0.374099`, H2 is raw best at `0.374793`, and H3 scores `0.372663`.
H3-minus-historical is `+0.006431 [-0.026891,+0.039473]`, 5/0/3 cell W/T/L,
with worst cell `-0.034609`. H2-minus-historical is
`+0.008560 [-0.024610,+0.040869]`. The positive point estimates are
`PROMISING_UNCONFIRMED`; intervals crossing zero are not rejection.

The detector/localizer cross does not support an isolated H3 localization
gain. Under the historical detector, H3 localization changes F1 by
`-0.003070 [-0.028453,+0.020618]`; replacing only the detector while keeping
the historical localizer changes it by `+0.013119
[-0.005697,+0.034240]`; interaction is
`-0.003619 [-0.012159,+0.004363]`. H3 retains supported PRMBench specialization
but does not satisfy the preregistered matched-ProcessBench promotion gate.
Retain H2 as the ProcessBench raw-best candidate and H3 as the PRMBench-enhanced
candidate for independent confirmation.

## Step 338 [localization] — H3 is a supported PRMBench specialist

The frozen H0→H2→H3 ladder was transferred without PRMBench tuning to 83,280
annotated steps in 6,208 paired source groups. A first parent-alias attempt
hard-failed before labels because it compared top-ten H0 against the Phase-1
top-five artifact. The corrected V2 non-rankable top-five control and all
imported Qwen H0/H2/H3 scores reproduce exactly.

H0 scores `0.592057` AUROC; H2 scores `0.597871`; H3 scores `0.619469`.
Bonferroni-simultaneous deltas are H2−H0
`+0.005814 [+0.004710,+0.006973]`, H3−H0
`+0.027412 [+0.023675,+0.031091]`, and H3−H2
`+0.021598 [+0.017653,+0.025457]`. H3 also improves AUPRC over both parents
and wins all eight evaluable error families.

Verdict: `PRMBENCH_SPECIALIST / NO_PHASE4_PROMOTION`. The result is a strong
task interaction: C8 adds dense every-step ranking value, while its
incremental ProcessBench first-error value remains unconfirmed. Historically
opened labels, outcome-selected ancestry, missing source strata and absent
ProcessBench promotion prevent a universal or confirmatory claim.

## Step 337 [localization] — H2 remains positive on Llama; H3 adds no confirmed increment

No new ProcessBench questions were available locally: all three scorer models
cover the exact same 3,400 source groups already opened in Phase 1. The next
registered experiment was therefore a four-cell Llama scorer-family transfer,
not fresh confirmation. Before importing Llama labels, the executable
reconstructed the frozen eight-Qwen H0-combined, H2 and H3 scores with maximum
absolute error zero. H2/H3 then copied H0's abstention decisions exactly.

H0 scores `0.348909` macro F1. H2 is raw best at `0.355583`, delta
`+0.006674 [-0.007091,+0.020943]`; H3 scores `0.353281`, delta versus H0
`+0.004372 [-0.009677,+0.018452]`. H3-minus-H2 is
`-0.002303 [-0.011662,+0.007001]`. The intervals are Bonferroni-simultaneous
across the three frozen primary contrasts. Both candidate-versus-H0 point
estimates are positive but uncertain, so they are `PROMISING_UNCONFIRMED`, not
rejected. H3 improves the secondary within-one metric in all four families,
but its worst macro-F1 family delta is `-0.021595` and it has not established
incremental value over H2.

Verdict: retain H0→H2→H3 as a bounded fresh-question confirmation ladder, with
H2 as a required separate parent. Do not promote H3, open Phase 3, or call this
fresh confirmation. PRMBench remains unopened.

---

## Step 330 — C7/C8 transfer does not clear the top-ten gate

The exact C7 EDIS-onset and C8 self-innovation scorers were transferred without
retuning from the completed Qwen-eight atomic screen to the four Llama-3.1
ProcessBench scorer cells.  Both remain stronger than top-five and family6 on
this panel, and C8 is directionally stronger than its matched IU29/top-ten
parent.  Neither improves the stronger entropy/top-ten reference: C7 delta is
`-0.004056 [-0.021370,+0.012994]`; C8 is
`-0.003511 [-0.017028,+0.010138]` under the frozen simultaneous family.

Decision overlap with family6 leaves a nontrivial descriptive oracle ceiling,
but step-score rank correlations are `0.963--0.983` and no accessible router or
fusion was evaluated.  The transfer branch closes `NO_TRANSFER_PROMOTION`; the
family6 fusion proposed from these candidates remains `NOT_RUN_BY_GATE`.
Because the Llama source questions and labels were opened in Phase 1, this is
transfer evidence rather than independent fresh confirmation.

---

## Step 332 [localization] — Phase-2C removal audit identifies two load-bearing families

The exact current R2 implementation was first corrected: despite the historical
`family6` lineage name, it gives nonzero local weight to five non-structural
families, while the retained structural context stream has zero local weight.
An exact five-family/top-ten parent (`F1=0.354261`) and thirteen conditional
contrasts were frozen before the first candidate result.

All nine planned removals then completed on the common eight-Qwen ProcessBench
population with 20,000 paired whole-question bootstrap draws. Entropy level
contributes `+0.024646 [0.007506,0.042031]` macro F1 and top-k distribution
contributes `+0.022669 [0.004708,0.040473]`; both clear the registered aggregate
conditional-contribution boundary. Each also has a material exact-versus-clean
tradeoff, so neither passes the complete promotion gate. Partition energy is
promising but unconfirmed at `+0.010633 [-0.000676,0.022323]`. Entropy dynamics,
sampled energy, SWVar16, CUSUM, sampled level and partition level remain
inconclusive; no interval crossing zero is interpreted as rejection.

The corrected `STG_SU_STABLE` final-answer side-worktree evidence was also
verified. It recovers canonical SU-PCR to near IU/DUFS-LIU parity but does not
show supported superiority or localization value. A later survivor-gated
feature/temporal graph branch was registered with zero-strength alias, grouped
fold stability, random-support, time-permutation and feature-permutation
controls. The next Phase-2C state is the structural insertion control.

---

## Step 336 [localization] — H3 equal combines the favorable edits under role separation

The isolated H2/H3 experiment tested the exact combined idea requested after
Phase 2C. H0 retained its threshold and every clean/error decision. H2 removed
sampled-token energy and the partition-level `energy_series` view, then added
C7 inside entropy dynamics. H3 fused H2 with frozen C8 step ranks only for H0
non-abstentions.

H3 equal is raw best at macro F1 `0.366653`, improving H0 by `+0.012392` with
a four-contrast simultaneous interval `[+0.001769,+0.022807]`. Exact-error
accuracy improves `+0.012941`, within-one improves in all eight cells, and
clean abstention is identical by construction. Reliability weighting does not
beat equal weighting and learns weights very close to 0.5.

This is directional development evidence, not promotion: the lower interval
bound remains below the registered `+0.003` practical threshold and the roster
was outcome-selected on opened questions. The next admissible claim-bearing
step is a frozen H3-equal confirmation on fresh questions. The historical
`0.3662` remains a separate regime and is not a paired comparator.

---

## Step 335 [localization] — STEP-CUT closes the within-answer graph premise

The development-only STEP-CUT screen froze donor-only scores before label
import but used a previously opened ProcessBench population. Its full temporal
chain-plus-mutual-2NN graph beat length-matched uniform chance in Qwen
late-error Hit@1, yet failed every graph-content negative control: chain-only,
step-permuted features, and random edges all did materially better. MRR was
also below chance. Equal-rank entropy-plus-graph fusion caused interval-
supported Hit@1 harm on Qwen and repeated it on Llama.

This is evidence that the apparent chance lift comes from topology/position
bias, not useful learned graph content. The planned Phase-3
`P3G_T1_TEMPORAL_GRAPH` arm was not itself executed and is now
`NOT_RUN_BY_GATE / NO_PROMOTION`; combined feature-by-time graph fusion is
also closed. The separate STG feature/family-support candidate remains planned
because it tests a different axis. Chain-only remains a diagnostic observation
only and needs a new preregistered premise before it can run.

---

## Step 334 [localization] — Phase 2C closes with a promising C8 outer expert

The structural insertion, exact C1-SWVar swap, C7-in-family insertion and C8
outer-expert insertion completed the frozen thirteen-contrast conditional
study. Structural equal-family insertion is directionally negative but
inconclusive (`-0.004564 [-0.010259,+0.000784]`); the SWVar formulation swap
is nearly neutral (`-0.000148 [-0.001961,+0.001477]`). C7 gives a small
uncertain gain (`+0.002116 [-0.002537,+0.006754]`).

C8 is the raw best at F1 `0.364997`, improving the five-family/top-ten parent
by `+0.010735`, but its simultaneous interval
`[-0.002481,+0.024255]` crosses zero. It raises exact-error accuracy by
`+0.014429` while lowering clean abstention by `-0.022491`. It is therefore
`PROMISING_UNCONFIRMED / NO_PROMOTION`, not rejected and not a Phase-3
survivor on the current opened population.

Phase 2C closes `NO_FULL_CONDITIONAL_PROMOTION`. The next scientifically valid
use of partition energy or C8 is independent confirmation under a frozen
contract; the current results do not authorize an outcome-selected fusion.

---

