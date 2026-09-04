# Joint L-SML localization — Claude Code handoff

Updated: 2026-09-04

## Start here

This branch contains the complete, versioned Joint L-SML localization study:

- Branch: `codex/joint-lsml-localization-eval-v1`
- Structural implementation commit: `c5a658a6`
- Existing-data localization and PRMBench evaluation commit: `18f9d270`
- ProcessBench coverage amendment and evaluation commit: `88866daf`
- Canonical narrative: `HISTORY.md`, Steps 347 and 348
- Current project handoff: the first two entries in `PROGRESS.md`

Read the following artifacts before changing the method:

1. `docs/experiments/JOINT_LSML_V1.md`
2. `results/joint_lsml_v1_r2/REPORT.md`
3. `docs/experiments/JOINT_LSML_LOCALIZATION_EVALUATION_V1.md`
4. `results/joint_lsml_existing_localization_v1/REPORT.md`
5. `docs/experiments/JOINT_LSML_PROCESSBENCH_AMENDMENT_V1.md`
6. `results/joint_lsml_existing_localization_v1/processbench_amendment_v1/REPORT.md`

The result reports, registries, manifests, independent audits, CSV summaries,
and static PNG plots are committed. Large target-free NPZ inputs and score
arrays remain intentionally ignored; their identities and hashes are frozen in
the committed manifests. Do not infer that their absence from Git means that a
score or source was not audited.

## What was built

The candidate is a continuous Joint L-SML fusion head over the frozen active-23
roster. It estimates one residual partition without labels, then jointly fits a
global rank-one factor and partition-specific factors. All efficacy arms share
the same absolute raw-domain orientation, active roster, preprocessing, and
task reducer.

The development chain also includes:

- absolute orientation and unstable/weak-feature removal;
- exact leave-one-answer-out consensus partition selection;
- deterministic multi-start structural fitting and Jacobian/conditioning gates;
- an exact empty-group dispatch to flat SML;
- score freezing before outcome access;
- paired source-group uncertainty;
- independent pre-label, post-freeze, amendment, and result audits.

Component B/SUMMA prevalence is not part of this method. No binary token-level
latent target, `rho_hat`, prevalence pruning, rank transform, Katz centrality,
DUFS, or graph-normalized SML was added to the evaluated candidate.

## Structural result

The structural mechanism worked conditionally but did not establish efficacy.
Joint reduced off-diagonal covariance misfit relative to hard L-SML in every
fitted lane. PRMBench selected K=3 with group sizes 13/7/3 and reduced misfit
from 0.245183 to 0.203177.

Seven of eight ProcessBench cells were structurally admissible. The sole block,
`processbench_math_qwen3_4b`, had no partition satisfying the frozen minimum
group-size rule. The original all-eight ProcessBench policy therefore returned
`STRUCTURAL_NO_SCORE` before labels.

## PRMBench efficacy result

The registered pure Joint L-SML head was evaluated on 6,208 error responses and
83,280 official steps with 2,000 paired source-group bootstrap draws.

- Joint L-SML AUROC: 0.669063
- IU-PCR AUROC: 0.671539
- Equal-family AUROC: 0.668774
- Fixed-family continuous L-SML AUROC: 0.672619
- Joint minus IU-PCR: -0.002476, 95% interval [-0.004103, -0.000908]
- Joint minus fixed-family L-SML: -0.003556, 95% interval [-0.004572, -0.002578]
- Joint minus equal-family: +0.000289, 95% interval [-0.001620, +0.002271]

Joint AUPRC was 0.251757, below all three controls. Registered verdict: `HARM`.

## ProcessBench efficacy result

After an explicit user override, a separate versioned coverage amendment was
registered. It reuses the exact frozen Joint weights in the seven admissible
cells and uses the already-implemented `G=[]` flat-SML alias in the sole blocked
cell. This is therefore `Joint-or-flat`, not pure Joint L-SML on all eight cells.

The evaluation uses 3,400 source questions, 6,800 paired Qwen model rows, five
grouped folds, and 2,000 paired threshold-refit bootstrap draws.

- Joint-or-flat macro-F1: 0.269290
- IU-PCR macro-F1: 0.340378
- Equal-family macro-F1: 0.285986
- Fixed-family continuous L-SML macro-F1: 0.342940
- Candidate minus IU-PCR: -0.071087, 95% interval [-0.084721, -0.054335]
- Candidate minus fixed-family L-SML: -0.073650, 95% interval [-0.091279, -0.061706]
- Candidate minus equal-family: -0.016696, 95% interval [-0.033615, -0.000629]

The fallback Qwen3-4B/MATH cell is poor at 0.068258 F1, but it is not the sole
cause. The pure-Joint Qwen3-8B/GSM8K cell is also poor at 0.143111. The
selection-conditioned mean over the seven structurally fitted cells is 0.298009
for Joint, versus 0.341730 for IU-PCR and 0.341834 for fixed-family L-SML.
That diagnostic shares thresholds calibrated by the all-eight procedure, has no
interval, and is not fallback-independent.

Registered verdict: `HARM__NO_PROMOTION`.

## Reducer and historical boundary

ProcessBench uses detector=max token risk and locator=argmax of the fixed mean
of the top `min(10, step_length)` token risks. This is top-ten, not top-five and
not top-ten-percent. PRMBench uses maximum token risk inside each official step
span.

The historical 0.3662328342 ProcessBench value belongs to a different
population/split/detector/reducer contract. It is an audit anchor, not a matched
comparator for this active-23 experiment. The matched ProcessBench controls are
IU-PCR 0.340378, equal-family 0.285986, and fixed-family L-SML 0.342940.

## Scientific conclusion

Lower covariance reconstruction error was not a sufficient model-selection
criterion for localization. The current joint factor fit converts its structural
solution into a weight map that degrades token/step ranking. The negative result
is present on both tasks and is not explained only by the one-cell flat fallback.

Keep the target-free orientation, active-23 roster, structural diagnostics,
score-freeze discipline, and paired uncertainty machinery. Do not promote the
current Joint weight map as the localization fusion head.

Any future redesign should address weight-map regularization or stability rather
than adding covariance-fit flexibility. It must use a new frozen protocol and
fresh data for any generalization or new-leader claim. Do not select or tune a
new variant from the already-open ProcessBench/PRMBench outcomes.

## Reproduction and verification entry points

- Core estimator: `spectral_utils/joint_lsml.py`
- Localization adapter: `spectral_utils/joint_lsml_localization.py`
- ProcessBench amendment adapter: `spectral_utils/joint_lsml_processbench_amendment.py`
- Structural/localization runner: `scripts/joint_lsml_localization/run_existing_v1.py`
- PRMBench evaluator: `scripts/joint_lsml_localization/evaluate_existing_v1_r2.py`
- ProcessBench runner: `scripts/joint_lsml_localization/run_processbench_amendment_v1.py`
- ProcessBench evaluator: `scripts/joint_lsml_localization/evaluate_processbench_amendment_v1.py`
- PRMBench result audit: `results/joint_lsml_existing_localization_v1/INDEPENDENT_EVALUATION_RESULT_AUDIT.json`
- ProcessBench result audit: `results/joint_lsml_existing_localization_v1/processbench_amendment_v1/INDEPENDENT_EVALUATION_RESULT_AUDIT.json`

The last independent ProcessBench audit reproduced all four point estimates,
all 32 per-cell F1 values, the seven-cell diagnostic, and every one of the 2,000
paired bootstrap intervals with maximum absolute error zero. The PRMBench audit
reproduced point estimates and intervals independently to numerical precision.

## Post-hoc failure localization

The versioned forensic package is under
`results/joint_lsml_existing_localization_v1/failure_diagnostic_v1/`; start with
its `REPORT.md`, `DIAGNOSTIC_SUMMARY.json`, and `INDEPENDENT_AUDIT.json`. It
reuses only the frozen scores, weights and opened evaluation outcomes. It fits
no new fusion candidate and is explicitly
`POSTHOC_RETROSPECTIVE_FAILURE_DIAGNOSTIC`.

The strongest supported diagnosis is:

1. ProcessBench is dominated by a score-scale transfer failure. The final Joint
   weight-vector L2 norm varies from roughly 1.30 to 1.78 across pure-Joint
   cells, while fixed-family continuous L-SML is unit norm in every cell. The
   q4/MATH fallback is also unit-norm flat SML, mixed under the same model-level
   threshold with larger-norm Joint cells.
2. q4/MATH and pure-Joint q8/GSM8K account for 89.4% of the net candidate-minus-IU
   panel loss. Their detector Spearman against fixed L-SML is 0.989/0.980 and
   locator agreement is 91.8%/88.8%, but the threshold activates only 9.6%/14.5%
   of responses, versus 67.0%/53.8% for fixed L-SML.
3. PRMBench rules out threshold scale as the complete explanation. Joint reduces
   off-diagonal misfit by 17.1% but loses 0.248 AUROC percentage points to IU
   and 0.356 percentage points to fixed L-SML. The deployed hierarchical head
   uses the fitted global loading and a second SML across virtual groups, but
   does not directly use the fitted
   group-factor loadings. The covariance objective and deployed head are
   therefore not aligned.

This does not prove INTERNAL K=3 grouping is wrong. The original frozen study
did not score ordinary continuous L-SML with the same INTERNAL groups, so group
discovery and map construction must be separated in the next experiment.

The proposed bounded successor study is
`docs/experiments/JOINT_LSML_OPTIMIZATION_PLAN_V1.md`. Its first invariant is
donor-frozen fused-score standard deviation one before cross-cell thresholding.
It then crosses INTERNAL/provenance grouping with ordinary continuous-LSML/the
hierarchical map, compares token-fuse-then-step-reduce with per-feature
step-reduce-then-fuse, and gives IU and Joint the same maximum eight-config inner
budget. ProcessBench and PRMBench remain separate objectives.

DUFS is not a K selector: its output is a per-feature gate, not a partition
count. A single parameter-free soft-affinity DUFS route is allowed only as a
separate, fold-contained successor candidate. Prior project DUFS localization
comparisons were effectively tied with IU, so it is secondary to fixing scale
and the objective/head mismatch.

## Claude Code continuation boundary

Claude may inspect, reproduce, or critique this branch directly. Before any new
experiment, it should read `CLAUDE.md`, `PROGRESS.md`, this handoff, the two
canonical reports, and their independent audits. A proposed successor should be
registered as a new method/protocol; the committed negative results and their
frozen registries must remain immutable.
