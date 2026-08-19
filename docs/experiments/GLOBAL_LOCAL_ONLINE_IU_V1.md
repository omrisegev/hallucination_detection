# Global-Local-Online IU retrospective optimization — v1

Status: frozen before candidate scoring, 2026-08-16.

This protocol authorizes only CPU analysis of already materialized caches and
canonical score artifacts. It does not authorize inference, a cluster job, a
large download, Google Drive mutation, or any change to the frozen A6/PTNI
program.

## Question and claim boundary

Find the smallest causal, label-free-scored Global-Local-Online IU architecture
on the joint performance/compute Pareto frontier for:

1. first-error reasoning localization; and
2. prediction of final-answer error from an unfinished prefix.

The panels remain separate. A candidate may be promoted only if grouped
evidence improves one co-primary panel, the other remains within its frozen
non-inferiority margin, and the candidate is not dominated in measured cost.
All opened ProcessBench, PRMBench, Phase-15, and existing 11-cell online data
are retrospective development evidence, never fresh confirmation.

The score constructor is label-free. Correctness labels may be used only for
the development-only roster check described below, decision-threshold
calibration, and evaluation. The proper description is **unsupervised scorer
with calibrated decision policies**.

## Frozen data roles and independence

- Localization regression anchor: the eight ProcessBench cells in
  `results/fixed_application_pipelines_v1`, grouped by original question.
  Qwen3-4B/GSM8K and Qwen3-4B/MATH are historical component-development cells;
  the other six are retrospective non-selection/model-transfer cells. The 4B
  and 8B rows reuse the same four dataset populations and are not independent
  datasets.
- PRMBench regression anchor: the Qwen3-8B teacher-forced trajectory, grouped
  by original PRMBench response. Its every-step AUROC/F1 remains a separate
  task and is never averaged with ProcessBench.
- Early panel: the frozen 11-cell screen under
  `results/early_online_localization_models_v1`, grouped by the saved question
  identifier. Its five equal-weight families are MATH-500, ProcessBench GSM8K,
  MATH, OlympiadBench, and OmniMath. Generator-rescored copies do not multiply
  family weight.
- Temperatures and protocols outside those panels are inventory/heterogeneity
  evidence only unless their cache passes the causal-prefix contract and the
  same frozen runner is applied without adaptation.

No row or token is treated as statistically independent when questions are
shared. Bootstrap resampling is by question within cell, followed by equal
weighting of dataset family. The seed is `20260816` and the primary interval
uses 2,000 bootstrap draws.

## Frozen feature and causal contract

- `CONFIDENCE_FEATURE_SIGNS_V1` is immutable. Larger oriented values mean
  confidence and risk uses the negative direction.
- A quarantined raw view may be absent or replaced by its frozen mixed-v2
  transform; raw and transformed copies may not coexist.
- IU28 without final length is the primary online adapter. Elapsed prefix
  length is a separately named 29th-stream ablation.
- At budget `b`, every feature is a function only of telemetry at indices
  `<= b`. A completed feature matrix may not be sliced. Appending or replacing
  the suffix must leave the decision at `b` bit-identical.
- Step spans are used only after scoring, to map a frozen token prediction to
  a ProcessBench/PRMBench step.
- All IU arms use two components, L2 additive solve, no exclusion, no
  difficulty fallback, and `scale_ratio=0.25`. Ordinary IU is exactly
  `lambda=0`.

## Frozen baseline and candidate roster

The Global and Local heads are frozen for this bounded cycle:

- global presence: the ordinary mixed-v2 IU-PCR head from the fixed reasoning
  package;
- local onset/token risk: the ordinary shared core-five trajectory IU-PCR head
  from the same package;
- ProcessBench decision: the existing 0.75 global / 0.25 local standardized
  blend and calibration-half threshold;
- PRMBench step score: maximum frozen token risk inside the supplied step span.

Their score hashes must reproduce before online candidates are considered.
The new candidates change only the Online head, so localization non-inferiority
is tested mechanically by score-hash equality as well as numerically.

All online candidates consume only the causally rebuilt `cusum_max` and
`sw_var_peak` trajectories at the absolute monitor grid. Reference medians,
scales, ordinary-IU weights, and orientation are fitted without labels on the
calibration questions. Every calibration trace contributes the same number of
fit rows. Dynamic coordinates are risk-oriented by their declared mechanism;
final score direction is aligned to the unlabeled mean-coordinate consensus.

| id | online coordinates | fusion | hypothesis |
|---|---|---|---|
| `iu28_no_length` | frozen 28-stream maximum token risk | ordinary IU | primary historical online baseline |
| `deepconf_entropy_w64` | lowest 64-token entropy-group confidence proxy | fixed proxy | strongest access-matched published-family control |
| `cusum_swvar_equal` | current standardized CUSUM maximum and `sw_var` peak | fixed 1/2 + 1/2 | magnitude-only mechanism control |
| `dyn_level4_iu` | current and running maximum for each of CUSUM/`sw_var` | ordinary IU | accumulated extremes retain early warning |
| `dyn_persist6_iu` | current, elapsed-normalized positive area, and persistent run fraction for each signal | ordinary IU | sustained abnormal dynamics beat one-off maxima |
| `dyn_change6_iu` | current, last slope, and current-minus-running-maximum recovery for each signal | ordinary IU | direction and failure-to-recover add information |

The reference level for positive area/run length is the calibration median of
that component. No candidate may be added, deleted, sign-flipped, or renamed
after its performance is read. `dyn_level4_iu` is the preferred candidate if
two dynamic arms are statistically tied because it has the smallest state.

Graph arms are controls, not search candidates. A same-matrix check compares
ordinary IU (`lambda=0`) with uniform, DUFS, and temporal Laplacians where the
existing component artifact supplies all four. No graph hyperparameter may be
tuned in this cycle. The exact `lambda=0` output must equal ordinary IU.

## Metrics and frozen gates

### Localization

- global answer-error AUROC/AUPRC;
- exact first-error step, within-one-step SLA, clean abstention accuracy, and
  ProcessBench F1;
- PRMBench every-step AUROC/AUPRC/F1 separately.

Primary localization reference is fixed reasoning IU-PCR: ProcessBench macro
F1 `0.306999` and PRMBench step AUROC `0.671149`. Frozen non-inferiority margins
are `-0.010` absolute ProcessBench F1, `-0.010` global answer AUROC, and
`-0.010` PRMBench AUROC. These are smaller than the typical 0.016--0.038
per-cell repeated-split ProcessBench SD while still excluding a practically
meaningful one-point loss.

### Early/online

- AUROC/AUPRC at 16/32/64/128/256/512 among unfinished traces;
- eligible trace, cell, and family counts;
- Spearman correlation to the method's own completed score and agreement with
  its own frozen final decision;
- declaration coverage, full-horizon ever-wrong rate, selective error, false
  alarm/clearance, and declaration budget.

The single primary early ranking endpoint is the equal-family mean of AUROC at
64 and 128 tokens. This was frozen because both budgets retain 10/11 usable
cells and were the promising region in the prior screen. `iu28_no_length` is
the primary reference; DeepConf-w64 is the required control. The early
non-inferiority margin for a candidate promoted on localization is `-0.015`
AUROC, derived conservatively from the prior family-bootstrap uncertainty.
For declaration, ever-wrong may increase by at most `0.020`, and coverage may
fall by at most `0.050`; neither declaration metric can by itself rescue a
ranking failure.

Promotion on the early panel requires the 95% family-bootstrap interval for
the 64/128 mean paired delta versus IU28 to be wholly above zero, positive
direction in at least three of five families, localization score-hash equality
or all localization margins satisfied, and no cost domination. A positive
cell macro with an interval crossing zero is parity, not promotion.

## Efficiency accounting

For every arm record feature count, fit wall time, scoring wall time, peak
resident-memory increment where measurable, persistent scalar state per trace,
and asymptotic per-monitor update cost. The dynamic candidates use O(1) state
and O(1) work per new monitoring point. A feature/coordinate is retained only
if add-one/drop-one evidence exceeds grouped uncertainty or it is necessary to
define the simpler winning mechanism.

## Required regression tests

Before scoring candidates:

1. reproduce the canonical early and localization anchors;
2. suffix invariance under arbitrary suffix replacement;
3. feature-order invariance;
4. fit identity after labels are permuted or removed;
5. repeated-run bit identity;
6. declared missing-feature handling;
7. exact ordinary-IU equality for every `lambda=0` graph path;
8. localization score-hash identity for online-only candidates.

## Decision and next boundary

At completion, select the simplest empirically non-dominated candidate that
passes the joint gate. Otherwise retain the ordinary-IU/previous online
baseline and close the tested dynamic mechanism with the observed reason.
Fresh confirmation, exact DeepConf/REFRAIN conditions, new inference, and GPU
work remain a separate explicit approval gate.
