# Global-Local-Online IU v1 — retrospective optimization report

## Scope correction (Step 271)

This report closes only the three preregistered transformations of completed
CUSUM/`sw_var` summaries on the saved coarse monitor grid. The experiment froze
the Global and Local heads, their 0.75/0.25 decision blend, IU settings, feature
allocation, and number of heads. Consequently it does not establish that IU28
or the frozen stack is optimal for completed-answer detection, first-error
localization, and causal early prediction. “No GPU follow-up” below applies to
these three mechanisms, not to the broader research direction. The broader
token-native architecture search is separately frozen in
`docs/experiments/GLOBAL_LOCAL_ONLINE_ARCHITECTURE_V2.md`.

## Decision

**Retain IU28 without final length as the Online head; retain the frozen Global/Local heads; close the tested coarse-monitor dynamic family.** None of the three frozen dynamic candidates passes the early-panel superiority gate. Localization is mechanically unchanged (bit-identical score hashes), so there is no localization regression and also no localization gain to trade against the early result.

This is a development-only conclusion from existing caches. It does not authorize inference, a GPU/cluster run, a large download, or fresh-confirmation language. The correct claim remains **unsupervised scorer with calibrated decision policies**.

## Evidence inventory and independence

The inventory contains **113** cache/artifact records: **41** causal-prefix-valid, **1** localization-only, and **71** unusable for this cycle. The early screen has 11 cells but only five equal-weight dataset families; generator/model copies within a family do not create independent family evidence. ProcessBench is grouped by original question, and PRMBench remains a separate teacher-forced step task.

The Google Drive check was read-only and recorded path, size, and modification metadata. No Drive artifact was copied, moved, overwritten, or deleted.

## Localization panel (kept separate)

The frozen localization evidence remains:

- GL-LIU v1 ProcessBench macro F1: **0.3136**, versus **0.2571** for Mind the Gap.
- Fixed trajectory-first ProcessBench macro F1: **0.3070**; matched Qwen3-8B F1 **0.3035** versus **0.2496**.
- Fixed trajectory-first PRMBench step AUROC: **0.6711**, versus **0.6136** for the older step-first adapter.
- Online-only candidates reproduce the ProcessBench and PRMBench score hashes exactly: `9918dbffa15f49ef0c6e559adc6f96d48816825a5e0382b92dd7d744eb70738a` and `fbca592262868b52f8e5dd3ce93255e457caff8298a1d055ad3462e5bcbafe9f`.

Same-matrix graph controls remain tiny for ordinary/uniform/DUFS and harmful for the temporal detector arm:

| component | ordinary | uniform | DUFS | temporal |
|---|---:|---:|---:|---:|
| global answer AUROC | 0.7914 | 0.7920 | 0.7936 | — |
| local top-5 detector AUROC | 0.7233 | 0.7238 | 0.7239 | 0.6915 |
| exact locator rate | 0.2662 | 0.2668 | 0.2670 | 0.2641 |

Ordinary IU is the exact `lambda=0` path; the regression test confirms bit identity. These graph rows are historical same-matrix controls, not a new hyperparameter search.

## Early-ranking panel

Equal-family AUROC among unfinished traces:

| method | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---:|---:|---:|---:|---:|---:|
| `iu28_no_length` | 0.5096 | 0.5692 | 0.6317 | 0.6751 | 0.6176 | 0.6548 |
| `deepconf_entropy_w64` | 0.5397 | 0.5938 | 0.6072 | 0.6613 | 0.6603 | 0.7110 |
| `cusum_swvar_equal` | 0.5368 | 0.5811 | 0.6264 | 0.6740 | 0.6514 | 0.7018 |
| `dyn_level4_iu` | 0.5371 | 0.5811 | 0.6257 | 0.6709 | 0.6450 | 0.7022 |
| `dyn_persist6_iu` | 0.5377 | 0.5808 | 0.6200 | 0.6709 | 0.6368 | 0.6736 |
| `dyn_change6_iu` | 0.5365 | 0.5661 | 0.6030 | 0.6498 | 0.6261 | 0.6841 |

The frozen primary endpoint is the equal-family mean across 64 and 128 tokens. Paired question/family bootstrap results:

| candidate | reference | delta | 95% CI | family W/T/L |
|---|---|---:|---:|---:|
| `cusum_swvar_equal` | `iu28_no_length` | -0.0032 | [-0.0518, +0.0535] | 2/0/3 |
| `cusum_swvar_equal` | `deepconf_entropy_w64` | +0.0159 | [-0.0418, +0.0730] | 4/0/1 |
| `dyn_level4_iu` | `iu28_no_length` | -0.0051 | [-0.0553, +0.0519] | 2/0/3 |
| `dyn_level4_iu` | `deepconf_entropy_w64` | +0.0141 | [-0.0411, +0.0699] | 4/0/1 |
| `dyn_persist6_iu` | `iu28_no_length` | -0.0079 | [-0.0663, +0.0639] | 2/0/3 |
| `dyn_persist6_iu` | `deepconf_entropy_w64` | +0.0112 | [-0.0492, +0.0731] | 3/0/2 |
| `dyn_change6_iu` | `iu28_no_length` | -0.0270 | [-0.0979, +0.0561] | 1/0/4 |
| `dyn_change6_iu` | `deepconf_entropy_w64` | -0.0079 | [-0.0718, +0.0529] | 2/0/3 |

The least complex dynamic arm, `dyn_level4_iu`, is essentially a re-expression of the magnitude control and does not improve IU28: **-0.0051 [-0.0553, +0.0519]**, with wins in 2/5 families. Persistence and change/recovery coordinates also fail. DeepConf comparisons cross zero as well. Therefore no candidate is promoted.

The canonical elapsed-prefix-length arm remains an ablation, not part of IU28. The prior 11-cell result showed no stable reason to add it; this cycle did not refit or silently merge that feature.

## Convergence and declaration behavior

At the two primary budgets, rank correlation with each method's own completed score and final-decision agreement remain descriptive convergence metrics, not substitutes for discrimination:

| method | Spearman @64 | Spearman @128 | agreement @64 | agreement @128 |
|---|---:|---:|---:|---:|
| `iu28_no_length` | 0.4172 | 0.6585 | 0.6396 | 0.7387 |
| `deepconf_entropy_w64` | 0.4901 | 0.6801 | 0.5998 | 0.7692 |
| `cusum_swvar_equal` | 0.4231 | 0.6089 | 0.5330 | 0.6049 |
| `dyn_level4_iu` | 0.4110 | 0.5789 | 0.5305 | 0.5968 |
| `dyn_persist6_iu` | 0.5410 | 0.7196 | 0.5235 | 0.5835 |
| `dyn_change6_iu` | 0.2526 | 0.3075 | 0.6468 | 0.6379 |

Calibrated three-way declaration summaries (equal-family averages):

| method | coverage | ever wrong | selective error | mean decision budget | potential tokens remaining |
|---|---:|---:|---:|---:|---:|
| `iu28_no_length` | 0.3251 | 0.1182 | 0.3517 | 183.8111 | 551.8552 |
| `deepconf_entropy_w64` | 0.3970 | 0.1359 | 0.3232 | 168.6262 | 524.8375 |
| `cusum_swvar_equal` | 0.3349 | 0.1132 | 0.3389 | 177.6979 | 624.0496 |
| `dyn_level4_iu` | 0.3281 | 0.1136 | 0.3556 | 174.5587 | 617.5831 |
| `dyn_persist6_iu` | 0.3219 | 0.1147 | 0.3874 | 179.1783 | 583.5155 |
| `dyn_change6_iu` | 0.3510 | 0.1266 | 0.3520 | 153.4133 | 520.6952 |

Declaration behavior cannot rescue the failed ranking gate. All thresholds were fit from calibration labels; the score constructors did not see labels.

## Redundancy, missing streams, and cost

The dynamic heads are highly redundant with the equal CUSUM/`sw_var` magnitude control:

| method | equal-family Spearman vs magnitude control @64/128 |
|---|---:|
| `dyn_level4_iu` | 0.9926 |
| `dyn_persist6_iu` | 0.9490 |
| `dyn_change6_iu` | 0.7080 |

A missing component is deterministically replaced at its fitted reference level. Sensitivity of the 64/128 endpoint:

| method | missing stream | equal-family delta vs full | worst cell delta |
|---|---|---:|---:|
| `dyn_level4_iu` | `cusum_max` | +0.0018 | -0.0189 |
| `dyn_level4_iu` | `sw_var_peak` | -0.0259 | -0.1374 |
| `dyn_persist6_iu` | `cusum_max` | +0.0019 | -0.0157 |
| `dyn_persist6_iu` | `sw_var_peak` | -0.0333 | -0.1380 |
| `dyn_change6_iu` | `cusum_max` | +0.0178 | -0.0251 |
| `dyn_change6_iu` | `sw_var_peak` | -0.0464 | -0.1579 |

Measured Online-head cost (11 cells):

| method | retained features | state scalars/trace | median fit s | median score s | max Python traced peak |
|---|---:|---:|---:|---:|---:|
| `dyn_level4_iu` | 4-4 | 6 | 0.0223 | 0.0407 | 1.71 MiB |
| `dyn_persist6_iu` | 6-6 | 10 | 0.0212 | 0.0384 | 1.64 MiB |
| `dyn_change6_iu` | 5-5 | 8 | 0.0213 | 0.0379 | 1.64 MiB |

Each dynamic arm uses O(1) work and O(1) persistent state per new **monitor observation**. This benchmark does not measure upstream telemetry extraction or full IU28 stream-computation cost, so it cannot establish an end-to-end compute Pareto win. The available trajectories are saved at the existing absolute monitor grid; they are causal but are not a newly generated token-by-token recurrence.

## Candidate ledger and disposition

| candidate | hypothesis | delta vs IU28 (95% CI) | declarations vs IU28 (coverage / ever-wrong) | decision |
|---|---|---:|---:|---|
| `dyn_level4_iu` | running extremes retain early warning | -0.0051 [-0.0553, +0.0519] | +0.0029 / -0.0046 | close |
| `dyn_persist6_iu` | positive area and run persistence beat one-off magnitude | -0.0079 [-0.0663, +0.0639] | -0.0033 / -0.0035 | close |
| `dyn_change6_iu` | slope and failure-to-recover add information | -0.0270 [-0.0979, +0.0561] | +0.0259 / +0.0084 | close |

**Retain:** frozen Global/Local heads and `iu28_no_length` Online head.
**Close:** current/running-maximum, persistence/area, and slope/recovery transformations of the existing coarse CUSUM/`sw_var` trajectories. The failure mode is lack of independent signal: effects are small, intervals are wide and cross zero, family directions are inconsistent, and the simplest arm is almost perfectly redundant with the magnitude control.
**Do not promote:** graph regularization, elapsed length, or a declaration-only variant.
**Next gate:** no GPU or fresh-inference run is justified by this screen. Reopen only for a token-native causal recurrence or genuinely new telemetry/data, under a separately frozen protocol and explicit authorization.

## Audit trail

- Anchor regression: PASS.
- Suffix, feature-order, label-removal/permutation, repeated-run, missing-component, and exact `lambda=0` tests: PASS.
- New inference: no. GPU hours: 0. Drive mutation: no.
- A6/PTNI artifacts and protocol: untouched.
- All opened data are retrospective development evidence; no fresh confirmation claim is made.

Machine-readable outputs are in this directory, including `AUDIT.json`, `CANDIDATE_LEDGER.csv`, `AGGREGATE_PERFORMANCE.csv`, `AGGREGATE_DECLARATIONS.csv`, `GROUPED_INTERVALS.csv`, `PER_CELL_METRICS.csv`, `PER_QUESTION_SCORES.csv`, `PER_TRACE_CONVERGENCE.csv`, `MISSING_STREAM_AGGREGATE.csv`, and `EFFICIENCY_AGGREGATE.csv`.
