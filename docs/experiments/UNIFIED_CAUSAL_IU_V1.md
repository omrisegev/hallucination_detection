# Unified Causal IU-PCR v1 — frozen development protocol

**Status:** implemented CPU protocol for supervised development on historically opened
ProcessBench telemetry. It is not an untouched confirmation experiment and authorizes no
cluster mutation.

## 1. Estimand and single-score contract

One stateful causal process consumes token telemetry in generation order. At token `t`, a
frozen IU-PCR head emits instantaneous standardized evidence `e_t`; one frozen accumulator
emits `R_t`.

- Global detection targets final-answer wrongness and uses `R_T`.
- Early detection uses the same `R_t` at budgets 16/32/64/128/256/512 and the first warning
  crossing.
- Localization uses the token with the largest positive injected contribution to the
  accumulator update. First crossing and a persistent three-token crossing are ablations.
- `finalize().global_score` is the already stored final online risk. It is not recomputed.

Labels may select the roster, signs, and accumulator inside development folds. The frozen
online method receives no label or future field. The paper description must therefore say:
**supervised-developed, IU-PCR-fused, causal streaming method**, not fully label-free.

## 2. Data and independent unit

Primary development uses the intersection of the four ProcessBench families (`gsm8k`,
`math`, `olympiadbench`, `omnimath`) with Qwen3-4B and Qwen3-8B scorer telemetry. The
dataset-qualified source question is the independent group, and both scorer copies always
remain in the same fold. Llama-3.1-8B is a robustness model and is not part of primary roster
selection. All these labels were previously opened.

After nested evaluation, the most frequently selected outer-fold accumulator (ties follow
the frozen simplicity order) is refitted with a newly developed roster on all opened primary
rows. Without any further tuning, that one frozen model scores the registered eleven-cell
Early panel: MATH-500 Qwen2.5-Math-7B T=1.0 plus ten ProcessBench generator/family cells.
This panel reports Global/Early robustness only and never changes selection or the promotion
gate.

Each row must contain token entropy, spilled energy, token log-sum-exp, top-k log
probabilities, step-token spans, final-answer correctness, and ProcessBench first-error label.

## 3. Causal feature contract

The 37 base streams are the existing nine primitive risk-oriented channels plus causal
versions of the historical broad-28 token views. The historical broad reconstruction cannot
be used verbatim: it backfills the first complete window into earlier tokens and centres
CUSUM by the completed-trace mean. Unified v1 instead uses the observed prefix for short
windows and an online-centred CUSUM. This deliberate change is required by suffix invariance.

Every standardized base stream receives these 28 transforms:

1. level;
2. EWMA spans 4, 8, 16, 32, 64 (`alpha=2/(span+1)`);
3. fast-minus-slow 4/16, 8/32, 16/64;
4. trailing mean, population variance, and MAD over 8, 16, 32, 64, using the available
   prefix when `t < window`;
5. innovation relative to the previous span-64 EWMA;
6. normalized cumulative positive area and positive persistence;
7. one-sided CUSUM with drift 0.25 and Page-Hinkley with drift 0.05;
8. conjugate Gaussian BOCPD change probability at hazards 1/50 and 1/100, with a fixed
   run-length state cap of 128.

This yields 1,036 coordinates before selection. Final trace length, elapsed/final-length
adapters, future windows, suffix features, and lookahead are forbidden by name and by test.
Missing observations map deterministically to the frozen base reference (standardized zero).

## 4. Information Atlas and supervised development

Every source trace contributes the same number of observations. Global/Early samples use
the six prefix budgets (Global also adds the terminal point); Localization uses 32
deterministic positions and guarantees inclusion of the annotated first-error span when one
exists.

For each coordinate and target, report:

- descriptive mutual information;
- mean test-fold log-loss improvement from adding the coordinate to a depth-limited
histogram gradient-boosting model containing budget/dataset/model context plus entropy,
  `sw_var`, and a fold-fitted IU28-without-length score;
- the four family-specific held-out gains;
- a source-question block-permutation p-value and Benjamini-Hochberg FDR.

The grouped permutation statistic is mutual information; the primary magnitude remains
held-out nonlinear log-loss gain. Publication runs use at least 200 permutations. Transform
families and base-stream families also receive joint held-out gains and synergy beyond their
best single member.

A coordinate or family passes if its conditional gain is positive in at least three
ProcessBench families and grouped-permutation FDR is at most 0.10. The roster is the union of
target-specific passes. Exact duplicates and pairs with `|rho| >= 0.98` are removed only
inside the development fold; the higher conditional-gain member survives. There is no
feature-count cap. If fewer than three features pass in a diagnostic/smoke run, the runner
uses a stamped top-three identifiability fallback; such a run cannot support a research
claim.

One sign per retained coordinate is frozen from an equal-target/equal-family robust effect
vote. A supervised nonlinear model over the selected bank is evaluated only as a diagnostic
ceiling, never as the proposed method.

## 5. IU-PCR and accumulators

Ordinary IU-PCR is fitted once per fit fold on 32 equally spaced rows per trace, after
median/IQR scaling. It uses L2, exactly two components, no exclusion, no difficulty gate,
no simple-average fallback, and the established scale ratio 0.25. The roster, signs,
references, order, robust scales, and IU weights are frozen at every future time point.

DUFS-LIU-PCR is a same-matrix fusion ablation, not a separately selected feature method.
It receives the exact retained roster, scaling, signs, 32 positions per trace, accumulator,
and two-component IU subspace used by ordinary IU. Parameter-free adapted DUFS learns one
unlabeled soft feature metric on the fit partition only (seeds 11/23/37, 80 epochs, k=7),
which constructs the sample-neighbourhood graph. Only the final projected IU equation gets
the trace-matched Laplacian penalty. `lambda=0` must reproduce ordinary IU bit-for-bit.

The frozen DUFS primary is `lambda=0.1`; `0.3` and `1.0` are the bounded aggressive
sensitivity arms. They are never selected on an outer test fold. Lambda is selected from
inner-fold results by the worst delta across Global, Localization and Early, with ordinary
IU represented explicitly as `lambda=0`; therefore an aggressive arm cannot purchase one
task's gain with a regression on another. The fixed 0.1 row and every aggressive row are
reported even when ordinary IU remains selected.

Accumulator roster:

- identity: `R_t=e_t`;
- leaky recovery: `R_t=max(0, exp(-1/span) R_(t-1) + e_t - drift)` for spans
  8/16/32/64 and drift 0/0.25/0.5;
- irreversible hazard control: `h_t=max(0, 2 sigmoid(e_t)-1)` and
  `R_t=R_(t-1)+(1-R_(t-1))h_t`.

The positive localization contribution is `max(0,e_t)` for identity,
`max(0,e_t-drift)` for leaky accumulation, and the newly injected hazard mass for the
irreversible control.

## 6. Selection, thresholds, and metrics

The accumulator is selected first by nested source-question-grouped CV using ordinary IU.
Roster selection, signs,
base references, IU28 conditioning, and IU-PCR fitting are repeated using only each inner
fit partition; its validation questions are used only to score accumulator arms. Any arm regressing more
than 0.010 Global AUROC, 0.010 ProcessBench Localization F1, or 0.015 mean Early AUROC at
64/128 relative to identity is rejected. Survivors maximize their worst task delta; ties
prefer identity, then leaky, then hazard, and finally lexical order.

After the accumulator is frozen, the same inner partitions compare ordinary IU with the
three DUFS lambda arms. This ordering deliberately prevents DUFS from winning through a
different accumulator. The most frequently selected outer-fold accumulator and graph
lambda (ties choose the smaller lambda) are refitted for the eleven-cell robustness panel.

Warning thresholds use only clean calibration traces and the maximum risk over the complete
monitored horizon. The 95th and 90th percentiles are the 5% and 10% false-warning operating
points. Localization clean/error threshold is calibrated separately for ProcessBench F1.

- Global: family-macro AUROC/AUPRC and fixed-FPR operating points.
- Localization: family-macro ProcessBench F1, exact, within one, clean abstention.
- Early: family-macro AUROC/AUPRC at all six budgets, alarm time, ever-warning FPR.
- Efficiency: per-token update time and fixed state size.

AUROC/AUPRC are computed inside each untouched outer test fold and then averaged. OOF
probabilities from separately fitted folds must never be concatenated for AUROC.

Live access-matched baselines are IU28 without elapsed length, running mean/max entropy,
max-entropy with the registered per-step top-five-mean locator, `sw_var`, and the
DeepConf-w64 entropy proxy. The prior one-shared, two-head, and length-free Global IU-PCR
heads are refitted inside each model/family cell of every outer training fold; their
original cell-local calibration and registered prefix-budget formulas are replayed. Their
old artifacts are provenance references only and are not mixed into new fold metrics.
The length-free Global replay is the direct comparison with the classic mixed-v2
30-feature contract (up to 29 eligible coordinates after the forbidden final-length
coordinate is removed); it is refitted and rescored on the same outer folds rather than
copied from the historical 0.7895 development result.
Because the v2 artifacts registered only monitor-budget scores, exact warning-horizon
comparisons are reported only for streaming baselines with full trajectories; Global,
Localization, and six-budget Early primaries remain directly recomputed.

## 7. Required tests and gates

Before any full run:

- suffix invariance;
- tokenwise/chunked identity;
- no final length/future-window names;
- bit-identical terminal and Global score;
- group/split isolation;
- grouped label permutation removes conditional signal;
- feature-order invariance;
- spike, drift, sustained error, and recovery scenarios;
- deterministic missing-channel behavior.
- DUFS `lambda=0` identity, same-roster identity, bounded positive lambda validation, and
  maximin lambda-selection behavior.

A development finalist must remain inside all three regression margins relative to the
strongest live baseline for each task and improve at least one task by 0.010. New inference
is considered only after 2,000 paired source-question bootstrap replicates (scorer copies
resampled together, family metrics computed inside each outer fold) give a positive 95%
interval against that task's strongest live baseline, without regression on the other two.
Any later confirmation requires a newly generated dataset/model family and a separately
frozen plan.

## 8. Commands and expected artifacts

```bash
python scripts/test_unified_causal_iu.py
python scripts/run_unified_causal_iu_v1.py preflight --data-root /path/to/repo
python scripts/run_unified_causal_iu_v1.py atlas --permutations 200 --workers 16
python scripts/run_unified_causal_iu_v1.py full --permutations 200 --bootstrap-repeats 2000 \
  --dufs-lambdas 0.1,0.3,1.0 --dufs-epochs 80 --workers 16
```

`--candidate-limit`, `--max-questions-per-family`, `--skip-synergy`, `--skip-robustness`,
`--skip-dufs`, fewer than 80 DUFS epochs, fewer than 200
permutations, and fewer than 2,000 bootstrap replicates are debug/smoke controls and are
stamped into `RUN_DEFINITION.json`; results using them are non-reportable. The runner is
resumable at completed outer-fold JSON files only when its matching full-trajectory NPZ is
also present.

Artifacts: inventory and hashes, frozen run definition, Information Atlas, group synergy,
roster/sign/redundancy audit, nonlinear ceiling, per-fold frozen models and ledgers,
compressed full token trajectories, per-question scores, paired-bootstrap intervals,
decision gates, baseline tables, the eleven-cell robustness inventory/scores/trajectories,
nested-CV summary, and `REPORT.md`.
