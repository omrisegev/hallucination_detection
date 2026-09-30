# Full localization sampling comparison v3

This extends the existing full development benchmark; it does not select a
new method from a pilot. User policy: small runs check code and feasibility
only. Comparative findings use all 13,769 registered model-answer records.
All cached answers are exposed development data. Untouched confirmation
requires a later locked recipe and separate data.

## Fixed population and access

Use exactly the selected records from `localization_full_benchmark_v3`, the
v3 PRMB one-based label conversion, canonical source groups and FOLDS_V2.
Preserve the original v1 scoring identity namespace and graph seeds. No new
inference: one cached teacher-forced model pass over each provided answer.
Primary normalization, selectors, groups, gates and fusion weights use only
that answer's telemetry. Ground-truth labels enter evaluation only.

Each answer uses the original 8-token window plan, original moment/context
bank route, feature definitions, polarity rules, and dense step readout.
Fusion remains central. Selected windows supply fitting rows; all windows
are scored and all original fitting windows supply the unchanged GMM gate.
This is not yet sparse feature extraction or a causal online localizer.

## Frozen method roster

Eight selectors: full, uniform, risk_top, dufs_transposed, dufs_permuted,
window_diffusion, entropy_tails and entropy_quantiles. Non-full budget is
`min(N,max(32,ceil(N/2)))`; full uses N. Tails allocates floor(m/2) low and
the rest high mean-entropy windows. Quantiles covers the entropy ranks.
Equal entropy has deterministic time-index tie breaking. High entropy is
an observable heuristic, not an error label or proof of class membership.

Seven cores per selector: equal, IU, Joint lambda0, Joint-LIU lambda0.1,
node-permuted Joint-LIU, equal-graph and node-permuted equal-graph. These are
the exact previously implemented sampling recipes, including condition100
and explicit sampled-IU fallback when the Joint model fit fails. Numerical
graph/readout failures remain failures. Unsupported short traces remain
unsupported; do not omit them from PB's denominator.

The 56 sampling entries include seven full-window aliases and 49 additions.
Preserve all 36 full-shortlist entries and five completed historical controls
in the report: 97 displayed entries, not 97 independent algorithms. Historical
controls fit other outer-training answers and use nested training-label PB
calibration. Separate their access explicitly. The ongoing historical Joint
extension and remaining comparator registry are separate unfinished blocks.

## Mechanism distinctions

Feature grouping uses feature nodes. Joint-LIU builds a graph on fitting
observations (windows here; training tokens in the historical comparator),
then uses `Z.T @ L @ Z / N` to penalize feature weights. It retains all rows.
Transposed DUFS instead learns gates for window coordinates to choose fitting
rows. Window diffusion also selects observations. Similarity edges do not
encode chronological adjacency by themselves.

The exact prior 93 sampling and 31 entropy contrasts are retained; their
overlap is deduplicated. Additional corresponding controls cover the new
entropy selectors. The full fixed contrast list is serialized in MANIFEST.
Compare each sampled core with its full counterpart, IU with equal, graph
with lambda0/permuted/equal-graph controls, and DUFS with uniform/risk/permuted
sampling. Eligible-only and both-native populations are conditional
diagnostics, never substitutes for the full benchmark.

## Evaluation and review

PRMB reports fixed-fold mean AUC (historical convention), pooled AUC for
continuity, and macro within-answer AUC with coverage. These are not official
PRMScore. Pooled AUC remains the registered answer-only endpoint; adding the
fold-mean column for historical comparison does not silently replace it.
Within-answer and PB results are required when judging localization gains.
PB reports exact first-error/no-error harmonic accuracy per cell,
Q4/Q8 four-cell macros and all-eight macro. Every invalid PB decision fails.
Also report raw exact peak accuracy and correct peaks suppressed by the gate.

Use 1,000 paired canonical-source bootstrap draws with the existing seed
2026090707; one draw spans tasks, scorers and fixed folds. PRMB paired
comparisons use common valid rows. PB uses the entire stated population.
Intervals condition on saved predictions, are exploratory, and are not
adjusted for many comparisons or subsequent winner selection. A two-task
point lead alone does not establish a publishable winner.

Report selected-window support for first errors, including <=32-token errors,
and PRMB error steps on sampling-eligible records. This is fitting support,
not sparse-scoring recall. Preserve original DUFS/selector perturbation
stability diagnostics; tails/quantiles have no perturbation result under
that older contract. Report runtime with diagnostic overhead disclosed.

Before full scoring, replay 110 previous answers times 56 method bundles
(6,160) plus shortest/longest full-data traces. This validates implementation,
not scientific improvement. Freeze code, inputs and protocol with hashes.
After scoring: validate joins, score/decision coverage, exact full-window
aliases, all historical/shortlist metrics, independent fold-AUC arithmetic,
direct PB counts, interval point arithmetic, HTML rows and local links.

## Execution

`preflight_full_sampling_v3.py` performs the replay. The full driver
`run_full_sampling_v3.py` uses two workers, a single-invocation OS lock,
atomic per-answer checkpoints and an eight-hour submission cap. It drains
active answers at the cap; resume the same driver/output after it exits.
No duplicate live driver. Preflight scores are adopted with source hashes.

Execution-only acceleration replaces small-partition ARI arithmetic inside
each worker's imported Joint module, restoring it afterward. No source file
or sklearn installation is modified. `preflight_full_sampling_fast_v3.py`
must match every array and every non-timing scientific diagnostic from the
112 plain preflight records exactly, plus independent ARI arithmetic checks,
before the full driver uses it. Timing comparisons disclose machine contention.

Evaluation is attached automatically after complete scoring and reviewed
pass2a availability. If pass2a is still pending, record that dependency and
resume with `--phase evaluate`. No partial performance ranking is published.
Do not modify live frozen scoring dependencies or Claude's worktree.
