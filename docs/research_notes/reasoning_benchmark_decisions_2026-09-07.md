# Reasoning benchmark priorities and evidence behind the decisions

## User decisions, 2026-09-07

1. A stable, full, matched benchmark is the first research deliverable. Historical
   leaders and corrected refits are part of that deliverable, not optional context.
2. Analyze existing experiments alongside benchmark execution. Identify algorithmic
   gains that improve the actual application, rather than opening another sweep.
3. Focus the active claim on reasoning. RAG, retrieval grounding, claim/span
   detection and agent/tool-use tasks remain historical or deferred scope.
4. LOCA, Diverging Flows, KalmanNet and Shlezinger-inspired extensions are LOW
   priority. They remain supporting ideas for fusion, not replacement methods.
5. Preserve both feature fusion and observation selection/chronological processing.
   Single-answer fitting remains the primary arm; borrowed fits need a separate label.

This is a priority amendment, not a change to the frozen full-run scoring protocol.
No running scientific source, input or output was changed for this decision.

## What risk selection actually does

The implementation is `spectral_utils/fusion_window_sampling.py`, `budget`,
`top_indices`, and `choose_all`; the replication adapter is
`spectral_utils/fusion_sampling_replication.py`, `score_sampling`.

- Start with full nonoverlapping 8-token fitting windows from one answer.
- Build the existing 27-feature moment or context bank. In both banks column0
  is mean token entropy in that window. This is the risk proxy, not a calibrated
  probability of a reasoning error and not a label-trained risk model.
- With N fitting windows retain m = min(N, max(32, ceil(N/2))). Select the m
  largest column0 values; ties prefer earlier indices. Restore temporal order.
  For N<=32 nothing changes. Example: 800 tokens ->100 fitting windows ->50
  entropy-selected fitting windows. This is an illustrative count, not new data.
- Refit normalization/orientation and IU or Joint fusion on the selected rows
  in the original bank. Joint-fit failures use the declared selected-row IU fallback.
- Score ALL original windows, including unselected ones. The GMM is fitted on
  ALL original fitting-window scores, and official steps receive mapped scores.
  Selection is not sparse end-to-end scoring and does not reduce LLM inference.

Current110 corrected-label evidence: risk IU pooled PRMB AUROC .75835198 versus
full .68130595, within-answer .77892319 versus .76881338, and PB .28348525
versus .30159850. The pooled improvement is promising but is not a .758 first-
error localizer and is not the official PRMBench PRMScore.

The fixed affine diagnostic preserves full IU ranking and changes only answer
mean/SD, reaching pooled .748761. Sampled ranking on the old scale gives .689378.
This supports a substantial answer-scale contribution; it does not prove a
complete causal decomposition. Risk-selected equal+permuted graph remains a
mandatory strong control (.76842231 / PB .33920033); its PB difference against
its full counterpart has CI[-.03293,+.08029], so it is not a confirmed winner.

## TOKENS and TRAJECTORY are operations on the same axis

The matrix is N chronological observations by P features. FEATURES fusion
combines columns to produce a risk curve. TOKENS/windows describe its row axis.
Within that axis distinguish:

| Operation | Question | Existing examples |
|---|---|---|
| Observation selection | Which rows help estimate fusion weights? | full, uniform, entropy-risk, transposed DUFS |
| Chronological processing | How should evidence evolve over ordered positions? | IMM, HMM, Kalman, change-point methods |
| Multiple-curve combination | How should IU and Joint risk curves be combined? | static mean, GLS; followed optionally by IMM |
| Decision/readout | Which official step, or no error, is the final output? | peak plus mixture gate |

TRAJECTORY is not a third independent matrix axis. Static mean/GLS combine
curves without a sequential state; IMM adds chronological dependence.
Claude's sorted top10 within-step order statistics are a different observation
representation and do not preserve the chronological token sequence.

## Claude conversation reviewed

Read the actual recent local session, not only the August literature map:
`C:/Users/DELL/.claude/projects/C--Users-omris-TAU-hallucination-detection/f533e29f-e227-4acf-aa74-6910521c2726.jsonl`.
Relevant messages: user pasted the other-computer analysis at
2026-09-06T22:54:25.725Z; Claude's synthesis at 2026-09-06T23:11:17.867Z.
The old instruction inside that conversation to avoid file edits was historical;
the active user explicitly requests documentation now.

The synthesis flags relevant issues: mismatched historical scoring models,
PRMBench AUROC versus PRMScore, reproduction gaps for external controls,
teacher-forced scorer versus original generator, and new reasoning datasets.
Its literature numbers and statements such as an 'uncontested slot' are not
accepted as verified publication claims by copying the conversation.

Two qualifications to its recommendations matter:
- An open original generator is helpful for a generator-matched diagnostic,
  but is not mandatory for our separately disclosed proxy-scorer setting.
- A new scoring model on previously inspected ProcessBench answers is NOT an
  untouched dataset confirmation. MR-GSM8K also needs question/source overlap
  checks against GSM8K-derived development data before any independence claim.
  Synthetic edits can create style/splice cues; test this rather than asserting
  that they explain our measured PRMBench improvement.

## Reasoning benchmark filter

| Resource | Decision | Reason / remaining check |
|---|---|---|
| ProcessBench | CORE NOW | First-error/no-error task. Full same-population comparisons, clean/error breakdown and corrected source groups. |
| PRMBench | CORE NOW, separate panel | Dense step evaluation. Pooled and within-answer AUROC plus an explicitly specified native PRMScore adapter. Never compare .758 AUROC with published PRMScore75.8. |
| MR-GSM8K | First external candidate after locking method | Mathematical meta-reasoning transfer; verify data/step/generator contract and question overlap. MR-Score is not PB F1. |
| ReTraceQA | Later reasoning transfer candidate | Commonsense process evaluation. Public paper verified; executable data availability and exact label/telemetry alignment still need checking. |
| GR-Ben | Later reasoning transfer candidate | Science/logic process evaluation. Public paper verified; data release and exact evaluation adapter still need checking. |
| Hard2Verify | Later difficult-math stress test | Step-level frontier-math verification. Do not automatically promote it above the matched core; inspect annotation/access/population fit first. |
| Socratic-PRMBench / PRMBench-STEM / DeltaBench | Deferred compatibility audit | Potential reasoning stress tests from Claude's list; current data availability, versions and contracts not independently established in this pass. |
| PRM800K | Supervision/provenance resource | Do not declare training-source material an independent test without a verified split/overlap audit. |
| RAGTruth / KnowHalBench / RAG claim or span panels / agent tasks | Outside active claim | Different application and label/metric contracts; preserve old artifacts, do not dilute the current reasoning leaderboard. |
| Historical24 | Separate later transfer | Preserve the complete historical contract. Report reasoning strata explicitly rather than selecting favorable cells after seeing outcomes or silently changing the24-cell macro. |

Primary pages checked in this pass (existence and high-level task scope, not
all counts, availability or leaderboard claims from Claude):
- MR-GSM8K: https://arxiv.org/abs/2312.17080
- ReTraceQA: https://arxiv.org/abs/2510.09351
- GR-Ben: https://arxiv.org/abs/2605.01203
- Hard2Verify: https://arxiv.org/abs/2510.13744

## Parallel evidence analysis: initial synthesis and next actionable checks

This consolidates the reviewed current110 and bridged earlier58 experiments;
it does not claim an exhaustive new audit of every historical branch or paper.

| Evidence | Type of gain | Application implication / next analysis |
|---|---|---|
| Pair-safe grouping and additional residual banks raise native-fit coverage but do not improve both endpoints | Numerical availability, not demonstrated detection gain | Retain coverage/fallback records; do not promote on convergence alone. |
| Condition100 + graph improves original Joint points, with strong equal/permuted controls | Candidate algorithm gain relative to own parent | Isolate conditioning, graph identity and learned weights on the full matched population. |
| Risk selection substantially raises pooled PRMB with small within-answer movement and lower PB | Predominantly scale-related evidence, plus possible local gain | Separate within/between-answer effects, error types, length and retained-error support. Keep simple risk controls. |
| GLS has small positive points but intervals include0 | Unresolved fusion gain | Full paired comparison and error overlap before further temporal complexity. |
| IMM damages PB and its calibrated short-jump sensitivity fails the fixed screen | Negative result with identifiable decision tradeoff | Keep fixed control; do not automatically launch a new calibration sweep. |
| Corrected labels/group folds change interpretation of prior results | Evaluation repair | Historical fitted methods need corrected refits; do not count repairs as algorithmic accuracy improvements. |

Evidence files: `results/fusion_sampling_replication_v1/REPORT.html`,
`results/fusion_trajectory_imm_v1/REPORT.html`,
`results/fusion_gate_calibration_v1/REPORT.md`,
`results/localization_history_bridge_v3/REPORT.html`, and
`results/localization_prm_label_audit_v1/REPORT.html`.

The next analysis deliverable is a linked evidence ledger per unique recipe:
parent method, cohort/labels/fit scope, changed component, paired effect,
within-answer evidence, clean/error outcomes, coverage, runtime and decision.
Do not add variant counts across repeated controls or cohorts. Analyze saved
scores while the benchmark runs; a new fit/sweep is a separate decision.

## Sampling proposal from Omri: high and low entropy, 2026-09-07

Omri points out that fitting only high-entropy windows learns feature behavior
conditional on that region of the trace, and proposes25 high plus25 low instead
of50 high. This is a plausible selection-bias concern: changing the selected
rows changes normalization and estimated feature relationships. However entropy
is an unlabeled proxy; high/low entropy must not be called hallucinated/correct
classes, and equal tail counts are not class balancing.

Register for the next sampling comparison, AFTER full-benchmark priority:
- Existing high-only selection at the same fixed budget.
- Omri's high/low-tail selection (e.g.25+25 for budget50).
- Quantile-stratified selection across the entire entropy range (e.g. five
  representatives from each of ten equal-count rank strata for budget50).
- Existing full-grid and uniform-in-time controls.

High/low tails recover range but omit the middle and can overrepresent extremes;
quantile stratification addresses distribution coverage more directly. Neither
is a proven better fusion fit. Restore chronological order, use no labels, fit
inside the same answer, and score every original window. Keep the source of
normalization explicit: changes to scale versus fusion weights require separate
attribution. Do not silently compare new full-data normalization with the old
sample-normalized baseline as if only selection changed.

The active registered selectors contain full/uniform/risk_top/dufs_transposed/
dufs_permuted/window_diffusion. A targeted code/protocol search did not find
an implemented high/low or entropy-stratified fitting-row selector in this
current study; this is not proof that no related historical idea ever existed.
No new selector was executed and the frozen full anchor run remains unchanged.

Parallel existing-evidence work now has an actual machine-readable inventory:
`results/localization_evidence_ledger_v1/CURRENT110.csv` and `CURRENT110.json`.
It covers176 displayed entries, with coverage and both PRMB metrics/PB decisions,
and167 exact saved-output fingerprints. Nine duplicate-output groups are made
explicit. Fingerprint equality applies only to saved final scores/validity/
decisions on current110, not algorithm equivalence. This inventory introduces
no new fit, significance result or winner; earlier58/historical/full-population
contracts remain separate additions rather than claimed completed analysis.

## Execution authorization update

Omri subsequently explicitly asked to run this comparison in the background.
Step324 now runs the bounded current110 study under
`docs/experiments/FUSION_ENTROPY_SAMPLING_V1.md`. This supersedes the earlier
sampling deferral for this experiment only. Main full benchmarking continues.
The protocol compares empirical mid-quantiles (e.g. five per decile atN100/m50)
with equal low/high tails and unchanged full/uniform/high-only controls.
See PROGRESS.md and results/fusion_entropy_sampling_v1/RUN_STATE.json for status.

## Step324 completed: interpretation

**Step324 COMPLETE - entropy sampling comparison; review PASS.**
`results/fusion_entropy_sampling_v1/REPORT.html` contains the matched35-row
comparison and31 registered paired intervals. Same110 development answers:
24 PRMBench/86 PB-Qwen3-8B;72 sampling eligible,38 exact full-window replays.
All176 prior entries remain unchanged;14 additions,190 displayed entries.
Run handle24353 is terminal, exit0. Scoring453.83s; no new inference.

IU pooled PRMB / within-answer PRMB / PB score:
full .68131 / .76881 /30.16%; uniform .67643 / .77697 /28.34%;
high-only .75835 / .77892 /28.35%; low/high .72746 / .77174 /26.27%;
entropy quantiles .71523 / .79198 /20.77%.
Joint condition100 graph0.1: full .65545 /30.22% (pooled/PB),
high-only .74133 /28.20%, low/high .71439 /26.36%, quantiles .67024 /24.37%.
All14 new outputs have24/24 PRMB and86/86 PB coverage INCLUDING IU fallback;
Joint fallback20/110 tails,19/110 quantiles (18/72 and17/72 among new fits).

Interpretation: neither new selector dominates the existing references.
Quantile IU improves within-answer AUC vs full by .02316 (exploratory paired
95% CI [.00213,.05232]), but PB drops9.38 percentage points
(CI [-17.51,-2.56] points). Its within-answer gain over high-only is uncertain.
Low/high tails do not beat high-only on either primary point metric for IU
or Joint graph. This does not prove high entropy identifies errors or resolve
normalization-versus-weight attribution. Keep fusion central and full matched
benchmark/refits first; no automatic new sweep or promoted publication winner.

Review:220 selections,1008 linear scores,1540 dense readouts,six representative
shared-kernel bank refits and14 metric bundles PASS. Additional explicit-pair
within-answer arithmetic for14 methods,35 HTML/CSV metric rows,31 interval rows
and six local links PASS. Same-session review, not external/browser validation.
Scientific driver/core/protocol stayed frozen. The priority HANDLE workaround
remains an operational launcher only. Main full-benchmark handle50440 continues;
its full evaluation and historical comparisons are still pending.


## Step325 status and metric-gap reasoning

**Step325: full anchor scoring COMPLETE; evaluation RUNNING.**
Scoring exec50440 ended successfully after19425.09s (5h23m45s), at19:57:17
Israel time on2026-09-07. All13769 rows and19 original anchor methods are
frozen. This did NOT include the newer condition100/sampling/trajectory
shortlist or corrected historical refits. No automatic evaluation stage was
connected to the scoring driver; the run stopped at its declared score freeze.
The missing adapter was implemented on Omri's status follow-up, not during
those5.4 hours of scoring. Do not imply continuous research execution after
scoring completion or call the historical comparison complete.

Evaluation exec65524 is now live. It verifies all27538 score-file hashes,
joins corrected labels/groups, replays110 pilot outputs/19 metric bundles,
then computes full PRMB/PB metrics and1000 joint-source bootstrap draws.
All3483 canonical groups are drawn together, including66 cross-task groups
and repeated Q4/Q8 scorers. Fixed26 contrasts; no new model/scorer fitting.
Protocol: docs/experiments/FULL_ANCHOR_EVALUATION_V3.md.
State: results/localization_full_benchmark_v3/evaluation/RUN_STATE.json.
Scientific completion is pending its metrics/interval/review report.
Read-only second-agent code/contract inspection found no blocking arithmetic
bug; source-arm totals must not be labelled native-valid coverage. Supplemental
resume-provenance and stratified source/validity checks remain review items.

**Step324 metric-gap diagnosis:** frozen-score inspection, no new fit.
Quantile IU vs full loses six correct PB decisions and gains none: four
correct peaks move to wrong steps, one correct peak is suppressed by its gate,
and one clean answer becomes a false alarm. Raw exact peaks20/53 ->16/53;
final correct error decisions12/53 ->7/53; clean decisions18/33 ->17/33.
With original full-IU gate and new peaks, PB is22.24% vs30.16% baseline
(native quantile20.77%). This is a post-hoc output swap, not a new candidate.
PRMB uses different answers; its within-answer gain does not imply better PB
ranking. PB-only first-error-versus-correct-prefix ranking falls .7094 ->.6911
on44 eligible erroneous answers (nine step-zero errors excluded; post-error
steps remain unlabeled). Do not dismiss PRMB gain as only calibration, and do
not attribute the whole PB loss to the gate. Continue full benchmark first.
See results/fusion_entropy_sampling_v1/METRIC_GAP_DIAGNOSTIC.html and JSON.


## Step325 completion and full-population interpretation

**Step325 COMPLETE: full19-anchor evaluation and supplemental review PASS.**
Scoring exec50440 ended after19425.09s (5h23m45s), at19:57:17 Israel time.
The evaluation adapter had not been connected then; it was implemented in
this status follow-up. Eval exec65524 is now terminal, exit0,454.13s including
verification/joins/metrics/intervals. Supplemental review exec56541 also exit0.
No active scorer/evaluator remains from these three handles.

Full report: results/localization_full_benchmark_v3/evaluation/REPORT.html.
All13769 rows /19 original anchors; all27538 frozen score-file hashes checked.
110 pilot answers and19 metric bundles replay exactly (float tolerance only
for aggregate arithmetic).1000 joint draws across3483 canonical groups keep
repeated scorers/answers and66 cross-task links together;19 absolute interval
bundles and26 fixed paired contrasts. All are development/exploratory evidence.
Supplement verifies joined provenance,nine label hashes,261611 method-validity
records,363 per-cell source/validity groups,47 report rows and five links.
Code/contract independently inspected by a read-only second agent; no external
scientific or browser validation. Scientific scoring files stayed frozen.

Full routed anchors (PRMB pooled / within-answer / PB-Q8 four-cell macro):
IU .668225/.707230/20.3828%; equal .665479/.702973/20.7035%;
Joint0 .668519/.706393/21.0893%; graph .667464/.706779/21.4172%;
permuted graph .669289/.707787/21.3452%.
PRMB coverage6952/6969, mixed6021;17 short PRMB scores unsupported. PB keeps
all3400 answers per scorer including two short unsupported MATH answers.
IU PB-Q4 is19.6252%, eight-cell macro20.0040%. These Joint arms retain the
ORIGINAL condition1000, not the current condition100 candidate. Full Q8 graph
minus IU difference+1.0344pp has95% CI[-1.0591,+3.0487]pp; no clear advantage.
Graph pooled PRMB is slightly lower than lambda0 and node-permuted controls;
no demonstrated graph mechanism benefit. Pilot PB points were more favorable
than the full population, not a refitting bug: matched pilot outputs replay.

Next still required: full fixed condition100/graph/equal/sampling/trajectory
shortlist, corrected-fold historical fusion/localizer refits and declared
access panels. No new shortlist run was silently launched in this diagnostic
stage. The historical benchmark is NOT complete and no method is promoted.
The method registry marks only stage1 FULL_EVALUATION_REVIEWED.

