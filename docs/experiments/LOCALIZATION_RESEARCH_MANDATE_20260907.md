# Localization research mandate and evidence ledger

## Step330: full shortlist complete; all prior sampling selectors running

The 36-entry full answer-only shortlist is reviewed on13,769 rows. The
IU+Joint-graph mean has a small positive PRMB within-answer difference, but
no clear PB gain; IMM improves the PB point while reducing PRMB. No consistent
two-task winner is established. See the full report under
results/localization_full_shortlist_v3/evaluation/ and current PROGRESS.md.

All eight prior sampling selectors now run over the complete benchmark,
crossed with seven existing fusion cores, with unchanged answer-only fitting.
The eventual report retains36 shortlist entries plus five corrected
historical controls and142 paired comparisons. Full preflight and exact
execution acceleration replay pass; they establish code fidelity only.
Historical Joint resumed its same frozen run after238/245 fits at the cap.

Full error accounting separates early/late peaks, suppressed correct peaks
and clean false alarms. It reloads the historical PB top10 readout rather
than treating PRMB spanmax scores as its decision curve. Both peak ranking
and the no-error decision remain material bottlenecks. Continue development
from full matched evidence while preserving the feature/trajectory fusion
method, comparator registry and untouched-confirmation requirement.

## Step329: full historical first controls reviewed; Joint extension executing

The corrected five-control historical panel is complete on all13,769 rows:
results/historical_fusion_refit_v3/REPORT.html. Historical IU34.29% and
fixed-family CONT34.87% PB-Q8 retain the old performance level; equal active23
is33.92%. These are pooled fits with nested label-calibrated PB decisions.
They are explicit comparators, not the primary unsupervised answer-only
method. The latter's IU20.38% PB-Q8 remains the comparison anchor and has
higher PRMB within-answer AUC (.70723 versus historical IU .69925).
No consistent two-task gain over the equal control has been established.
The complete ten-arm historical Joint extension is now running with exact
control reuse, source-fold isolation and reviewed ARI execution acceleration.
See PROGRESS.md for handles and docs/experiments/HISTORICAL_JOINT_REFIT_V3.md
for the frozen roster. Full sampling and other registry families remain open.

## Superseding instruction: feasibility pilots only (Step328)

Omri's 2026-09-07 instruction replaces previous pilot-based decision stages.
Use subsets only to check feasibility, implementation, runtime or numerical
behavior. Research comparisons and candidate selection require the complete
matched development benchmark, historical comparators, coverage/failures,
per-cell metrics and paired uncertainty. Neither an optimistic pilot nor a
full exposed-development score establishes publication generalization.
Confirm a frozen candidate on untouched data later. Full evaluation retains
the single-answer fitting objective and does not authorize silent pooling.
Current pass2a is preserved; historical refits and full sampling remain open.


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
remains an operational launcher only. Full scoring handle50440 and evaluation handle65524 are terminal;
full anchor evaluation passed and historical comparisons remain pending.

## Omri decision update - 2026-09-07: reasoning benchmark first

The full matched REASONING benchmark and corrected historical-leader refits
are priority1 and a research-method requirement, not an optional final report.
Analyze existing experiments in parallel to identify real algorithmic gains
and application gains (local ranking, first-error/no-error decisions, coverage
and runtime). Do not open another sweep merely because a pooled AUC improved.
LOCA, Diverging Flows, KalmanNet and Shlezinger-inspired extensions are LOW
priority supporting backlog. Fusion remains central; answer-only fit is primary.

ProcessBench and PRMBench are the active core in separate metric panels.
RAG/grounding, claim/span and agent tasks are outside the active claim; preserve
historical artifacts. MR-GSM8K is the first external candidate to audit after
method lock, including GSM8K source overlap; ReTraceQA/GR-Ben are later transfer
candidates pending executable data contracts. A new scorer on previously seen
PB answers is not untouched confirmation. Preserve historical24 as a separate
later transfer with predeclared reasoning strata and its unchanged full macro.

Risk selection means highest mean-entropy8-token fitting windows, not known
errors: m=min(N,max(32,ceil(N/2))). Refit on selected rows and score ALL rows.
TOKENS and TRAJECTORY refer to the same observation axis: distinguish row
selection, chronological processing and combination of multiple fusion curves.

Reasoning, source conversation and initial gain analysis are recorded in
`docs/research_notes/reasoning_benchmark_decisions_2026-09-07.md`.
This amendment does not modify the frozen Step321 scoring protocol or run.

## Active priority update from Omri - 2026-09-07

Full matched localization benchmarking now takes priority over additional
small development sweeps. See `docs/experiments/LOCALIZATION_FULL_BENCHMARK_V3.md`
and `results/localization_full_benchmark_v3/METHOD_REGISTRY.json`. The unchanged
answer-only anchor pass has started on13769 model-answer rows; this is not
the completed shortlist/historical comparison. Keep IU/LIU/L-SML/Joint and
historical dedicated localizers in the continuing registry. Corrected-label
and source-fold refits, coverage and within-answer evidence are required.
See PROGRESS.md for live handles. Full cached data remain development data.

Date: 2026-09-07. Status: active research; no new winner established.

**Step319 completed: controlled mixture-gate mechanism check.**
`results/fusion_gate_null_v1/REPORT.html`; scientific and artifact review PASS.
768 synthetic trials: N16/64/256, AR rho0/.6/.9,64 replicates per cell,
plus rho0/+3SD mean-jump controls. Raw, ordinary Kalman cold/warm and IMM
cold/warm;3840 valid outputs, no failures. Warm processes256 preceding
synthetic points and is a startup diagnostic, not a free-data candidate.
No current110 score/prediction/metric changed; all176 anchors remain current.

One stationary Gaussian source can trigger the gate after filtering:
N256/rho0 raw0/64, IMM cold26/64, warm26/64; N64/rho0 raw0, cold11, warm12.
The effect survives this startup control. N16/rho0 raw already opens15/64,
so short-sample behavior also matters. At N64 with the fixed mean jump,
raw opens21/64 while both IMM variants open64/64: preserve sensitivity in
any correction. These are simulation frequencies, not real-answer false-
positive rates. BIC has no declared5% alarm guarantee; nonlinear filtering
need not preserve a Gaussian marginal. No numerical GMM bug is claimed.

Three tests;768 source paths,1536 independent scalar Kalman trajectories,
3840 normalization/mixture-algebra checks,48 direct vector IMM trajectories
and120 actual GMM refits pass. Simulation106.04s, review12.06s. Artifact
validation1551 hashes,11 local links/images,137 static rows,five ASTs. Both
PNG plots inspected; no browser rendering or external review claimed.

Next bounded stage: verify one fixed calibration procedure on independent
synthetic calibration/evaluation replicates, carrying the source through the
SAME filtering/normalization/GMM procedure. Include mean-jump sensitivity
and nuisance-parameter uncertainty; do not treat an AR Gaussian source as a
proven model of correct reasoning. Then a gate-only current110 comparison
must keep frozen fusion curves/peaks and all176 anchors. A synthetic fix
alone is not an achieved localization improvement. The full research scope
remains active: Joint/feature/IU, named supporting methods, corrected multi-
answer refits, full comparators, sparse/short-error sampling, untouched
confirmation and historical24 transfer are still open.

**Step318 completed: full-trajectory fusion and supporting IMM.**
`results/fusion_trajectory_imm_v1/REPORT.html`; scientific review PASS.
Same110/v3 labels/v2 groups, original banks/fits/windows and149 unchanged
anchors;27 additions,176 total,50 contrasts. All final outputs valid110;
three inherited Joint fallbacks collapse to one IU observation in paired
families. No new inference, feature refit or causal-online claim.

Primary mean/GLS/IMM PRMB/PB: .669437/31.98298%, .683384/30.34837%,
.693254/16.69023%. Original IU .681306/30.15985%, Joint graph100
.655451/30.22293%; strong risk equal-permuted .768422/33.92003% retained.
GLS vsIU intervals[-.02612,+.02882] AUC/[-8.7608,+9.3277]pp include0.
No winner. Static mean/GLS are feature-weight combinations; IMM introduces
chronological state evolution, not semantic correct/error states.

Primary IMM vs static hold gains4 PB successes/loses13. Raw exact peaks
18->17, final exact errors12->7, clean successes15->11. Post-evaluation
fixed-component exchange: hold peak/hold gate30.34837%; hold peak/IMM gate
23.27899%; IMM peak/hold gate27.43170%; IMM peak/IMM gate16.69023%.
These are diagnostics, not new candidates or causal mediation. Clean median
lag1 rises .28649->.63670, median BIC1-BIC2 .95381->6.44959; false alarms
18->22. This motivates checking the mixture gate under dependence, but does
not establish a semantic-error null or prove an effective-N correction.

Seven tests,880 independent R/GLS and990 direct vector-IMM replays,2970
outputs,176 metric bundles,50 paired comparisons and five explicit1000-draw
bootstraps pass. Post-evaluation audit adds48 metric/component checks and1376
lag replays. Scoring52.80s; review113.13s. Same-session/shared-kernel scope
is disclosed. Full HTML, all comparisons and two inspected PNG/SVG plots
are saved; no browser rendering or external review claimed.

Artifact validation PASS:719 hashes,14 links/images,198 static rows,12 Node-DOM cases/256 numeric rows, seven ASTs and44 guide IDs. All handles terminal.

Next bounded priority: audit existing no-error methods and use controlled
unimodal serially dependent trajectories to check whether smoothing alone
can trigger this mixture gate. Then decide one gate-only comparison with
unchanged fusion trajectories/peaks. No broader temporal/graph sweep from
these point estimates. Full comparators, corrected multi-answer refits,
actual KalmanNet/LOCA/Flows, sparse/short-error sampling, untouched confirmation
and historical24 transfer remain open; full research goal active.

**Step317 sampling replication completed:**
`results/fusion_sampling_replication_v1/REPORT.html`; same corrected current110,
original banks/routes,72 eligible for reduction.42 selector/core entries
(seven full aliases) plus107 historical anchors;93 paired comparisons,
review PASS. Sampling changes fusion fitting rows, not dense scoring or GMM
support. Joint failures explicitly use selected-row IU in the original bank.

Risk IU .758352/28.34853%, risk Joint graph .741328/28.20441%: pooled PRMB
rises strongly but PB falls relative to full IU .681306/30.15985% and full
graph .655451/30.22293%. Risk equal+permuted graph .768422/33.92003% remains
visible; its PB improvement interval includes0. No candidate promoted.
Affine diagnostics show the pooled jump is largely consistent with changed
answer location/scale, with much smaller within-answer changes. Full IU
ranking with sampled mean/SD gives .748761; sampled ranking on full scale
gives .689378. These post-evaluation diagnostics are not candidate scores.
Risk IU gains one PB success/loses two; risk Joint gains two/loses three.
No <=32-token first errors among38 eligible erroneous PB answers. Dense
features remain computed, so neither short-error nor sparse-efficiency claims
are established. Native Joint full/uniform/risk/DUFS/permuted/diffusion:
107/101/90/84/92/99; all final outputs valid with disclosed fallbacks.

Next audit and then bound one whole-trajectory fusion comparison of the
current IU/Joint maps, preserving both fusion axes and peak/gate distinctions.
Do not confuse full-trajectory combination with choosing an existing peak,
or call an old experiment new. Keep149 anchors and the strong simple control.
The broad comparator/refit/named-track/confirmation/24-cell requirements
remain open. The research goal is active, not achieved.

**Step316 historical bridge completed:**
`results/localization_history_bridge_v3/REPORT.html`:131 earlier58 entries,
199 original comparisons,25 corrected fallback references and seven gate
diagnostics. Separate current110/107 context panel. All frozen scores,
predictions, failures and PB points preserved; v3 labels/v2 groups applied
and reviewed. Earlier HMM/Kalman/IMM/BOCPD, sampling, representation and
regularization evidence can now be used within this stated scope.

No consistent winner. Risk-sampled IU has modest positive primary points,
but both intervals include0, within-answer ranking weakens and total PB
correct decisions do not increase. Joint+DUFS's .763333 PRMB uses4 valid
answers with an original Joint baseline .750000 on those SAME4. Answer
offsets improve pooled AUC without local ranking/decision changes.
Next a bounded current110 sampling replication: target-free budgets,
original bank routes, matched full/uniform/risk/graph/permutation and
IU/Joint/equal controls, short-error retention and explicit failures.
All broader comparator/refit/named-track/confirmation/24-cell obligations
remain open; the goal is not complete.

**Step315 gap representation completed:**
`results/fusion_token_gap_v1/REPORT.html`. Same-answer P27 fusion with only
the three surprisal coordinates replaced; original windows, bank routes,
graph seeds and corrected benchmark retained. Seven cores + two scalar
controls,98 inherited anchors,107 total rows/25 comparisons; review PASS.
Gap-IU .680187/28.82295%, gap-Joint graph .649936/30.22293%; neither improves
both primary endpoints against its original core. Joint valid100/110 versus
107, ten explicit IU fallbacks. The scalar controls flag all33 clean answers.
Original raw top50 and source audit extend confidence-feature fidelity;
they are not new inference or proof that a model preference is correctness.

Next bridge older unique readout/sampling/regularization/representation
outputs to v3 targets and v2 groups before reusing their PRMB conclusions.
No pooling of distinct cohorts or silent alteration of old score contracts.
This does not replace corrected multi-answer refits, the named supporting
tracks, comparator coverage, untouched confirmation or historical24 transfer.
All remain part of the active goal; no algorithm winner is claimed.

**Step314 forensics completed:**
`results/fusion_localization_forensics_v3/REPORT.html`. Corrected-label
downstream alignment passes on all110 answers. IU raw peak20/53, graph18/53;
gates hide8/5 good peaks and create15/16 clean false alarms. Shared-window
ties offer limited conditional headroom (label-perfect tie choice gives
IU31.77% versus actual30.16%). Both tied top sets still miss28/53. AR graph
changes16 gates and13 peaks; most changed peaks did not have small old
margins. Keep the original fusion cores and inspect existing realized-token
versus distribution-confidence evidence before one new supporting feature/
readout test. No new method score or oracle-based winner is claimed.

**Step313 correction supersedes inherited PRMB numbers below:**
`results/localization_prm_label_audit_v1/REPORT.html`. Raw PRMB `error_steps`
are one-based; v2 wrote them one step late. New `RELEASE_V3.json` fixes all6969
rows and preserves v2 source groups/folds and PB labels. Current110/all98
and original58/all25 frozen-score bridges,207 contrasts and review complete.
Corrected IU .68131/30.16%, graph100 Joint .65545/30.22%; no proven winner.
Permuted equal-graph .69226/31.32% must remain visible. Old blanket statements
about all21 residual recipes losing on both points and an AR-IU resolved PRMB
drop are withdrawn. No fitting/score/decision changed. The raw-label check
exposed a gap in the earlier derived-NPZ reviews.

Resume interrupted text/peak forensics on v3. Other historical unique arms
need label bridges; multi-answer pipelines need corrected labels and folds
during fitting/selection. These repairs preserve the original research goal
and do not count as an algorithm gain or untouched confirmation.

## What Omri wants preserved

The target is a method with a clear, reproducible advantage on PRMBench and
ProcessBench, suitable for presenting to the advisors. The full research
mandate persists across sessions. It includes Joint L-SML with graphs and a
better-suited feature representation, IU-PCR improvement, feature and
trajectory fusion, chronological HMM/BOCPD, Shlezinger-inspired ideas, and
graph-based selection of token/window sampling points. The 24-cell final-answer
transfer remains a separate follow-up after the localization recipe is locked.

Additional explicit user request, September 7: retain **IMM, LOCA, Diverging
Flows and KalmanNet** as candidate components. Historical testing must be
verified at implementation level, distinguishing named methods from simpler
precursors. Their corrected history, sources and candidate roles are in
`docs/reviews/temporal_geometry_revisit_2026-09-07.md`. These remain part of the
authorized program, not discarded because the current pilot uses other arms.

Every experiment must show relevant historical results and rerun compatible
incumbents under the same evaluation contract. Old numbers from a different
task, fit scope, population or label version remain visible as context, with
their differences stated. They cannot serve as a paired algorithm comparison.

The required access is gray-box, one model pass, unsupervised. The current
localization caches score fixed official answers by teacher forcing. Primary
fitting uses one answer. Any externally learned sign, group, normalization,
threshold or parameter makes that component borrowed and must be labelled.
Repeated CPU processing of a cached trace does not require a second model
pass; it does not make an offline localizer causal.

## Fusion is the central method

Explicit user clarification, September 7: the advisor-facing contribution
must develop our existing **IU-PCR / Joint L-SML fusion method**. Supporting
ideas serve this architecture:

`one cached answer -> N windows x P measurements -> IU-PCR / Joint L-SML -> fused trajectory -> step/no-error decision`

LOCA-inspired geometry can improve the matrix or groups; DUFS-inspired
sampling can choose its rows; IMM/HMM/BOCPD and KalmanNet can stabilize or
interpret the fused trajectory. Flow prediction/transport can supply an
additional view or reliability signal to fusion. These are hypotheses to
test separately, not mandatory layers or standalone replacements.

For every addition, compare (1) unchanged fusion, (2) the same fusion plus
the addition, and (3) simple aggregation plus the same addition. Retain
IU-PCR and Joint anchors and graph lambda-zero/permutation controls where
applicable. An auxiliary-only score is an optional diagnostic ablation, not
a replacement candidate. If the addition works but learned fusion contributes
nothing over its matched simple control, report that limitation explicitly.
Method names and advisor figures should identify the fusion core and added
component so continuity and attribution are visible.

## Work sequence and completion evidence

| Stage | Work | Required evidence | Status |
|---|---|---|---|
| A | Audit latest Codex pilots and Claude v2 | Actual imported module paths/hashes, source-group alignment, fit validity, label access and completion inventory | Initial numerical/source audit, source-group correction and pair-Jacobian review complete; corrected-fold multi-answer refits and cross-lane reconciliation remain open |
| B | Freeze common two-benchmark release and history bridge | Stable source-group IDs, label versions, endpoints, no-error rule, fit/access lanes, incumbent registry, development/confirmation ledger | Corrected v3 labels/v2 groups, current110 anchors and older58 unique-method bridges complete; multi-answer refits, full comparator replay and untouched confirmation remain open |
| C | Reconsider Joint features and hyperparameters; improve IU | Feature feasibility and stability within each answer; admissible groups; equal-budget, label-free selection; lambda-zero and permuted-graph controls | Representation, regularization, context-bank/K, replication, pairs, native conditioning, graph interaction and residual-view quality complete. All330 augmented Joint fits pass. On v3 all21 trail both PB references;20/21 trail IU and9/21 trail graph100 on both points. Raw-text/peak forensics completed; no winner |
| D | Compare chronological readouts of fused scores | Same frozen feature scores and IDs; fixed readout versus HMM/BOCPD adaptation; explicit clean-answer handling | Seven-readout pilot and normalization/gate audit complete: both gate and location constrain PB; improved gating and local feature/readout work remain open |
| D2 | Support fusion with IMM and unsupervised KalmanNet | Within-answer state/innovation components, chronological validation, ordinary Kalman and HMM controls; fusion contribution ablations | Earlier scalar IMM/ordinary Kalman and Step318 paired-trajectory IMM completed; no two-task improvement. Peak/gate exchanges and serial-dependence audit complete; learned KalmanNet gain remains pending |
| E2 | Support fusion with LOCA and Diverging Flows | Geometry for representation/grouping or telemetry prediction as an added fusion view; verify assumptions and reject label-selected clean training pools | LOCA full-paper digest complete; initial flow source audit complete; supporting adaptations pending |
| E | Token/window sampling | Full-grid, uniform, risk and graph selectors at matched budgets; retain coordinates and local-window support; test short-error retention and runtime | Earlier fitting-row pilot plus Step317 fixed current110 replication (six selectors/seven cores,93 contrasts) complete. No consistent IU/Joint gain; retain strong risk equal-permuted control. Dense scoring retained; sparse pipeline and short-error subgroup remain open |
| F | Freeze candidate and confirm | One fixed selection procedure, paired uncertainty on both benchmarks, coverage and subgroup failures, genuinely untouched confirmation data | Pending |
| G | Advisor report and historical transfer | Reproducible artifact, limitations and mechanism ablations; frozen-candidate transfer to historical 24 cells | Pending |

Stages C–E should use small decision-oriented experiments before expanding.
The user has authorized these tracks; another method-selection permission
question is unnecessary. A failed pilot closes only the tested configuration.
Keep negative results and do not repeatedly inspect confirmation labels.

## What counts as a clear winner

Before confirmation, freeze the comparator set, primary metrics, practical
improvement margins, uncertainty procedure and handling of failed fits. The
primary recipe must be the same declared procedure on both benchmarks; a
different label-selected winner per benchmark does not meet that objective.
Report paired differences to relevant matched leaders on each benchmark,
adjusting for confirmatory multiplicity where applicable. Report coverage,
clean/no-error decisions, source-group uncertainty, subgroup consistency and
runtime. A favorable average with catastrophic failures elsewhere is not a
general improvement. Discovery results are not untouched confirmation.

The numerical practical margins and confirmatory test identity are **not yet
frozen**. They must be chosen before new confirmation labels are inspected.
Do not claim an advantage simply because a confidence interval excludes zero
for one of many retrospectively chosen variants. If no method wins, state that
result; the objective cannot honestly be guaranteed in advance.

## Important distinctions retained from the conversation

- Joint L-SML combines global and group-specific covariance structure. Its
  name does not establish that feature and chronological trajectory fusion
  were jointly tested.
- K groups and P features are different quantities. Three groups are not
  three features. A minimum group-size restriction must be assessed against
  actual group membership and model identifiability.
- Constant-width windows give equal support per full window, not an equal
  number of windows per answer. Overlap adds correlated observations.
- Sampling observations changes which rows are used. Feature gates change
  columns; a graph penalty changes a fitted map. Testing one does not test
  the other two.
- A sparse selector can preserve global graph structure while missing a short
  local error. Retain coordinates, evaluate all official steps, and measure
  short-event retention. Boundary labels are for evaluation, not selection.
- Older HMM/BOCPD scalar-feature failures do not automatically answer whether
  a chronological readout of the new answer-only matrix helps. Later local
  DSP studies must also be reviewed before calling an adaptation new.

## Completed prediction-view quality test — Step312

`results/fusion_prediction_quality_v1/REPORT.html` preserves the fusion core
and compares all21 AR/last/EMA additions with equal, IU, Joint0/graph/permuted
and matched equal-graph controls. All77 historical references remain exact;
98 arms and74 registered contrasts complete on the same110 development answers.
Original bank routing is fixed81moment/29context. All330 augmented native
Joint fits pass; no fallback is used. AR K3/4/6/8 counts34/32/39/5. This tests
the user's feature/grouping hypothesis: more groups and valid fits are
possible, but did not yield better localization in this recipe.

Original IU .63797/30.16% and graph100 Joint .63847/30.22% remain above every
new arm on both primary points. AR+IU .61334/23.50%; AR+Joint graph
.62355/18.01%. AR-IU PRMB delta CI[-.05027,-.00477]; AR-graph PB delta
CI[-23.824,-.764]pp. The107 both-native-fit-valid comparison retains the
AR-graph PB loss, CI[-24.466,-1.072]pp. Intervals unadjusted/exploratory.
Last+IU .62491/29.74%; EMA+Joint graph .62066/28.39%, also below old anchors.
No consistent improvement over matched simple/graph controls. Fixed-IU PB
and within-answer ranking also weaken for AR, so the observed change is
not confined to pooled calibration or the native no-error gate.

Review/tests pass:330 covariance/Jacobian/IU/graph reconstructions,2310
projection/step/GMM paths,98 metrics,74 point/scope bundles,six explicit
bootstraps and15 each representative group/Joint/gate refits. Largest
risk discrepancy2.26e-14. Scoring359.84s/contrasts40.89s/review106.96s.
Review-harness record/rounding issues fixed; frozen code/scores unchanged.
Exported figure visually inspected; HTML structure/14 local links pass.

Post-evaluation sign/weight/correlation and PB-hit diagnostics are
descriptive, not a demonstrated cause. Next inspect actual first-error
spans/text, near-tied maxima and clean-decision changes with frozen scores
before choosing another fusion/readout addition. Do not widen the residual
grid to rescue this negative result. The full named program, repaired-fold
refits, comparator inventory, untouched confirmation and historical24 remain
open. No winning method has been established.

## Completed prediction-view history and feasibility audit — Step 311

`results/fusion_prediction_view_audit_v1/REPORT.html` and
`docs/experiments/FUSION_PREDICTION_VIEW_AUDIT_V1.md`. This is not a new
quality experiment. Earlier token B3, Local/Online IU and CIW innovations
use donor/calibration-answer fitting; the history was broader than scalar
AR/Kalman final-answer tests. Named B3 result-freeze execution outside the
scoped local filename search is unverified, not declared absent everywhere.

The new AR(1) support prototype learns from preceding pairs in one answer,
with fixed shrinkage toward the last observation. Nine mean-absolute-error
columns are appended to moment27/context27; all originals replay exactly.
Last-value and EMA32 residuals are controls. No external learned quantities
or new model calls; full-answer fusion would remain offline. This small
predictor is not actual KalmanNet or Diverging Flows.

All 110 development answers / 71,385 tokens support finite varying columns.
Entropy AR/EMA32 MSE median .97367, wins78/110; top1/tail-mass medians exceed
one. Those are telemetry prediction results only. AR closest-original-column
median absolute correlation is .87072 moment / .83750 context. EMA32 is less
redundant than AR for moment; neither low redundancy nor higher matrix rank
establishes added error information. 40/110 have N<36; 31 original matrices
already reach rank N-1. Actual Joint validity remains to be measured.

Three tests and independent batch prediction/feature/correlation review pass
(2,970 stream traces, 330 residual/MSE arrays, 660 exact original-column and
correlation/rank checks; largest discrepancy1.35e-13). Audit10.34s/review11.74s.
No new localization head or label-quality evaluation. Next freeze the matched
fusion-core/equal-control augmentation study, including AR/last/EMA and graph
controls, fixed bank routing, explicit failures/fallback, common coverage and
both benchmarks. Keep Step310 as the latest quality evidence and all broader
tracks open. This stage implements the fusion-central mandate concretely.

## Initial audit observations (historical snapshot)

The completion statements below describe the initial inspection. Later
completed-stage entries record subsequent Jacobian, grouping and quality
checks; do not treat this snapshot as the current run status.

1. `local_cache/short_cycle01_code/spectral_utils/short_cycle_localization.py`
   is the module resolved first by `scripts/run_short_cycle01.py`. It calls
   `confidence_sign_vector` and records borrowed feature calibration. This is
   not a newly established all-components-answer-only result. Audit the exact
   historical authorization separately; preserve the stored scores.
2. The capsule reports `converged` and `multistart_status` separately. Before
   interpreting "strict coverage", check whether the published filter required
   both and whether Jacobian identifiability was tested. Pending verification.
3. `report_contrasts.json` in Claude v2 exists with six PRMB contrasts and an
   empty ProcessBench object at inspection. The v2 report still contains empty
   sections. Do not infer all report work finished from its filename.
4. These are retrospective pilots. A new row-ID cohort is not automatically
   an untouched source-group test, especially if Claude evaluated it already.

Audit artifact: `results/localization_short_cycles_audit_20260907/REPORT.md`
and `AUDIT.json`, produced by `scripts/audit_localization_short_cycles.py`.
The original AUROCs reproduce, 8/8 upstream hashes match, and all original OK
Joint fits also have multistart PASS. The 30 answers contain 29 registered
source groups; revised group-bootstrap intervals do not reverse the pooled
conclusions. A retrospective within-answer diagnostic narrows the graph/IU
gap but establishes no significant graph benefit. Preserve both metrics with
their distinct estimands, populations and disclosure of when they were added.

## Completed representation pilot

`results/answer_localization_representation_pilot_v1/REPORT.md` records 19
arms on 58 previously exposed development answers (12 PRMB, 46 PB), with
historical recipe replay on the same IDs. Width-8 moment features raise
IU coverage to 58/58 and Joint to 43/58, versus 38/58 and 32/58 for the legacy
width-32 recipe. Joint selects four groups in 14 moment-8 answers; three
groups were not a universal architectural limit. On seven common PRMB
answers, graph-minus-IU AUROC is +0.01291 [-0.03016,+0.04864]; no advantage
is established. The fixed fused-score mixture/first-crossing PB readout gives
zero macro-F1 for all 18 fusion arms. Preserve this failure and isolate the
readout next; do not change frozen predictions after inspecting labels.

Five contract tests, 583 projection/span replays, all 19 endpoint
recomputations, direct label-ID joins for all 58 answers, raw-column identity
and frozen hashes pass. This establishes numerical consistency, not a
validated correctness interpretation for unsupervised mixture states.

## Completed supporting-readout pilot

`results/fused_trajectory_readout_pilot_v1/REPORT.html` compares parent first
crossing, raw peak, held-tail peak, reversible HMM, ordinary Kalman, IMM and
a verified BOCPD adaptation. All core fits and binary gates stayed fixed.
IU raw peak gives PB 17.71% (was zero) and unchanged PRMB 0.62261; equal gives
17.76% and entropy 19.85%. IU+IMM gives 14.79%, IU+BOCPD 13.69%, HMM zero.
Joint graph peak gives 8.33%, versus lambda zero 12.50%; small paired
intervals establish no graph benefit. All 58 contrasts and independent
checks completed. The parent-peak PB gain CI includes zero; do not promote
it as a confirmed research advantage.

The fixed graph error gate caps this cohort at 28.24% even with perfect
localization; IU's gate ceiling is 55.18%. These are post-hoc oracle
diagnostics, not achieved results. Next address fusion inputs/reliability,
label-free Joint parameters and observation sampling; a different clean/error
gate needs its own version. Preserve all still-open stages above.

LOCA's full-paper digest is `papers/digests/loca-local-conformal-autoencoder.md`.
It motivates estimating local measurement uncertainty as a supporting
adaptation, but does not establish valid bursts from consecutive tokens.
The old BOCPD boundary-convention issue and the new exact-enumeration check
are in `docs/reviews/bocpd_boundary_audit_2026-09-07.md`.

## Nir Shlezinger: relevant sampling leads

Graph-compression full read is now cached in `papers/digests/task-based-graph-signal-compression.md`. The ADC item remains an initial source check; neither is a tested localization method:

- [Task-Based Graph Signal Compression](https://arxiv.org/abs/2110.12387)
  jointly designs sampling and recovery for bandlimited graph signals under
  quantization constraints. This motivates preserving information needed by a
  downstream operation. It does not establish that hallucination boundaries
  are bandlimited or that selecting token nodes will improve localization.
- [Deep Task-Based Analog-to-Digital Conversion](https://arxiv.org/abs/2201.12634)
  learns sampling/quantization together with downstream processing. Its
  task-training setup must not be imported as unsupervised localization.

Our proposed adaptation is to select observation locations using only a
single trace, with a label-free objective such as reconstruction or predictive
stability, and then evaluate localization. This is our hypothesis, not a claim
that either paper provides an unsupervised hallucination detector. Compare it
with DUFS-inspired selection; graph compression may mix nodes rather than
select literal token locations, so a selection-only constraint needs its own
implementation. Review the full papers before freezing an implementation.

## Completed fitting-window sampling pilot

`results/fusion_window_sampling_pilot_v1/REPORT.html`: six selectors support
five fixed IU/Joint/equal cores on all 58 development answers. Selection
reduces fitting rows in 37 answers (21 exact full-row replays); every window
is still scored. The full arms reproduce the previous peak endpoints exactly.
IU transposed DUFS gives PRMB 0.63663 / PB 13.69%, versus full 0.62261 / 17.71%.
Its shuffled control gives 0.63761 / 21.88%, with a PB improvement CI including
zero. Top-risk IU gives 0.64217 / 19.05%, but its fixed-parent-gate PB is
unchanged and within-answer PRMB ranking decreases. No sampling variant is
promoted. Joint graph reduced-row PB is 4.17% for every selector versus 8.33%
full, with changing fit coverage that requires common-ID comparisons.

Six scientific tests and independent selection, fit-weight, span, label-ID,
endpoint and hash reviews pass. All 57 paired contrasts completed. The
eligible PB group has no <=32-token first-error steps, so short-event
retention remains untested. Block perturbation gives mean Jaccard 0.646 for
DUFS and 0.474 for window diffusion, compared with DUFS seed agreement 0.859.
Next focus on full-row feature reliability and label-free Joint regularization,
retaining the fusion contribution controls. This does not close task-aware
sampling, replace fusion, or finish confirmation/comparator/24-cell work.

The 21-page attached DUFS arXiv v3 (2020) was fully read and indexed:
`papers/digests/differentiable-unsupervised-feature-selection-based-on-a-gat.md`.
It clarifies feature gates versus graph nodes after transposition, paper
normalization, parameter-free versus additive losses, and differences from
the repository's short fixed-epoch adaptation. Do not mix the attached
five-author preprint with the later six-author NeurIPS metadata.

## Completed fusion sensitivity regularization pilot

`results/fusion_reliability_regularization_v1/REPORT.html`: all 23 arms and
38 paired contrasts completed on the same 58 answers. The existing feature
bank, fitting rows and Joint groups stayed fixed. Joint uses native covariance
regularization; IU/equal use a disclosed identity-head correction of their
existing weights. Six exact parent controls provide the historical bridge.

Joint graph lambda 0.1, 1 and 10 gives PRMB 0.66255, 0.61426 and 0.51713 on
seven common answers, and PB 8.33%, 8.33%, 9.17%. Automatic label-free graph
selection gives 0.66760 / 5.88%. No consistent gain is established. Full-matrix
sensitivity regularization reduces variability on unused perturbations while
reducing localization accuracy. Do not equate a smaller robustness loss with
better correctness information. The added eight-replicate robustness audit
was post-freeze, unlabeled and did not change scores or lambda selections.

Next audit the answer-only normalization/no-error interface, then freeze a
small gate experiment holding fusion scores and simple controls fixed. A
pooled unlabeled gate, if tested, remains an explicitly hybrid comparison.
Broader feature-bank/grouping work, comparator coverage, untouched confirmation
and historical transfer stay open. The research goal is not complete.

## Completed normalization/no-error interface audit

`results/fusion_gate_interface_audit_v1/REPORT.html`: seven frozen cores on
58 exposed answers, 14 original task endpoints replayed. Five scientific
identity tests and independent label, score, oracle and AUC review pass.
No new candidate or label-fitted threshold was introduced.

IU PB is 17.71%; a perfect gate with its same peak gives 41.89%, its same gate
with a perfect locator 55.18%. Joint graph values are 8.33%, 41.50% and 28.24%.
These are failure-preserving oracle diagnostics, not achieved gains. IU's
peak is exact in 7/25 erroneous answers; Joint lambda zero in 9/25, graph in
8/25, with four invalid Joint error fits. Both components require work.

For IU, 89.33% of pooled PRMB AUC comparisons cross answers. Removing a
centering offset raises pooled AUC 0.62261 -> 0.68174 without changing any
within-answer pair. The resulting origin projection is not calibrated or a
selected method. Keep the registered endpoint and the distinct within-answer
metric. All 361 score-translation tests preserve the free-mean GMM decision;
restoring an offset alone cannot repair its correctness meaning. This is a
restricted pipeline invariance, not an impossibility result for gray-box work.

Next return to a local multiscale feature bank and Joint grouping, with equal
and IU anchors plus fixed-parent/native gate diagnostics. First audit existing
causal DSP mechanisms. `unified_causal_iu.py` and its subset-search module
explicitly distinguish label-free update operations from supervised
historical development. Reuse mechanisms without silently borrowing fitted
signs, references or label-selected rosters. A separate pooled unlabeled gate
remains a hybrid scope. All still-open mandate stages remain active.

## Completed context-bank and K-roster pilot

`results/fusion_context_bank_pilot_v1/REPORT.html`: 17 arms and 31 paired
contrasts, same 58 exposed answers. The same nine primitive streams now supply
window means of level/EMA8/EMA32 alongside the old moment bank; all feasible
K=3..9 is compared with legacy {3,4,6,8}. No historical supervised-developed
DSP references or selected roster enters this answer-only adaptation.

Joint fits 50 instead of 43 answers, with 13 rescues and six regressions.
Its PB lambda-zero result is 27.43% instead of 12.50%; context equal is 27.67%.
The banks share only four valid PRMB answers: Joint context 0.66823 versus
moment 0.72613 there. Expanded K finds five-group solutions but PB stays
unchanged. A consistent two-task advantage is not established. All six parent
controls and 12 original task endpoints replay exactly.

Six scientific tests, 545 independent weights, 848 gates/span maps, 1,044
group-candidate checks and source/label/metric reviews pass. The faster
source-group bootstrap exactly matches the old implementation on three real
1,000-draw comparisons; all 31 contrasts and diagnostics are complete.

Next proposed bounded test: explicit moment-Joint -> context-Joint -> IU
fallback within the same answer, with numerical eligibility and gate handling
frozen before evaluation. Keep pure failing methods, a moment-Joint + IU
fallback comparator, plain IU and equal aggregation under the same routing.
This tests complementary coverage; it is not yet implemented or an accuracy
claim. All remaining mandate stages remain open.

## Completed explicit fallback pilot

`results/fusion_explicit_fallback_pilot_v1/REPORT.html` closes the small
fallback question. The 25-arm table retains all 17 context-pilot anchors;
30 registered contrasts, five scientific tests and independent review pass.
The single policy uses moment Joint -> moment IU; dual uses moment Joint ->
context Joint -> moment IU. Eligibility uses audited fit validity only.
All composite fits cover 58 answers, while pure Joint failures stay visible.

Single Joint0 -> IU has PRMB 0.63663 / PB 27.22% against IU 0.62261 / 17.71%;
paired intervals include zero. Context equal remains stronger on both
headline points (0.66087 / 27.67%). Review added its missing paired comparison
as an explicitly post-evaluation supplement; it did not alter the frozen
protocol. Within-answer rankings differ, so do not conflate these metrics.
Dual routes 13 extra answers through Joint but adds no PB exact successes.
Graph remains unproven. PB clean hits fall 12 -> 8 and exact-error hits rise
4 -> 6 for single Joint versus IU; the macro gain is not uniform improvement.

Next: exposure/source-group audit and a small fixed-recipe disjoint
replication, including both equal banks and graph controls. Existing release
metadata says previously evaluated by v2, so that would be development
replication. A genuinely untouched confirmation set still needs separate
provenance. Remaining Joint features/grouping, temporal/geometry/sampling
support, full comparator coverage and 24-cell transfer are not closed.

## Completed source-question grouping correction

The planned exposure audit found a prerequisite for valid comparison:
`results/localization_source_group_audit_v1/REPORT.html`. PRMB perturbation
IDs split source questions across old folds; PB repeats identical problems
under different answer IDs. The new source-component release and global
five-outer/five-inner folds are independently checked. All 25 answer-only
fallback metrics remain unchanged; 32 interval bridges are complete.

No multi-answer fit has been repaired just by changing group metadata.
Relevant Claude contenders need a corrected-fold rerun. The fixed-recipe
single-answer replication remains next, with no cohort selected yet. Its
prototype has two passing tests (including 19-arm replay in three existing
route cases), not new experimental evidence. Preserve the original scoring
namespace when using the v2 group release, to retain permutation identities.

Only 11 source components underlie the 12 pilot PRMB answers, and two overlap
earlier short-cycle cohorts. The documented Codex exclusion list covers 94
components, not all project exposure. Cached data were already evaluated by
v2 and cannot become untouched merely by taking a new subset. Both localization
caches were scored with one teacher-forced pass over fixed official answers.

Source-question groups for evaluation are separate from Joint feature groups.
Claude's new minimum-feature-group-size-two suggestion needs an independent
identifiability audit before changing that recipe. Joint/graph feature and
hyperparameter work, IU improvement, temporal/geometry/sampling support,
full comparator coverage, untouched confirmation and 24-cell transfer stay open.

## Completed fixed-recipe source-disjoint development replication

`results/fusion_replication_v1/REPORT.html`: 19 fixed recipes and 38
registered paired comparisons on 110 distinct corrected source groups,
excluding the documented 94-component Codex pilot inventory. The 24 PRMB
and 86 PB answers are length-stratified development data, not untouched
confirmation. No labels or fit outcomes entered selection; predictions were
frozen before label evaluation. No new model pass or Claude worktree edit.

The single Joint0 -> IU advantage did not recur: PRMB/PB is now
0.59733/21.45% versus moment IU 0.60140/26.38%. Dual IU gives
0.63797/30.16%, dual Joint0 0.62884/25.33%, dual graph0.1
0.63350/29.94%, and matched dual equal 0.62649/25.55%. Dual IU's
moment-IU difference intervals are [0.00764,0.06898] AUC and
[-2.64,+10.61] PB percentage points. Its within-answer and matched-equal
intervals include zero. There is no established two-task winner. The dual
IU bank rule still uses Joint fit eligibility, so its compute is not plain IU.

Dual graph minus its permuted control has PB delta +4.21 points,
exploratory CI [+0.17,+9.58]; its PRMB interval includes zero. The graph
versus lambda-zero and versus IU comparisons do not establish a two-task
advantage. Keep this signal visible without claiming confirmed superiority
from one of 38 unadjusted development comparisons. The score phase took
213.06 seconds on three CPU workers; all score/review handles are terminal.

Pure moment/context Joint coverage is 78/102 of 110: context rescues 29
and loses five. Dual uses 78 moment Joint / 29 context Joint / three IU.
All composites cover 110. A common-ID check reverses the misleading pure
context Joint/equal pooled ordering: on 21 PRMB answers Joint0 is
0.68397 versus equal 0.69658. PB clean/exact-error hits are 15/11 for
moment IU, 18/12 for dual IU and 17/13 for dual graph, out of 33/53.
Dual IU improves the registered F1 in three subsets and ties GSM, but
Omni-Math exact-error hits fall six -> five, offset by clean detection.

Four pre-freeze tests, independent raw/window/normalization/projection,
native GMM, routing, label joins, all 19 metric/history bundles and all 38
paired point checks pass. Five explicit 1,000-draw bootstrap replays match
four intervals and defined counts. Five representative Joint refits and 15
native inverse heads match, reusing the original optimizer/graph kernels.
The report includes historical results on their separate 58-answer cohort.

Next audit minimum FEATURE-group size two, distinguishing latent loading,
covariance and fusion-weight identifiability and solver behavior. This
supports the existing Joint fusion core. Do not confuse it with benchmark
SOURCE-question groups: corrected-fold multi-answer refits remain required.
Do not retune frozen recipes on the replication's labels. Keep both equal
banks, routed equal, moment/dual IU and Joint zero/graph/permutation anchors.
Broader fusion/supporting ideas, full comparator coverage, untouched
confirmation and the historical 24-cell transfer remain active and pending.

## Completed Joint pair-group audit

`results/joint_pair_identifiability_audit_v1/REPORT.html`: a mathematical
and unlabeled structural extension of our existing native Joint fusion.
A two-feature residual identifies u_i*u_j, not each loading. The frozen
legacy constructor's clipped diagonal can turn an arbitrary pair scale
into a different covariance and native map. A synthetic counterexample
shows this while the old profiled-global Jacobian still passes.

The new feasible pair covariance allocates equal fractions of available
residual variance, preserving the pair product and observed diagonal,
rejecting infeasible pairs and retaining the legacy minimum-three path.
Pair fits also require native covariance/map agreement across converged
starts. On the same 110 answers, moment valid coverage is 106 (was 78;
31 rescued, three lost), context 108 (was 102; seven rescued, one lost).
The union is 110 instead of 107. Moment chooses K=6/8 in 60 answers;
context still chooses 3/4. There are 55/12 valid actual pair fits in the
two banks. The lowered held-block minimum can also admit final partitions
without pairs. One infeasible pair and five convergence/multistart failures
remain visible. This is not a measured localization gain.

During review, a zero-product corner case exposed an additional weakness:
at u_i=u_j=0 the factor-coordinate nuisance Jacobian vanishes. A separately
frozen post-fit, unlabeled amendment instead profiles one direct product
coordinate per pair. Future experiments must use
`spectral_utils.joint_pair_jacobian.fit_joint_pairs_checked`. All 219
fitted audit records retain their eligibility under the amended check,
with 69 independent product-profile reconstructions and no current exact
zero pair product. A synthetic false-positive example is now rejected.
Do not use the original pair-fit prototype alone in a new quality trial.

Eight original and three amendment tests pass. Independent review checks
220 normalized covariances, 880 group guards/ARI summaries, 220 selections,
219 native maps, 107 pair allocations, the infeasibility, 117 unchanged-
partition parent maps and nine representative refits. Original clustering
labels and optimizer reused. Runtime 130.96 s, final review 8.44 s; all
handles terminal. HTML structural/link and interactive-JS syntax checks
pass. No browser visual inspection, label decoding or new inference.

The follow-up, completed below, measures checked-pair native Joint, lambda-zero/0.1/
permuted-graph controls and the existing minimum-three/IU/equal anchors
with matched routing and native/fixed-IU diagnostics. Do not widen the
K/condition grid simultaneously. Supporting ideas remain in service of
fusion. Corrected-fold multi-answer refits, wider research tracks, full
comparators, untouched confirmation and historical24 transfer stay open.

## Completed checked-pair localization quality

`results/fusion_pair_quality_v1/REPORT.html`: 19 frozen minimum-three/IU/
equal/entropy anchors and 14 pair extensions on the same 110 development
answers. All 63 registered paired contrasts and independent review pass.
The scorer calls `fit_joint_pairs_checked`, replays the 219 fitted audit
covariances and freezes new predictions before reading labels. No new
model pass, cohort selection, wider K/condition sweep or Claude edit.

| Full-coverage recipe | PRMB AUC | PB F1 |
|---|---|---|
| Original moment IU | 0.60140 | 26.38% |
| Original dual IU | 0.63797 | 30.16% |
| Original dual Joint graph0.1 | 0.63350 | 29.94% |
| Checked-pair dual Joint graph0.1 | 0.58974 | 19.29% |
| Checked-pair dual IU routing control | 0.59222 | 26.10% |
| Original single Joint0 -> IU | 0.59733 | 21.45% |
| Checked-pair single Joint0 -> IU | 0.59742 | 19.18% |

New minus old dual graph is -0.04376 AUC, CI [-0.08151,-0.01527], and
-10.65 PB points, CI [-21.04,-1.51]. These are unadjusted exploratory
intervals. The corresponding within-answer and fixed-IU-gate diagnostic
intervals include zero. No new pair recipe is promoted.

Pure fit coverage rises to 106/108 moment/context. The dual route changes
78/29/3 moment-Joint/context-Joint/IU into 106/4/0. It changes banks on
31 answers (28 context -> moment, three reverse); three additional
moment-IU -> moment-Joint transitions leave the bank unchanged. IU maps
within each bank replay exactly. Its routing-only decline shows that fit
eligibility alone does not choose the useful bank. Joint regression can
also involve weight changes, not just routing.

Keep the common-ID qualification: context Joint0's unmatched AUC is
0.68397 on 21 old valid answers versus 0.64353 on 23 new. On the 20
common answers the delta is +0.00109, with within-answer delta exactly
zero. Its PB full-population delta is +3.92 points, CI [0,+11.50].
The full-row AUC decline is not evidence of a common-ID ranking loss.

Two new tests supplement the 11 preceding pair tests. Review reconstructs
110 label/group joins, 4,710 parent arrays, 2,090 parent metadata records,
220 groupings, 219 covariance/factor fits, 67 product Jacobians, 642
native/graph maps and decisions, 880 route inheritances, 33 metric bundles
and 63 paired point bundles; six explicit 1,000-draw interval replays match.
Original optimizer/DUFS/graph-builder kernels reused. All handles terminal.
Scoring 157.49 s, contrasts 31.31 s, review 36.50 s. HTML structure and
11 links pass; no browser visual inspection. The old 58-answer table is
preserved separately and is not compared as an algorithmic improvement.

The follow-up, completed below, keeps ORIGINAL minimum-three Joint fits, groups and bank routing
fixed and test stronger native inverse conditioning, initially at lambda
zero with existing graph/IU/equal controls. A post-evaluation unlabeled
diagnostic finds condition >=999 in 104/106 valid pair-moment and 97/108
pair-context maps; old/new common-fit medians are both 1000. This motivates
the test, without proving conditioning caused the errors. Keep the safe
pair mathematics and the negative result. Supporting ideas continue to
serve our fusion; no standalone replacement is proposed. Corrected-fold
multi-answer refits, broader tracks, full comparators, untouched confirmation
and historical24 transfer remain open.

## Completed original Joint native conditioning

`results/fusion_native_conditioning_v1/REPORT.html`: 45 arms and 69 frozen
paired comparisons on the same 110 development answers. The 33 preceding
arms replay exactly. Twelve new heads apply native inverse condition caps
30/100/300 to original moment/context/single/dual Joint. Graph lambda is
zero. Groups, validity and routes remain original, with no pair extension.

Original C/v were not persisted. Reproduced all 180 valid original fits on
their original same-answer matrices/partitions with the same seeds and
optimizer, requiring condition-1000 weights/scores/decisions to replay.
The recovered C/v/u arrays are now stored for subsequent experiments.
The 40 originally invalid bank fits stay invalid. Single routes 78 Joint/
32 IU; dual routes 78 moment Joint/29 context Joint/three IU throughout.

| Full-coverage original-route recipe | PRMB AUC | PB native F1 |
|---|---|---|
| Dual Joint, condition1000, lambda0 | 0.62884 | 25.33% |
| Dual Joint, condition300, lambda0 | 0.63079 | 25.47% |
| Dual Joint, condition100, lambda0 | 0.63296 | 25.47% |
| Dual Joint, condition30, lambda0 | 0.63639 | 28.43% |
| Dual Joint, original graph0.1 | 0.63350 | 29.94% |
| Dual IU | 0.63797 | 30.16% |
| Dual equal | 0.62649 | 25.55% |

Condition30 minus original Joint0 is +0.00755 AUC, CI [-0.00062,+0.01777],
and +3.10 PB points, CI [-2.58,+9.74]. Both include zero. Dual IU is higher
on both primary points; matched IU/equal intervals also include zero.
These exploratory unadjusted intervals do not establish a best condition
cap or a consistent winner. Keep the original IU and graph references.

Dual condition30 clean/exact successes are 16/12, versus original 14/11;
raw-peak successes stay 17/53, without implying identical answer-level hits.
Its fixed-IU PB declines 29.25% -> 28.73%. Within-answer AUC rises
0.63034 -> 0.67117, with a paired interval including zero. Native PB gains
must not be described entirely as better peak localization. Pure context
condition30 has AUC 0.68799 on 21 valid answers, but equal on those same
21 gives 0.69658; paired difference -0.00859, CI [-0.01748,-0.00100].
Do not compare its pure full-row AUC against equal on all 24 as an advantage.

Three scientific tests and independent review pass. Review covers 110
label/group joins, 8,411 exact parent arrays, 3,630 metadata records, 180
covariance reconstructions, 720 inverse/step/GMM reconstructions, 660
fixed-route inheritances, 45 metric and 69 point-contrast bundles. Ten
original-fit replays and six explicit 1,000-draw bootstrap replays match.
Optimizer and GMM kernels reused; input/algebra/decision/metric checks
remain independent. A first refit differed by about 1e-11 under independently
ordered normalization; preserving exact source reduction order fixes replay
without changing any frozen source, score or tolerance. All handles terminal.
Scoring 63.11 s, contrasts 33.44 s, review 32.12 s. HTML structure/two SVG
plots/11 links pass; no browser visual inspection. Older-58 results remain
separate historical context. No new inference or untouched confirmation.

The follow-up, completed below, tests the SAME caps with graph0.1 and permuted
graph, holding original C/v, groups and routes fixed and reusing same-answer
gates. This asks whether graph structure adds value beyond generic inverse
regularization, without widening K or graph-dose search or choosing a cap
from quality labels. Our fusion remains central. The broader supporting
tracks, corrected-fold refits, full comparators, untouched confirmation
and historical24 transfer remain open.

## Completed graph/conditioning interaction

`results/fusion_graph_conditioning_v1/REPORT.html`: 77 arms, 101 registered
contrasts and independent review on the same 110 development answers.
All 45 prior anchors replay exactly. Twenty-four original-fit native
graph heads cross moment/context/single/dual, caps 30/100/300 and real/
permuted graph .1. Eight graph-smoothed equal controls substitute identity
covariance and uniform loading in the SAME trace-matched mechanism.
Lambda0 replays equal; all caps give the same simple control because its
condition is <=3.18289. No Joint refit, regrouping or new inference.

| Full-coverage original dual route | PRMB AUC | PB native F1 |
|---|---|---|
| Joint graph0.1, original condition1000 | 0.63350 | 29.94% |
| Joint graph0.1, condition300 | 0.63684 | 29.94% |
| Joint graph0.1, condition100 | 0.63847 | 30.22% |
| Joint graph0.1, condition30 | 0.63838 | 29.45% |
| IU | 0.63797 | 30.16% |
| Equal with real graph | 0.62993 | 26.62% |
| Equal with permuted graph | 0.62170 | 31.32% |

Graph100 minus IU is +0.000497 AUC, CI [-0.03409,+0.02323], and
+0.063 PB points, CI [-7.75,+8.18]. This is not superiority or an optimal
cap. At cap100 the graph improves PB over zero/permutation by +4.75
points, CI [0,+10.37]; at cap300 it beats permutation by +5.22 points,
CI [+0.61,+11.28], but PRMB's interval includes zero. Joint versus the
matched equal-graph control remains inconclusive on both endpoints. These
are 101-comparison exploratory unadjusted intervals, not confirmation.

Pure context graph100 is 0.68452 on 21 valid PRMB answers; equal on the
same IDs is 0.69658, difference CI [-0.02432,-0.00124]. Keep common-ID
comparisons and all-population PB failure accounting. Original single/
dual routes remain 78/32 and 78/29/3. Simple controls follow the same bank
route but use moment equal-graph on the native IU fallback cases, matching
the routed-equal convention. Readout failures do not change routing.

An explicitly POST-EVALUATION diagnostic uses labels to describe error
overlap: raw peaks both hit 16/53 erroneous PB answers, IU alone four,
Joint alone two, neither 31. Joint hit sets are identical at all four
caps. A selector restricted to those existing peak locations can hit at
most 22/53; this does not limit whole-trajectory fusion, reranking or new
features. Graph100 and IU both get 30 PB answers right in total, with
clean/error trades 17/13 versus 18/12. IU retains higher within-answer
AUC and fixed-IU PB. A tiny primary macro lead is not uniform progress.

Three pre-freeze tests and review PASS. Review checks 110 direct label/
group joins, 11,351 exact arrays, 4,950 metadata records, 182 original
gate replays, 220 source graph/equal-zero replays, 440 Laplacians/simple-
control invariance checks, 1,880 inverse/step/GMM reconstructions, 1,760
route inheritances, 77 metric and 101 paired point bundles. Seven explicit
1,000-draw interval replays match. Ten representative new-control gate
recipes refit; source graph-builder/DUFS/GMM kernels reused. The post-
evaluation overlap artifact receives a separate Boolean-set recheck.
Scoring 51.35 s, contrasts 50.53 s, review 68.70 s; all handles terminal.
HTML structure/12 links pass, no browser visual inspection. No Claude edit.

Next retain IU and original/conditioned Joint graph references, audit the
old AR/Kalman innovation code and test one same-answer prediction-residual
view inside the fusion matrix. Require unchanged-core and matched equal
controls plus information beyond entropy. Old final-answer scalar
innovations are precursors, not evidence about this new use or actual
KalmanNet/Flows implementations. This complements, rather than replaces,
our fusion. Broader supporting ideas, corrected-fold multi-answer refits,
full comparators, untouched confirmation and historical24 stay open.
