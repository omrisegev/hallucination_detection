# Window-localization feasibility and fit-scope study

Status: WINDOW EXTRACTION AND GSM8K FEASIBILITY COMPLETE; FUSION METHOD SELECTION PENDING. Started September 6, 2026.

## September 6 correction: preserve the Joint development line

Omri challenged the recommendation to start with only IU-PCR and continuous
L-SML, pointing to the advisor update and Claude's current study. Joint is
the intended advanced candidate and should be central in the window study;
the earlier two-method shortlist was too narrow. No new scoring has run.

The latest locally saved advisor update is August 27: it documents 13 aligned
methods, plus the CIW-DEEM challenger, rather than a two-method history.
Gmail access on September 6 returned reauthentication required, so the saved
document was reviewed without claiming it is the latest actually sent email.

Claude's current HEAD is `7803cd55`, with R1 and R2 amendments. The registered
comparison has 16 IU settings and 16 Joint/L-SML settings. Joint/L-SML crosses
learned versus provenance groups and continuous versus Joint readouts, and
tests soft gates, gated grouping, a graph-regularized model-covariance inverse,
a graph-free diagonal-regularized inverse, and a historical DUFS hard-selector
control. R1 also adds a 3x3 trajectory-fusion comparison. Joint model-inverse
rows matter because they use fitted group factors through the model covariance;
the original hierarchical readout did not directly use those fitted factors.

Revised recommendation: carry the Joint/L-SML development family and matched
IU tuning budget into the window representation, preserving the fixed S1/S2
rows and simple controls. The exact 23-to-30 feature mapping, grouping rules,
fit-scope eligibility and any regularization changes need explicit registration;
do not paste token-specific constants onto window features. Evaluate each
admissible recipe under both single-answer and pooled training-window fitting.
Small N is a stability question, not a reason to omit Joint in advance.
If current-run results guide selection, keep selection inside development/
training boundaries and label it as such. A token-level winner is not already
a proven window-level winner. Preserve label-free fixed rows alongside tuning.

R2 excludes configurations with incomplete panel coverage from tuned selection,
while reporting their available lanes descriptively. At its pre-label snapshot,
the hard-selector row covered 32/40 PB lanes; ordinary INTERNAL grouping used
same-map provenance fallback in 5/40 lanes and gated-affinity grouping in 24/40.
These are structural diagnostics, not accuracy results. All 40 PB folds and
one PRMB fold were complete at the new check; PRMB outer1/outer2 processes were
active. No evaluation outcomes were read. The prepared repair handoff must
preserve the newer R2 evaluator changes before integration.
Branch: `codex/per-answer-localization-v1`, sparse worktree `C:/Users/omris/TAU/hd_per_answer_wt`, base `ff800082468ecceca4ef15217c5b045314b5fea6`.

## Authorized purpose

Omri requests implementation of the window-size/observation-count tradeoff, feature validity, stable fitting, mapping back to official steps and a no-error decision, plus isolated audit repairs and cluster assessment. The intended new matrix retains P feature definitions; rows are windows from one answer. Omri subsequently allows the multiple-answer fit as fallback if the one-answer fit is not viable. Do not force one-answer fitting or declare it impossible from covariance rank alone.

An asynchronous question explicitly requests which fusion arms to implement/evaluate: deployed IU-PCR; IU-PCR plus maintained L-SML; or those plus Joint. Until answered, feature/window feasibility and integrity engineering can proceed; outcome evaluation and arm-dependent decisions remain pending. Do not silently choose a legacy U-PCR entry point.

## Boundaries

- The live Claude worktree `hd_jlsml_v2_wt` and its frozen score namespaces remain untouched.
- No correctness labels or unopened v2 evaluation outcomes enter window selection, extraction, integrity checks or the feasibility audit.
- Independent fitting per answer and pooled fitting are distinct arms. Pooling centered answers is not a substitute for the one-answer arm.
- Use one fixed window/feature representation for a fit-scope comparison. A change of representation and fit scope together cannot isolate the latter.
- A numerical fit, structural stability and localization accuracy are separate questions. Any optimum is relative to stated constraints and an explicit objective, not guaranteed best error localization.
- Short/degenerate inputs must return a typed unsupported/unstable status or a disclosed pooled fallback. Never silently convert failure into all-correct.
- PB first-error/all-correct and PRMB every-step ranking retain separate evaluation contracts. Offline full-answer fitting does not establish causal streaming.

## Initial measured geometry (labels not inspected)

The canonical feature extractor has FFT minimum 8, STFT minimum 32 (returns placeholder zeros below it), sliding variance window 16, and Hurst scale dependence. A 32-token lower width is a computability starting point, not proof of reliable spectral estimates.

PB source lengths (Qwen scorer copies share these lengths): GSM8K median269.5, MATH451, Olympiad712, Omni-MATH735.5 tokens. PRMB median287. Only2/400 GSM8K traces reach928tokens (29 nonoverlapping32-token windows); PRMB363/6969 do. Therefore requiring N>P for every trace would exclude most data. Regularized low-dimensional methods can still fit N<P, but must be checked for stability rather than assumed sound. Overlap gives more rows, not independent information.

## Engineering work list

1. Recompute full feature measurements on windows of raw per-token streams; preserve schema and explicitly report unavailable/constant columns.
2. Enumerate feasible widths using feature minimums, actual trace length, nonoverlapping fit capacity and a separate scoring stride. Record diagnostics; choose a shared label-free rule before outcome evaluation.
3. Implement the selected canonical fuser(s), isolated per-answer fitting, stability assessment, and the disclosed pooled alternative/fallback.
4. Map window scores to token/official-step scores with explicit boundary and final-window handling. Separate localization from no-error calibration.
5. Implement reusable read-only integrity validation, collision-free additive manifests, honestly dated execution record and evaluator preflight; correct the misleading global-centering diagnostic in the isolated branch. Preserve old science artifacts.
6. Validate locally on fixtures and a small label-free real-telemetry sample; run a short CPU-only cluster timing pilot before scaling.
7. Complete registered evaluation only after the method choice and integrity checks. Report coverage, accuracy, stability, runtime and fit scope separately. A width-32 starting comparison with widths 48 and 64 as larger-window checks is supported by GSM8K count coverage, but remains subject to feature/fuser stability. Do not call it an accuracy optimum.
8. Update the HTML review and handoff with concrete outcomes and remaining limitations.

## Worktree and cluster observations

The initial full worktree checkout attempted to duplicate tracked cache pickles and filled the disk. It was stopped; Git rolled it back, including the duplicate tree. No original caches were removed. Recreated successfully with `--no-checkout` plus sparse patterns selecting code, tests and documents. About2.49GB free remained. All large inputs must be shared read-only; do not archive the entire Git tree for cluster transfer.

Live local Claude was active at27/45 outer folds, four workers, no evaluation directory. Local CPU is an i5-1035G1 with4cores/8threads and16GBRAM. Keep local development light.

AIRCC became reachable after the user enabled VPN. Slurm requires a login shell: `/etc/profile.d/slurm-configless.sh` sets `SLURM_CONF_SERVER=controller-primary`. Current account is `cycle3_tau_averbuch_prj`, partition`power-gpu`, QoS`owner_940`; project root `/shared/cycle3_tau_averbuch_prj`. User queue was empty. Read-only scheduling validation accepted CPU-only32CPU/64GB/2h and8CPU/32GB/10min requests; these were test-only, not submitted jobs. Nodes have160CPUs/~1.8TB. No measured speedup yet; CPU parallelism helps independent traces. No GPU required for NumPy/SciPy fitting.

Existing PB and PRMB raw telemetry remains readable under `/shared/cycle2_tau_averbuch_prj/omrisegev1/results/`. New runs must use an isolated non-home cycle3 directory, not overwrite old code/results. Native dependency readiness is unverified; check inside a compute job. Keep BLAS threads1 per worker, checkpoint atomically, and follow preemption rules.

## Initial implementation verification (before the full cluster audit)

18 targeted tests passed, including trace isolation, official-span mapping,
rank-deficient/constant inputs, a reader that refuses label members, resume
identity, and v2 integrity barriers. A three-answer Qwen3-4B/GSM8K smoke test
completed in 7.36 seconds (single local worker; competing Claude workload).
All feature diagnostics are descriptive. For the 269-token sampled answer,
32-token windows give 8 fitting rows, 29 varying features and rank 7.
Trace length is explicitly constant; some energy minima also become constant
at wider windows. No feature is silently replaced by a short-window placeholder.

Among all 400 GSM8K answers, 220 have at least eight full 32-token windows;
66 do at width 48, 13 at width 64, and 2 at either 96 or 128. Eight is an
engineering floor to explore, not a proven requirement. This motivates testing
the pooled fallback without declaring that low-rank single-answer fusion fails.

## Completed AIRCC feasibility check

The final run is job 247840, exact scientific capsule from commit `6dea2f79`,
under `/shared/cycle3_tau_averbuch_prj/omrisegev1/window_feasibility_20260906_v1/feasibility_gsm8k_full_v3/`.
All 400 Qwen3-4B/GSM8K answers completed with zero extraction errors and zero
rank-cap violations. Computation took 10.815 seconds on eight CPUs; total job
wall time was 21 seconds. Python 3.12.3, NumPy 2.2.4, SciPy 1.17.1; no GPU.
Memory accounting was unavailable. Source/input/config hashes and aggregate
diagnostics are in `docs/reviews/window_feasibility_2026-09-06.json`.

Width 32 has median N=8, active P=29, rank=7 and participation rank=3.651.
The length feature is constant in all 400 matrices; `min_spilled` is constant
in 166. Wider windows have fewer fitting rows and more constant minima.
The full feature check has not yet been repeated on the other cells.

Jobs 247835 (30-answer pilot) and 247838 (first 400-answer check) are retained
as earlier diagnostics. Comparing the shared 30 answers locally and remotely
exposed floating-point residual centering: three two-window matrices could
be reported as rank two despite the rank-one centered-data bound. The fixed
diagnostic recenters after scaling and enforces the N-1 bound. The corrected
local/cluster comparison agrees on all categorical diagnostics, with maximum
participation-rank difference 3.56e-15. No feature definition changed.

The initial 30-answer compute timing was 22.723 seconds with one local worker
versus 1.229 seconds with eight AIRCC workers. This is a practical comparison
including platform/startup and competing local-work effects, not a general
parallel speedup measurement. The first job also spent time installing its
environment; later jobs reused it.

27 targeted tests pass across the new window/reader/integrity tests and the
existing trajectory-reducer tests. The HTML has 212 valid local/anchor links,
no duplicate IDs, working paper filters and no horizontal overflow at width
430. Prepared v2 repairs are committed but not applied to the running study.

## Outstanding work requiring the named fusion arm

Implement and test the explicitly chosen canonical fuser on both fitting
scopes, keeping the same feature representation and grouped split. Check
boundary perturbations, held-block stability, admissible regularization and
short-answer coverage. Freeze a label-free width/fallback rule before scoring.
Keep absolute no-error calibration separate from relative within-answer risk;
state any use of shared training data or labels. Then evaluate PB first-error/
all-correct and PRMB step ranking under their separate contracts. A pooled
fallback can help insufficient fitting data but cannot restore features that
are not computable on an extremely short trace.
