# Window-localization feasibility and fit-scope study

Status: IMPLEMENTATION / METHOD SELECTION PENDING. Started September 6, 2026.
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
7. Complete registered evaluation only after the method choice and integrity checks. Report coverage, accuracy, stability, runtime and fit scope separately.
8. Update the HTML review and handoff with concrete outcomes and remaining limitations.

## Worktree and cluster observations

The initial full worktree checkout attempted to duplicate tracked cache pickles and filled the disk. It was stopped; Git rolled it back, including the duplicate tree. No original caches were removed. Recreated successfully with `--no-checkout` plus sparse patterns selecting code, tests and documents. About2.49GB free remained. All large inputs must be shared read-only; do not archive the entire Git tree for cluster transfer.

Live local Claude was active at27/45 outer folds, four workers, no evaluation directory. Local CPU is an i5-1035G1 with4cores/8threads and16GBRAM. Keep local development light.

AIRCC became reachable after the user enabled VPN. Slurm requires a login shell: `/etc/profile.d/slurm-configless.sh` sets `SLURM_CONF_SERVER=controller-primary`. Current account is `cycle3_tau_averbuch_prj`, partition`power-gpu`, QoS`owner_940`; project root `/shared/cycle3_tau_averbuch_prj`. User queue was empty. Read-only scheduling validation accepted CPU-only32CPU/64GB/2h and8CPU/32GB/10min requests; these were test-only, not submitted jobs. Nodes have160CPUs/~1.8TB. No measured speedup yet; CPU parallelism helps independent traces. No GPU required for NumPy/SciPy fitting.

Existing PB and PRMB raw telemetry remains readable under `/shared/cycle2_tau_averbuch_prj/omrisegev1/results/`. New runs must use an isolated non-home cycle3 directory, not overwrite old code/results. Native dependency readiness is unverified; check inside a compute job. Keep BLAS threads1 per worker, checkpoint atomically, and follow preemption rules.

## Initial implementation verification

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
