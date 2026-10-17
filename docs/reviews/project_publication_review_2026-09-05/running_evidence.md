# Running experiment: read-only evidence memo

Snapshot: 2026-09-05 20:48–20:52 Asia/Jerusalem (+03:00). Last explicit clock reading: 20:51:37. Inspection only; no experiment was started, stopped, changed, or evaluated. No label sidecar or result score array was decoded. SHA checks read bytes only.

## What is actually running

- Worktree: `C:/Users/omris/TAU/hd_jlsml_v2_wt`; branch `claude/joint-lsml-optimization-v2`; HEAD `ff800082`.
- Local Windows Python parent PID **137068**, started **17:46:32** local time. Command: `scripts\\joint_lsml_optimization_v2\\run_v2.py structure --workers 4`.
- Four live Python child workers: **145340, 141528, 139628, 143228**, all started 17:46:37. Each had substantial accumulated CPU time and about 0.30–0.38 GB working set. Their parent was alive.
- This is a **local CPU fit on existing telemetry**, not new GPU model inference. The runner sets PyTorch threads to one; DUFS uses ordinary CPU tensors. It reads existing caches from the main checkout. No local evaluator, inference, Docker, WSL, or transfer process was visible in the inspected process subset.
- **16/45 outer folds had COMPLETE.json**: GSM8K/Qwen3-4B 5/5; GSM8K/Qwen3-8B 5/5; MATH/Qwen3-4B 4/5; MATH/Qwen3-8B 2/5. The 45 are 8 ProcessBench cells x 5 outer folds plus PRMBench x 5. Each outer fold also has 5 inner fits. These counts are work completion, not scientific results.
- OlympiadBench q4/q8 already had in-progress inner artifacts; latest observed file write was MATH q4 outer4/inner1 at 20:44:16. `_structure2.log.err` was zero bytes. This supports active progress, not a completion or correctness verdict.
- `_structure.log` records earlier completed folds; `_structure2.log` records the current resumed process. The most recent whole-fold log line was MATH q8 outer1. A sparse log between completed folds is expected; inner-file timestamps are a better liveness indicator.
- Amendment R1 second-pass manifests existed for GSM8K q4 outer0 and outer1 only. No second-pass process was visible at the snapshot. The additive amendment pass is therefore not globally complete.
- No `evaluation/` directory existed. Presence of the separated `labels/` directory is not outcome evaluation: the loader extracts labels into sidecars, while structure uses telemetry bundles. No labels were opened by this review.
- AIRCC status workflow connectivity pre-check (`ssh -o ConnectTimeout=5 -o BatchMode=yes aircc 'echo ok'`) timed out; inspection stopped per the project workflow. **GPU queue state is unknown**, not empty. TAU VPN/connectivity is needed to check it. No current GPU job ID was established.

## Recorded status versus current state

`PROGRESS.md` on this worktree still opens with Step 349 (September 4). Its v1 draft says 8-vs-8 configurations and one possible DUFS successor. The actual September 5 protocol explicitly supersedes that with a 16-vs-16 study, four DUFS mechanisms, per-lane provenance fallback, and a learned trajectory reducer. A reader following only PROGRESS would miss the live experiment and its authorized scope.

Authoritative current documents:

- `docs/experiments/JOINT_LSML_OPTIMIZATION_PLAN_V2.md`
- `docs/experiments/JOINT_LSML_OPTIMIZATION_V2_AMENDMENT_R1.md`
- `docs/experiments/PRIOR_ORDER_AUDIT_JOINT_LSML_OPTIMIZATION_V2.md`

Recent commits record protocol+core (`601db6c6`), runners (`b2d64aad`), a claimed 21/21 pre-label audit (`84207568`), and pre-label R1 amendment (`ff800082`). This review did not re-run the audit because it is expensive and its implementation can write a fold golden when missing. The commit's audit claim is recorded evidence, not a fresh independent pass.

## What the experiment can answer

All Qwen ProcessBench and PRMBench populations are already-opened **retrospective development** data. This run can assess whether repairing Joint L-SML's score scale, grouping/map choices, DUFS coefficient integration, or trajectory reduction improves held-out-within-development performance. It cannot establish fresh-population generalization or promote a new leader.

Shared invariant: every learned score has donor SD=1, training-only preprocessing, shared orientation, and a small-m SML guard. INTERNAL grouping falls back to provenance grouping with the same map type, not flat SML.

- Module A compares a fully crossed 16-IU roster (components 1/2, scale .25/.10, loss l2/l1, exclusion+refit off/on) with 16 Joint/L-SML configurations. The latter include provenance/INTERNAL x continuous/hierarchical maps, four DUFS integration hooks, and a disclosed historical hard selector. Nested grouped CV is 5 outer x 5 inner; PB paired scorer copies share source-question folds; PRMB uses problem groups.
- Pre-fixed S1=`internal_joint` and S2=`internal_cont` remain separately reported label-free estimators. A winner chosen using labels is label-selected development even if its future fitting recipe consumes no labels.
- Named controls: deployed IU-PCR, deployed U-PCR exclusion+refit port, equal-all23, equal-family, and guarded fixed-family continuous L-SML. R1 adds an unguarded historical-estimator continuity row followed by the new SD/orientation boundary.
- Module B learns weights across top-10 token-risk order statistics. R1 adds a 3x3 grid: IU/continuous/Joint feature-axis substrate x SML/IU/Joint trajectory fuser. The one primary contrast is inner-selected best grid combination versus the frozen reducer on the same substrate; other grid cells are descriptive. A balanced LR on the same order statistics is a supervised comparator.
- Module A retains PB's top-min(10,step_length) locator and max-token answer detector; PRMB uses official-span max. Module B studies reducer changes separately. PB macro-F1 and PRMB step AUROC are never averaged.

## Registered opening and stopping rules

Structure completion is not permission to skip the independent pre-label audit. Before the separate evaluator: freeze every structure and R1 score artifact, verify source/config/fold hashes and immutable namespaces, verify aliases/firewall, and inspect real-data grouping/fallback/stability guards. Then run independent post-label recomputation after evaluation.

- Development support: paired CI lower >0 **and** delta >=.010 PB macro-F1 / >=.005 PRMB AUROC. Both panels must pass for joint support; only one yields partial-task support.
- Fresh-data non-inferiority: CI lower >-.005 PB / >-.0025 PRMB relative to deployed IU. Additional stability/activation guards still apply.
- INTERNAL fallback >10/40 PB lanes or >1/5 PRMB lanes requires a registered amendment before labels and bars S1 fresh-data eligibility under the original rule.
- PB activation must be at least max(.10, half fixed-family activation) per lane. At least two violating cells implies catastrophe and bars freezing. Low map agreement/cross-fold cosine, unstable gates, non-finite/undefined scale or orientation also bar the relevant arm.
- Gates that leave every weight cosine >=.995 are inert; a gated win needs its matching permutation control to fail the same gate before attributing benefit to that mechanism.
- Fresh scorer is provisionally Phi-4-reasoning-plus, revision unresolved. Fresh GPU generation is **not authorized by this protocol**. A separate fresh-population registration/execution is required.

## Verified pre-evaluation integrity blockers

These are engineering/provenance findings, **not evidence that the numerical candidate scores are wrong**. Leave the live process untouched; address them through a disclosed pre-label integrity amendment before evaluator execution.

### 1. Frozen manifest loses four of five inner folds (high severity)

`scripts/joint_lsml_optimization_v2/run_v2.py:385` builds the recursive manifest using `path.name`, so five `inner*/scores_inner.npz` paths overwrite one key and five `meta_inner.json` paths overwrite one key. `stage_check` at lines 422–428 then finds all matching basenames and hashes only the first match.

Verified using SHA-only reads of completed `structure/pb_gsm8k_q4/outer0`:

| File | Physical files | Manifest entries | Registered hash matches | Checker first match |
|---|---:|---:|---|---|
| meta_inner.json | 5 | 1 | inner4 only (af4cdc9b1ac6075d...) | inner0: mismatch |
| scores_inner.npz | 5 | 1 | inner4 only (7270c42ebfaeff6d...) | inner0: mismatch |

Thus the planned final check fails on a completed lane, and four inner folds are not individually bound even if that mismatch were bypassed. Recommended repair: preserve original manifests, add a versioned manifest keyed by relative path covering all outer/inner/R1 artifacts, verify those exact paths, and test completeness/missing-file/hash-drift rejection. Do not rewrite frozen historical manifests or merely change the checker to the last matching basename.

### 2. Required source/config execution registry absent (high severity provenance gap)

Protocol Section 8, item **1**, lines 294–295 requires source and config hashes in `results/joint_lsml_optimization_v2/EXECUTION_REGISTRY.json` at registration, **before** implementation/structure, not after completion. The file was absent. `configs/joint_lsml_optimization_v2.json`, referenced by the protocol and file map, was also absent locally and from `git ls-tree HEAD configs`; current constants/rosters reside in Python source. Fold JSON and its golden hash did exist.

Recommended repair: record an honest late pre-label registry/amendment binding actual committed source, exact in-code config, fold maps, loader/input provenance, and both original/R1 score manifests. Preserve available launch/resume commit evidence. Do not backdate the registry or imply the original preregistration artifact existed. Assess whether code changes during the resumed run affected imported estimator code; preserve completed scores while investigating.

### 3. Evaluator does not enforce its opening precondition (high severity if invoked)

`scripts/joint_lsml_optimization_v2/evaluate_v2.py:3` says it runs only after structure+check, but `main` lines 417–422 creates the evaluation directory and loads all labels immediately. It does not call a read-only completion/hash/registry preflight before `_labels`. Recommended repair: fail closed before any label load unless every required fold, base and R1 manifest, source/config/fold hash, and independent pre-label decision exists and passes. The absence of an evaluator process or output directory at this snapshot means this review found a preventable opening hazard, not evidence that outcomes were already evaluated.

## Practical next action

Allow the authorized CPU structure work to finish, while preparing and independently reviewing the additive manifest/registry/evaluator-preflight amendment. Resolve these mechanical issues before label-stage evaluation. Then judge v2 under its frozen separate-panel gates. The publication plan should reserve fresh-data confirmation and a small frozen roster regardless of how attractive the development leaderboard looks.
