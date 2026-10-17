# Prior-order audit: Joint L-SML optimization v2

Date: 2026-09-05
Worktree: `hd_jlsml_v2_wt` (sparse checkout)
Branch: `claude/joint-lsml-optimization-v2`
Base commit: `0ce4896` (tip of `codex/joint-lsml-localization-eval-v1`)
Protocol: `docs/experiments/JOINT_LSML_OPTIMIZATION_PLAN_V2.md`

## Authority

Omri's directives of 2026-09-05 authorize, on the already-opened Qwen ProcessBench/PRMBench
development populations:

1. a symmetric tuned-vs-tuned development study (16 IU configurations vs 16 Joint/L-SML
   configurations, nested 5-outer/5-inner source-group CV);
2. improving the Joint L-SML derivation with **all four** DUFS gate-integration mechanisms
   (gated-SML congruence; LIU transplant on the joint model-covariance map; diagonal gate
   prior; gated grouping affinity), each with identity rows and permutation controls;
3. two named incumbent controls: IU-PCR (localization control config) and the deployed U-PCR
   port (exclusion+refit);
4. freezing the per-task tuned winner for the separate fresh-population experiment, with the
   label-free rows S1/S2 co-reported;
5. Module B: a learned trajectory-axis reducer over step order statistics, with the frozen
   top-10 mean as control and a supervised LR competitor;
6. keeping the opened populations open afterward (no closure clause; exposure ledger is
   bookkeeping only).

## Explicit supersessions of frozen prior text

| Prior text | Supersession | Authorized by |
|---|---|---|
| v1 plan fusion-order axis | dropped; one PRMB-only diagnostic row | Omri 2026-09-05 (e) |
| v1 plan 8-vs-8 budget | 16-vs-16 | Omri 2026-09-05 (budget answer) |
| Step-225/343 one-DUFS-mechanism default; Step-349 soft-affinity slot | four mechanisms in one study, each with lambda=0 identity + permutation control + the Section-7.3 attribution rule | Omri 2026-09-05 (hooks answer) |
| Step-347 all-eight PB rule | per-lane provenance fallback (same map type), strict aggregation co-reported | Omri 2026-09-05 + SD=1 invariant removing the splice hazard |
| v1 "last retrospective study" / closure clause | removed | Omri 2026-09-05 ("we can keep opening qwen") |

## Repository-order mapping

- `CLAUDE.md`: work in an isolated claude/-prefixed worktree; frozen result namespaces
  untouched; ASK-which-method satisfied (Omri named the comparison targets explicitly).
- `PROGRESS.md` / `HISTORY.md` Steps 347-349: their artifacts and verdicts are immutable; the
  Step-347 frozen candidate is never re-scored; v2 re-tests the *repaired* estimators only.
- `Research_Directions.md`: prior negative conclusions (DUFS ranker, transplanted keep rules,
  adaptive-K, graph smoothing on the step axis) remain closed; the reopenings above are scoped
  to coefficient-space gating and the trajectory-axis reducer.
- `SUPERVISED_ORACLE_CORRECTION.md`: governs the Module-B LR competitor.
- C-v2 / joint_lsml_v1_r2 lineage: read-only.

## Forbidden actions

- No label or outcome access before the score freeze of the arm being evaluated; the evaluator
  runs as a separate firewalled process.
- No edits to `results/joint_lsml_*`, existing `configs/joint_lsml_*.json`,
  `spectral_utils/joint_lsml_processbench_amendment.py`, `spectral_utils/dependency_fusion.py`,
  or HISTORY entries of prior steps.
- No re-scoring of the frozen Step-347/348 candidate object.
- No gate-derived K selection; no ranking by |mu|; no reducer replacement outside Module B's
  registered rows.
- No cross-panel averaging; no promotion or generalization claim from this study.

Any violation fails the run.
