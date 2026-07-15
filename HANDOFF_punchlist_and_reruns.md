# Handoff: close remaining punch-list items, then re-run stale-data analyses

**Date**: 2026-07-14
**Purpose**: planning input only — nothing below has been executed except where explicitly
marked DONE. This is the output of a HISTORY.md-wide review (all 181 steps, cross-referenced
with PROGRESS.md and Research_Directions.md) plus an inspection of `results/repgrid/` vs
`results/subset_sweep/`. Feed this to a fresh planning session via the prompt block below —
do not re-derive the punch list from scratch, it's already done here.

---

## Paste this to start the next session

> Read `HANDOFF_punchlist_and_reruns.md` in full, then enter plan mode. Design a plan that:
> 1. Sequences **Phase 1** (closing the still-open punch-list items below) before **Phase 2**
>    (the two stale-data re-runs) — do not interleave them.
> 2. For Phase 1, group items by cost (CPU-only / blocked-on-a-Drive-pull /
>    needs-cluster-or-Colab-run / structural-no-GPU) and propose an order within Phase 1.
> 3. For Phase 2, design the repgrid-cell loader adapter needed before either the
>    feature-subset search or the LR-oracle re-run can run, and scope how much of it is shared
>    between the two.
> Ask me which Phase-1 items I want in scope before finalizing — several need a cluster or
> Colab run and I may want to defer those to a separate session.

---

## Already done (context — do not redo)

- **Step 181** (HISTORY.md): 4 of the 8 Step-158 Phase-15 CPU follow-ups closed — K-sweep,
  spilled-energy + extended-logprob feature orthogonality, fairer diversity set B′,
  cross-temperature probing. New `scripts/phase15_followups.py` +
  `spectral_utils/repgrid_scoring.py::logprob_features_extended`. Results:
  `results/repgrid/phase15_followups.json`.
- Same session: `PROGRESS.md`'s stale Step-132-era bottom half cleaned up (dead "pending Step
  132" references, a falsified M=9 feature-set idea, duplicated numbers tables replaced with
  pointers to the canonical CSVs).
- Neither is committed — still sitting in the working tree as of this handoff (project
  convention: "await Omri" for commits).

---

## Phase 1 — still-open punch-list items

### CPU-only, no new data needed
1. **MATH-500 fresh-trace 85.1 vs legacy-cache 94.4 discrepancy** — flagged Step 152, still
   explicitly called "open" at Step 174. Compare trace-length distributions / prompts /
   sampling between `local_cache/math500_qwen7b_T1.0_run0.pkl` (Phase 15, 85.1) and whatever
   produced the Step-135 94.4 headline. Local analysis only.
2. **RAG SelfCheckGPT below-chance orientation** — flagged Step 152, never actually
   investigated past being relabeled "suspected/unresolved" in later reports. Official-variant
   SelfCheckGPT scored *worse* than the hard variant on all 4 RAG datasets, HotpotQA-official
   significantly anti-predictive (CI [0.137,0.357]). Check score orientation/grading in the
   long-context L-CiteEval setting. Local analysis on existing RAG caches.
3. **Streaming earliest-prefix replication (Extension E)** — the Step-148 pilot's only
   significant finding (+9.8pp GSM8K / +4.6pp MATH-500 AUROC at earliest 10% of trace) was
   never replicated on a canonical cache, even though Phase 15 was explicitly run to repay
   that exact data debt (HISTORY Step 158: "closes the Extension E gap that blocked the
   streaming earliest-prefix replication"). The needed trace
   (`local_cache/math500_qwen7b_T1.0_run0.pkl`) is already local — this is CPU-only via
   `spectral_utils/streaming_utils.py` (`prefix_features`, `causal_trajectories`,
   `earliness_index`), not a GPU task. See Research_Directions.md "Extension E" for the full
   pilot writeup and the exact next-steps list (steps 2-3 there are this item).

### Blocked on one Drive pull (same blocker for both)
4. **Anchor/sign robustness across T** (Step-158 follow-up #3) — re-fuse with a `cusum_max`
   anchor instead of `epr`; tests whether Q1's low-T "poor detectability" (Step 158) is a
   fusion artifact rather than a real effect.
5. **Length-controlled AUROC per T** (Step-158 follow-up #7) — partial out trace length to
   confirm the spectral signal isn't just tracking trace length.

   Both need `cache/phase15_temperature/math500_qwen7b_T{0.3,0.6,1.5,2.0}_run0.pkl` copied to
   `local_cache/` — only the 5 T=1.0 runs are local today (`phase15_results.pkl`'s
   `feat_table` only has scalar per-(feature,temp) AUROCs, not per-sample values, which is why
   Step 181 couldn't do these two). Ask Omri for the files before scoping either.

### Needs one cluster/Colab run (new inference)
6. **Item-5 fusion replication on a 2nd cell** — Step 174's 95.2 AUROC fusion result
   (L-SML × answer-agreement SC) is single-cell. Explicit follow-up: a GSM8K/Llama-8B K=5 run
   to get 5 raw same-T passes the way MATH-500/Qwen-Math-7B already has.
7. **Verbalized confidence on 7B+** — null result only ever tested on Qwen2.5-Math-1.5B
   (Step 131, model didn't follow the "Confidence: X" instruction); flagged "expected to work
   on 7B+, untested" ever since, despite dozens of 7B+ models run for other purposes since (none
   of those runs used the confidence-eliciting prompt, so this needs a dedicated inference
   pass — `parse_verbalized_confidence` is already correct and ready).
8. **LapEigvals's own attention-Laplacian reducer** — still `NotImplementedError` in
   `generate_full()` (`capture_attention`/`capture_layer_fft` both raise). We've only ever run
   *our* L-SML on LapEigvals' GSM8K protocol; their own published unsupervised AttentionScore
   number has never been reproduced on our infra, only cited. Needs the on-GPU
   all-layer×head attention-Laplacian → PCA-512 → probe reducer implemented per the Step-156
   protocol card before any of the `lapeigvals_gsm8k_*` cells can carry a self-reproduced Y.

### Structural — never started (no GPU for the first step of either)
9. **Extension A, Conformal Calibration ("the Bracha chapter")** — called "severely
   underrated... should be the explicit thesis endpoint" (Step 39), never begun. **A1**
   (proposed, not yet implemented: `fit_lsml(calibration_batch)` frozen-weights scorer +
   `score_one(features)` true single-sample inference + `decision_report(scores, labels, τ)`
   imbalance-aware detection metrics — recall/precision/F1/balanced-accuracy/TPR@FPR/AUPRC) is
   pure engineering, no GPU, and there are now ~45 repgrid cells plus the full legacy battery
   to calibrate against. Full A1/A2/A3 spec: Research_Directions.md "Extension A".
10. **Extension F, ProcessBench/MR-GSM8K step-level localization** — deferred at Step 167 with
    the explicit condition "revisit after the reasoning replication grid completes." The grid
    completed at Step 172 (desk closed) — the deferral condition is now satisfied. Full spec:
    Research_Directions.md "Extension F".

---

## Phase 2 — re-run stale-data analyses (only after Phase 1 is scoped or done)

**The gap**: `results/repgrid/scores_lsml_upcr.csv` has **20 distinct (dataset,model) cells**
from the replication grid (Steps 160-181) — CoQA, SQuAD v2, TruthfulQA, SciQ, NQ-Open,
HotpotQA/LOS-Net, TriviaQA (SemanticEnergy / EPR / SE-ICLR variants), the Noise-Injection GSM8K
family, the LapEigvals GSM8K sweep, ARS R1-Distill, Internal-States GSM8K. This has **zero
overlap** with the 32 cells (`results/subset_sweep/*.npz`, all sourced from `local_cache/`) that
the exhaustive feature-subset search (Step 154, GOOD_5's validation) and the logistic-regression
oracle (Steps 142-147, the "features are the bottleneck not fusion" conclusion) were run on.
Neither analysis has ever seen this newer, more domain-diverse cell set.

1. **Feature-subset search on the repgrid cells** (higher value — this is GOOD_5's actual
   validation set, and it predates most of the current data). `scripts/run_subset_sweep.py
   --data-dir` defaults to `local_cache/`, and `subset_sweep.iter_cells()` expects the old
   `{gsm8k,gpqa,math500,rag,qa}_res.pkl` feats_dict schema — incompatible with
   `cache/repgrid/`'s candidate-list schema. Needs a loader adapter that reuses
   `spectral_utils/repgrid_scoring.py::load_repgrid_cell` + `subset_matrix` to produce the same
   `(domain, cell_key, feats_dict, labels)` tuples the exhaustive-fit core expects. Question to
   answer: is GOOD_5 still the LOCO-best subset outside the original 32-cell battery, or does
   this newer domain mix favor something else (e.g. does `cusum_max_spilled`, which passed a
   fusion gate at Step 181, or one of the new logprob features belong in an updated default
   set)?
2. **LR oracle on the repgrid cells** (lower cost — do this first within Phase 2). Re-run
   `scripts/logistic_oracle.py` / `oracle_report.py`'s 5-fold balanced-CV recipe (see
   `SUPERVISED_ORACLE_CORRECTION.md` for the mandatory evaluation rules — class-weight
   balancing, the `cross_val_predict` pitfall) on the same 20 cells. Tests whether "features
   are the bottleneck, not fusion" (Step 147) still holds on domains the oracle has never seen,
   particularly the very-short-trace QA cells where L-SML's regime is already known to be
   weaker.

**Explicitly not in scope for Phase 2**: re-running benchmarking (the desk is formally closed,
Step 172 — rerunning would just reproduce the same tally) or regenerating the per-domain
breakdown / QA extension tables (both auto-regenerate from the CSVs already and are current as
of the Step-180 report regen).

---

## Reference

- Canonical repgrid scoring recipe — reuse, do not re-derive: `spectral_utils/repgrid_scoring.py`
  lines 136-171 (`score_subset`). This is the exact recipe that produced
  `scores_lsml_upcr.csv`; both Phase 2 items should match it for comparability.
- `SUPERVISED_ORACLE_CORRECTION.md` — mandatory reading before any LR oracle work.
- Subset defs (`scripts/score_repgrid.py:33-45`): `GOOD_5` = [epr, low_band_power, sw_var_peak,
  cusum_max, spectral_entropy]; `consensus_4` = [spectral_entropy, sw_var_peak, cusum_max,
  cusum_shift_idx]; `STABLE_H9`, `ALL_H16` also defined there.
