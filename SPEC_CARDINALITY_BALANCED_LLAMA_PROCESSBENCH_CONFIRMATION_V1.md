# Specification: Frozen CB-CS-IU transfer to Llama ProcessBench v1

**Frozen:** 2026-08-12, before reading any per-row Llama score/label pairing.

**Status:** scorer-family confirmation.  The underlying 3,400 ProcessBench
reasoning chains and labels are the same examples used in the earlier Qwen3
study, but all telemetry in this test comes from a previously unused
Llama-3.1-8B scorer.

## 1. Question

Does the already-frozen Cardinality-Balanced Contribution-Subspace IU
(CB-CS-IU) correction transfer unchanged from Qwen3 telemetry to a different
model family's one-pass telemetry?

No formula, family registry, feature construction, scale, or exclusion is
selected on the Llama cell labels.

## 2. Frozen data roster

Source:
`gdrive:hallucination_detection/cluster_results/pb_llama31_8b_full/`, copied
without modification to `dataset_cache/repgrid/pb_llama31_8b/` after its path,
size, modification time, and manifest were inspected.

Scorer: `meta-llama/Llama-3.1-8B-Instruct`.

Cells: the full ProcessBench `gsm8k`, `math`, `olympiadbench`, and `omnimath`
subsets.  The upstream manifest records 400/1000/1000/1000 aligned rows and no
unmapped steps.  No cell is excluded.

The manifest contains aggregate class counts, which were visible during data
readiness.  Per-row labels and their pairing with the new scores remain closed
until score freezing.  This limits blinding but cannot select score rankings.

## 3. Frozen score construction

Use exactly the global mixed-v2 answer feature construction from GL-LIU v1:

- the registered one-pass trace features;
- at least 70% finite availability;
- median fill using the unlabeled batch;
- remove standard deviation below `1e-8` and median saturation above 40%;
- apply `dufs_liu_mixed_v2_matrix` with no changes.

Fit ordinary IU-PCR with the repository's `IU_FIT_DEFAULTS`.  The primary
candidate is exactly `spectral_utils.cardinality_balanced_iu_fit` as frozen in
`SPEC_CARDINALITY_BALANCED_CS_IU_V1.md`:

```text
d_g  = mean_h(log m_h) - log m_g
s_CB = standardized_IU + (1/G) * R d / std(R d).
```

`m_g` is the registered feature count of provenance family `g`; `R` contains
standardized family contributions residualized against standardized IU.  No
target or dataset identity enters the fit.

Frozen comparisons:

1. ordinary IU-PCR;
2. CB-CS-IU (primary);
3. LB-CS-IU using the already-frozen L1-leverage direction;
4. unified mixed-v2 DUFS-LIU at `lambda=0.1`, with DUFS seeds 11/23/37,
   80 epochs, and graph `k=7`;
5. uniform residual-direction control;
6. reversed CB direction as a falsifier.

All answer scores are converted to risk by negation before evaluation.

## 4. Target and uncertainty

Primary target:

```text
reasoning_error_present = (row["label"] != -1)
```

Metric: full-cell AUROC.  Report cell-macro AUROC and the mean of four subset
deltas.  Its 95% interval is a deterministic 20,000-draw paired bootstrap over
the four subsets.  The resampling unit is subset, not row, because the rows are
the same ProcessBench examples seen under other scorer models.

## 5. Physical label boundary

`fit` must:

- load aligned telemetry rows without accessing `label` or
  `final_answer_correct`;
- fit and save all scores plus exact effective CB/LB weight vectors;
- hash each data file, score file, and relevant source file;
- record numerical reconstruction, orthogonality, and trust-scale invariants.

`report` must first verify all hashes, then and only then load the primary
target.  A source mismatch invalidates the report and requires a new fit.

## 6. Frozen gates

The scorer-family transfer passes its primary efficacy gate only if all hold:

1. CB-CS-IU mean subset delta over IU is positive;
2. the paired subset bootstrap lower bound is above zero;
3. CB-CS-IU wins at least three of four cells;
4. no cell falls below IU by more than 1.0 percentage point;
5. contribution reconstruction, effective-weight reconstruction,
   IU/correction covariance, and `1/G` scale errors are all below `1e-10`.

Mechanism comparison is separately reported:

- CB-CS-IU minus LB-CS-IU;
- CB-CS-IU minus DUFS-LIU;
- primary versus uniform and reversed controls.

These comparisons do not trigger another method selection.  A negative
CB-minus-LB result weakens the cardinality-specific interpretation even if the
primary efficacy gate passes.

## 7. Claim boundary

A passing result confirms transfer to a new telemetry/scorer family without
new inference at deployment and without labels during fitting.  It is not an
independent-example confirmation because the reasoning chains and labels are
shared with the earlier Qwen3 ProcessBench study.  A genuinely new benchmark
family remains the strongest final confirmation.
