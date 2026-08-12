# Specification: Frozen CB-CS-IU transfer to SemGrad v1

**Frozen:** 2026-08-12, before evaluating any CB-CS-IU score against a SemGrad
BEM label.

**Status:** independent-example, independent-benchmark transfer with a disclosed
historical-label-visibility limitation.  Ordinary IU/DUFS results on these
cells already existed in the repository before this protocol; CB-CS-IU was not
developed, tuned, or selected on either cell.

## 1. Question

Does the fully frozen, label-free Cardinality-Balanced Contribution-Subspace IU
(CB-CS-IU) rule improve ordinary IU-PCR on new answer-level examples from the
SemGrad protocol, without changing inference, features, or fusion strength?

This is the strongest locally available independence test:

- new benchmark protocol relative to the 24-cell development bundle and
  ProcessBench;
- new generated answers and new Qwen3-4B-Instruct-2507 telemetry;
- paper-faithful BEM answer-equivalence labels;
- the CB formula was frozen before CB scores were paired with these labels.

It is not a pristine blinded benchmark because the repository already contains
the BEM labels and earlier IU/DUFS results from a different compact task
adapter.  No further method selection is permitted after this report.

## 2. Frozen data roster

Two READY datasets from `results/data_readiness_2026_08_11/`:

1. `local_cache/semgrad_bem_regraded/raw_semgrad_sciq_T0.0_bem.pkl`
   (1,000 distinct questions);
2. `local_cache/semgrad_bem_regraded/raw_semgrad_truthfulqa_T0.0_bem.pkl`
   (817 distinct questions).

Both use one greedy output per question from
`Qwen/Qwen3-4B-Instruct-2507`.  The registered grader is BEM
(`https://tfhub.dev/google/answer_equivalence/bem/1`) at threshold 0.8, with
the maximum score over accepted answers.  Input and regraded-output hashes must
match the BEM manifests and the data-readiness registry.

No row or dataset is excluded.  The primary risk target is

```text
bem_error = not candidate["bem_correct"].
```

The original temporary ROUGE-L `label` and continuous `bem_score` are never
used for fitting, selection, or primary evaluation.

## 3. Frozen feature and fusion contract

Use the same one-pass registered mixed-v2 construction as the Qwen3 and Llama
ProcessBench transfers:

- `extract_all_features(..., allow_short=True)` on the existing token entropy
  and sampled-token-energy trajectories;
- existing log-partition and top-k summary functions;
- at least 70% finite availability;
- unlabeled median fill;
- remove standard deviation below `1e-8` and median saturation above 40%;
- apply the frozen `dufs_liu_mixed_v2_matrix` transform.

The short-trace fallback is an existing feature-extraction path and performs no
additional inference.  The pre-label readiness audit found 19 retained
features on SciQ and 20 on TruthfulQA, across the same six registered provenance
families.  Those counts are descriptive and do not change the method.

Fit ordinary IU-PCR with `IU_FIT_DEFAULTS`.  Apply the exact public
`cardinality_balanced_iu_fit` algorithm frozen in
`SPEC_CARDINALITY_BALANCED_CS_IU_V1.md`:

```text
d_g  = mean_h(log m_h) - log m_g
s_CB = standardized_IU + (1/G) * R d / std(R d).
```

Primary and frozen comparisons:

1. ordinary IU-PCR;
2. CB-CS-IU (primary);
3. LB-CS-IU (previous frozen variant; mechanism comparison only);
4. mixed-v2 DUFS-LIU, seeds 11/23/37, 80 epochs, graph `k=7`, `lambda=0.1`;
5. uniform contribution-residual direction;
6. reversed cardinality direction.

All confidence scores are negated once to obtain hallucination risk.

## 4. Physical label boundary

`fit` must construct a new telemetry-only dictionary for every candidate that
contains only:

- `token_entropies`;
- `token_spilled_energies`;
- `token_logsumexp`;
- `top_k_logprobs`.

It must not access or copy `bem_correct`, `bem_score`, or `label`.  It then saves
and hashes every score, original data file, BEM manifest, data-readiness
registry, source file, effective weight vector, intercept, and numerical
diagnostic.

`report` must verify all hashes before opening `bem_correct`.  It must verify
row IDs and the registered BEM class counts before calculating metrics.

## 5. Metrics and frozen gates

Primary metric: full-dataset AUROC for BEM error.  Use 20,000 deterministic
paired, class-stratified row bootstrap draws within each dataset.  Also report
an equal-dataset hierarchical bootstrap: resample the two datasets, then draw
from their saved paired-bootstrap delta distributions.

The independent-example transfer passes only if all hold:

1. CB-CS-IU AUROC delta over IU is positive in both datasets;
2. the equal-dataset hierarchical 95% interval has a lower bound above zero;
3. no dataset falls below IU by more than 1.0 percentage point (redundant if
   gate 1 passes, retained as the frozen safety gate);
4. CB exceeds the reversed direction in both datasets;
5. contribution/effective-weight reconstruction, IU-correction covariance,
   `1/G` scale, and IU identity errors are all below `1e-10`.

CB-minus-LB, CB-minus-DUFS, and CB-minus-uniform are reported without becoming
new selection gates.  If they contradict the proposed multiplicity mechanism,
that contradiction must be stated even when the primary efficacy gates pass.

## 6. Claim boundary

A pass supports CB-CS-IU as the requested non-supervised, fusion-internal
algorithmic addition across independent answer-level benchmark examples.  It
does not erase the historical visibility of BEM labels and prior baseline
scores.  Publication-grade confirmation should still add a future benchmark
whose labels have never been inspected by the project.
