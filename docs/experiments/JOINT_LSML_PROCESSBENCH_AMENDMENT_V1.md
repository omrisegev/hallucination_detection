# Joint L-SML ProcessBench coverage amendment v1

Status before ProcessBench label access: `REGISTERED_POST_PRM_OPENED__PB_LABEL_FREE`.

## Authority and scope

The parent experiment correctly returned `STRUCTURAL_NO_SCORE` for ProcessBench
because one of eight cells had no admissible learned partition. The user then
explicitly required a ProcessBench comparison. This amendment supersedes only
the parent all-eight no-score disposition. It does not alter or reinterpret the
already-open PRMBench result, and it creates no promotion or generalization claim.

PRMBench labels and its `HARM` result are already open. ProcessBench labels must
remain unopened until this amendment, its complete eight-cell score freeze, and
an independent score-freeze audit are hash-bound. The result is therefore
retrospective development evidence and must be disclosed as post-PRM-opened.

## Frozen candidate policy

There is one candidate and three unchanged controls. On each structurally
admissible cell, the candidate uses the parent Joint L-SML weight vector exactly.
On the sole parent-blocked cell, `processbench_math_qwen3_4b`, it uses the
already-tested `G=[]` dispatch to repository flat SML, followed only by the same
label-free global sign gauge toward mean active-23 standardized confidence. The
flat alias is bit-exact before that allowed global sign choice. This is a
coverage policy named
`joint_lsml23_hierarchical_v1_1__flat_sml_structural_fallback`; it must not be
reported as pure Joint L-SML on all eight cells.

The controls remain `iu_pcr_active23`, `equal_family_active23`, and
`fixed_family_continuous_lsml_active23`. Their blocked-cell weights are fit from
the same target-free active-23 standardized donor matrix with the same parent
implementations and frozen parameters. No label, threshold, subset outcome, or
PRMBench result selects a feature, group, weight, or reducer.

## Frozen ProcessBench adapter

- Population: the same eight Qwen3 ProcessBench cells and 3,400 paired source
  questions used by the parent protocol.
- Detector: maximum token risk within the response.
- Locator: argmax across steps of the mean of the largest
  `min(10, step_length)` token risks. This is fixed top-10, not top-5 and not
  top-10-percent.
- Calibration: five grouped folds; every bootstrap replicate refits each
  method/model threshold on four folds and applies it to the held-out fold.
- Primary metric: equal mean across the two scorer-model macro-F1 values; each
  model macro is an equal mean over GSM8K, MATH, OlympiadBench, and OmniMath.
- Uncertainty: 2,000 paired source-question bootstrap draws, stratified by
  subset and frozen fold, seed `2026090408`.

The primary analysis is the all-eight coverage-preserving policy. A single
selection-conditioned diagnostic reuses the primary all-eight OOF predictions
and fitted thresholds, excludes the fallback cell only at aggregation time, and
reports the equal-cell mean F1 across the seven parent-admissible cells. It gets
no interval and is not complete-panel efficacy or fallback-independent: its
thresholds were calibrated by the primary all-eight procedure and can therefore
be influenced by the fallback cell. Per-cell OOF F1 values must mark the fallback
cell. No thresholds are refit for this diagnostic.

This is the second opened candidate policy: pure Joint L-SML was already exposed
on PRMBench and returned `HARM`; the current Joint-or-flat coverage policy is a
new PB-only policy. There are three primary descriptive contrasts and one
secondary selection-conditioned diagnostic. This cumulative exposure is part of
the ledger and prevents treating the PB result as a fresh confirmation.

## Gates and artifacts

Registration binds this document, the plan, code, runtime, parent registries,
parent structural ledger, parent result audit, orientation/roster artifacts, and
the sanitized input manifest. It also binds the exact seven parent weight-vector
hashes, the sole blocked cell/status, and a fallback count of one. The score
freeze must contain all eight cells,
exactly four methods in the registered order, unique complete row IDs, finite
detector scores, and valid locators. It must claim `processbench_labels_accessed=false`.

Only a new independent audit with exact score-manifest hash may authorize the
evaluation registry. After the registry binds the canonical ProcessBench label
file by hash but before it parses labels, a second independent registry audit is
required. Results are states
`DEVELOPMENT_SUPPORTED`, `INCONCLUSIVE`, or `HARM`; all intervals are descriptive.
The support rule is deliberately stricter than the parent rule because this is a
second candidate policy designed after a prior opened panel: the candidate-IU
point must be positive and the lower interval bound must be nonnegative against
all three controls. Harm remains a wholly negative candidate-minus-IU interval;
all other cases are inconclusive. A post-result independent audit is required
before reporting.
