# Localization source-question grouping repair v2

Date: 2026-09-07. This is a benchmark integrity correction and score bridge,
not a new detector or an untouched evaluation.

## Trigger

The exposure audit found that PRMB's cached `source_idx` is the perturbation
record identifier. It retains prefixes such as `confidence` and `circular`.
Different such records can have the same source seed and identical question.
ProcessBench also contains different answer IDs for identical problem text.
These identifiers cannot establish source-question-disjoint folds or cohorts.

The official PRMB format separates original and modified questions and keeps
the perturbation category in `idx`:
https://github.com/ssmisya/PRMBench#-data-format-for-prmbench
The exact local cache, not a current remote revision, defines this repair.

## Corrected identity, independent of labels

PRMB: extract the complete `prm_train_pN_I` or `prm_test_pN_I` suffix from
the cached source ID. Preserve train/test and partition indices. Connect rows
that share this seed, or the SHA256 of their question after collapsing only
whitespace. Connected components define source groups. Sharing a seed links
the intentional modifications of a question. The text link also catches
identical questions with different seed IDs. Do not normalize numbers or
mathematical notation. This does not claim to identify every paraphrase.

PB: identical whitespace-normalized problem text defines a group. Exact
question text is checked between the 4b and 8b caches for the same IDs. Share
PRMB component IDs when the same text occurs in both tasks. All source text
is read from project caches; labels in those containers are not used.

Retain every row, score, target, official span and access/feature contract.
Add `legacy_group_id`; replace only `group_id` in a new immutable release,
`localization-cached-v2-sourcegroups-20260907`. Preserve the v1 release and
every historical experiment. The raw telemetry hashes remain identical.
Both benchmarks use one teacher-forced model pass over an existing official
answer; this is not generation of a new answer. Gray-box access is unchanged.

## Frozen score bridge

Reuse all 25 frozen fallback-pilot methods on the same 58 answers. Verify
all scores, targets, validity flags, decisions and metric bundles unchanged.
Recompute the 30 existing paired contrasts plus the two already identified
context-equal comparisons under corrected source groups: 32 bridge pairs.
Use the existing 1,000-draw cell-stratified source-group bootstrap and seed
2026090706; retain undefined draws. Audit that no PB corrected group spans
multiple PB subsets before using cell-stratified resampling.

Because the old RNG draws PRMB before PB, changing the number of PRMB groups
changes the subsequent PB resamples. A small PB interval change can therefore
be Monte Carlo variation even when its 46 pilot groups remain distinct.
All bridge intervals are retrospective, exploratory and unadjusted.

## New folds, with no claim of repaired trained scores

Produce five outer and five inner folds at corrected global source-group
level. Stratify by the set of benchmark subsets in which each group appears.
Within each stratum, SHA-sort groups and distribute round-robin. The namespace
is the new release ID; inner folds add the outer-fold identity. Restrict the
global assignments to each benchmark while preserving alignment of shared
questions across tasks and of PB questions across 4b/8b models.

Check zero source-question overlap for every outer and inner train/test
split. This prepares a rerun; it does not repair Claude's cached predictions
from old folds. Those fits, supervised configuration selections and scores
must be recomputed before claiming question-disjoint cross-validation.
The answer-only pilot scores need no refit because each answer was fitted
independently; only their resampling units and exposure claims are corrected.

## Review and continuation

Check metadata extraction against standard pickle on synthetic protocol-4/5
fixtures with shared strings, frames and NumPy arrays. Independently rebuild
components with a sparse graph algorithm, rejoin release IDs and folds,
and reproduce representative corrected intervals by explicit row resampling.
Record source hashes, a readable HTML report and the corrected release/folds.

Use the canonical components to exclude all documented earlier short-cycle
and 58-answer source groups before selecting a new development replication.
No new cohort or method is selected in this repair stage. Keep the fixed-recipe
replication prototype separate and unfinished. Untouched publication data,
the broader fusion research mandate and historical 24-cell transfer remain open.
