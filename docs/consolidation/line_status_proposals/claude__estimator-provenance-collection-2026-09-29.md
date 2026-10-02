# Line status proposal: claude/estimator-provenance-collection-2026-09-29

> **PROPOSAL by Claude, not an Omri decision.** Omri's decision (2026-10-01): write proposed statuses explicitly as proposals, with supporting evidence and reopening conditions; this is not blanket approval to close these directions; distinguish 'this implementation failed' from 'the research direction is closed'. The only explicit closure so far is Joint L-SML (discontinued 2026-10-01).
>
> How to read the status words: they describe the tested implementations and the branch, not the research direction. SATURATED = the implementations tested on current inputs gave no measured gain. IMPLEMENTATION-NEGATIVE = this implementation lost to its matched reference. SUPERSEDED = the branch's content is carried by a newer line. TOOL-ONLY = kept as code or data that other lines use. OUT OF SCOPE FOR NOW and DEFERRED are Omri's words (2026-10-02) and mean "not closed, not a next step". None of these words closes a research direction.
>
> Drafted 2026-10-01 by the consolidation session ae2dd164 (no owner session exists for this line); Omri's 2026-10-02 decisions applied on 2026-10-02.

Proposed status:
- Third-moment method-of-moments estimator (Jaffe et al. 2015) run on the uniform top-20% marks (Step 464): IMPLEMENTATION-NEGATIVE (direction open)
- Estimator provenance note and the two-week collection map (Step 463): TOOL-ONLY

Owner / last updated: no owner session. The branch tip is aa8bf95d5 (2026-09-30), identical on origin. Worktree `.worktrees/tensor-mom-v1`.

## Steps on this branch

The branch is `claude/ssl-pseudolabel-residual-v1` at Step 462 plus two steps of its own.

- Step 463 [Claude, estimator provenance + tensor MoM + collection map] - "DS estimate" in the label-free line is the Dawid-Skene EM of 1979, not the 2015 tensor method the advisor pointed to; the 2015 estimator was implemented (`er_stage_a.tensor_mom_estimate`, the paper's Algorithm 1, 4 tests) but not run; a glossary misattribution (`a1_residual` belongs to the 2016 paper) was fixed. Source: `docs/research_notes/ESTIMATOR_PROVENANCE_2026-09-29_HE.md`, `docs/HANDOFF_COLLECTION_2026-09-29.md`.
- Step 464 [Claude, tensor_mom_v1] - On the stage-A population (PRMBench steps of 5 folds, 13 channels) the 2015 estimator puts prevalence closer to the truth (estimate 0.231-0.232 against a true 0.137-0.141; Dawid-Skene about 0.28) but its per-channel sensitivity is worse (mean absolute error 0.179 against 0.118 for Dawid-Skene); the stage-A bar fails on 0 of 5 folds for every estimator, and it keeps exactly the same channels as Dawid-Skene on 5 of 5 folds, so the frozen candidate does not change. Source: `results/tensor_mom_v1/run_20260930/SUMMARY.json`, `CHANNELS.csv`, `results/tensor_mom_v1/NEGATIVE_RESULT.md`.

## Evidence

- `results/tensor_mom_v1/run_20260930/SUMMARY.json` @ `claude/estimator-provenance-collection-2026-09-29`, block `compare_mean_over_folds`: prevalence error 0.092 (method of moments) / 0.138 (Dawid-Skene) / 0.144 (hierarchical EM); sensitivity mean absolute error 0.179 / 0.118 / 0.110; Spearman of balanced accuracy against truth 0.752 / 0.752 / 0.779. N = 5 of 5 folds, 13 of 13 channels; truth replays stage A to 1e-12 on all 5 folds; `tests/test_er_stage_a.py` 10/10.
- Same file, `per_fold[k].kept_MoM == kept_DS` on 5 of 5 folds: the filter decision is unchanged, as Step 463 predicted mathematically (the keep rule depends only on the sign of the second-moment loading).
- The provenance note, section 4.2, records the label-using ceiling for group weights at 0.7667 against 0.7656 for the frozen candidate (8-bank mean, `results/algorithm_decisions_v1/SUMMARY.md` on the ssl branch): even perfect weight estimates had little headroom on that partition.

## What failed (implementation) vs what is still open (direction)

Failed, this implementation only (the negative result's own verdict box ticks "closes this IMPLEMENTATION only"):
- The 2015 third-moment estimator applied to binary marks that are a uniform top 20% inside every answer, with channels that are grouped (not conditionally independent given the label). The bias is nearly constant across folds (0.231-0.232), which points to a systematic model mismatch, not noise.

Still open:
- Label-free estimation of channel quality when the channels are not conditionally independent.
- Two insertion points documented in the provenance note were never run, because they were gated on a stage-A improvement that did not happen: the method-of-moments estimates as between-group weights (pipeline stage 10) and as the initialization of the EM (including the position-prior EM of Step 461).
- The paper's second method (restricted likelihood, section 4.2) is not implemented.
- An eigengap rule for the number of groups was not found to have been tested.
- Omri's 2026-10-01 decision (the plain average is not acceptable as the final fusion; replace it with a combination that uses the label-free SML or method-of-moments estimates) makes this estimator family part of the active direction again. Whether the unrun insertion points should now be run is Omri's call, not decided here.
- Deprioritized after discussion (not closed): a self-consistent mark threshold. The 20% value is inherited from the Step 443 tail recipe.

## Reopening conditions

From `results/tensor_mom_v1/NEGATIVE_RESULT.md` ("What would reopen it"), evaluated against the same stage-A truth on the same folds:
1. Votes whose dependence is removed before estimation: position-residualized channels, or one vote per declared block.
2. A marking rule that is not a uniform per-answer quantile (the uniform rule forces a fixed share of false marks into clean answers).

## Dependencies other lines have on it

- The provenance note (`docs/research_notes/ESTIMATOR_PROVENANCE_2026-09-29_HE.md`) is the advisor-ready source table for what "DS", "SML", "L-SML" and "HEM" mean in this project; Omri's 2026-10-01 fusion decision starts from the Step 457 runner-up described there (section 4.2).
- `docs/HANDOFF_COLLECTION_2026-09-29.md` is the lineage map the consolidation used (two lineages, merge order, conflict list).
- `scripts/experiments/er_stage_a.py` on this branch carries `tensor_mom_estimate`; the ssl branch's copy does not have it.
- Step-number collision: `Step 463 [Claude, self-generated step labels]` on `claude/self-generated-step-labels-v1` is a different step. Keep both, tagged.

## Outside git but needed

- Nothing of this line's own output is outside git: `run_20260930/SUMMARY.json`, `CHANNELS.csv` and the log are committed; the worktree's ignored files are only Python caches.
- Inputs read by absolute worktree paths in `scripts/experiments/tensor_mom_stage_a_run.py` (lines 23-37), none of them produced on this line:
  - `.worktrees/depth-feature-fusion-v1` on the Python path (for `spectral_utils.lsml_gate_locator_research.answer_standardize`; the same module is byte-identical on `origin/claude/lsml-ct7-levers-v1`).
  - `.worktrees/token-probability-fusion-v1/results/token_probability_fusion_v1/DERIVATIVE_CHANNELS.npz` (19 MB, git-ignored).
  - `.worktrees/cumulative-vote-fusion-v2/results/cumulative_vote_fusion_v2/ct7_profiles_v1/profiles.npy` (8.2 MB) and `PROFILE_VALIDATION.json`, plus the `cvf_v2` package imported by `er_stage_a.py` from `.worktrees/cumulative-vote-fusion-v2/scripts/experiments`.
  - `.worktrees/readout-quickest-detection-v1/results/step_evidence_v1/OOF_ANSWERS.csv` and `OOF_STEP_SCORES.npz` (54 MB, git-ignored).
  - Stage-A references `results/expectation_realization_v1/run_20260927/INPUT_MANIFEST.json` and `STAGE_A_CHANNELS.csv` (committed on the ssl branch).
