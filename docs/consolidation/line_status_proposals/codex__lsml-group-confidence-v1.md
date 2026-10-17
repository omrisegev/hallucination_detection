# Line status proposal: Codex group-confidence experiment (lsml_group_confidence_v1; no branch of its own)

> **PROPOSAL by Claude, not an Omri decision.** Omri's decision (2026-10-01): write proposed statuses explicitly as proposals, with supporting evidence and reopening conditions; this is not blanket approval to close these directions; distinguish 'this implementation failed' from 'the research direction is closed'. The only explicit closure so far is Joint L-SML (discontinued 2026-10-01).
>
> How to read the status words: they describe the tested implementations and the branch, not the research direction. SATURATED = the implementations tested on current inputs gave no measured gain. IMPLEMENTATION-NEGATIVE = this implementation lost to its matched reference. SUPERSEDED = the branch's content is carried by a newer line. TOOL-ONLY = kept as code or data that other lines use. OUT OF SCOPE FOR NOW and DEFERRED are Omri's words (2026-10-02) and mean "not closed, not a next step". None of these words closes a research direction.
>
> Drafted 2026-10-01 by the consolidation session ae2dd164 (no owner session exists for this line); Omri's 2026-10-02 decisions applied on 2026-10-02.

Proposed status: IMPLEMENTATION-NEGATIVE (direction open)

Owner / last updated: no owner session. This work never had its own branch: it was Codex's uncommitted work in the main checkout (the family-tail handoff of 2026-09-24, section 1, warned not to commit or overwrite it) and was committed for the first time in the rescue commit 4f18f5d3a on `rescue/main-checkout-loose-files-2026-10-01` (on origin). File name chosen by the consolidation session because there is no ref to name it after.

## Steps on this line

- 2026-09-24 [Codex] Binary latent-group confidence: full source negative result (untagged HISTORY block, no step number; present in the rescue branch's HISTORY.md) - A fixed-family binary latent-tree model fitted by EM with spectral initialization separates two reliabilities, each feature's sensitivity/specificity towards its family's latent variable and each family's towards the latent error label, and accumulates within-family evidence. It scored PRMScore 0.631878 against 0.643323 for the plain average of the same 15 families, -1.144 points, corrected interval [-1.501, -0.767]. Source: `docs/experiments/LSML_GROUP_CONFIDENCE_RESULTS_20260924_HE.md`, `results/lsml_group_confidence_v1/METRICS.csv`, `CONTRASTS.csv`, `NEGATIVE_RESULT.md`.

## Evidence

All @ `rescue/main-checkout-loose-files-2026-10-01`, `results/lsml_group_confidence_v1/`:
- `METRICS.csv`: candidate PRMScore 0.631878, within-answer AUC 0.757599, ProcessBench first-error exact localization 34.80%; family-15 average 0.643323 / 0.765854 / 36.51%; matched "average the evidence inside a family, same parameters" control 0.632473; bank11 L-SML 0.641723; CT7 0.646176. N: PRMScore on 6,211 non-control PRMBench answers / 83,371 steps; within-answer AUC on 6,030 answers; ProcessBench on 4,442 erroneous answers over 8 cells; fitting on all 13,769 answers / 145,597 steps (`STATUS.json`: COMPLETE, no failures).
- `CONTRASTS.csv`: 20,000 paired source-question draws, 707 PRMBench source groups, Bonferroni over 4 pre-declared contrasts. Candidate minus family average -1.144 [-1.501, -0.767]; minus 28-feature average -0.900 [-1.199, -0.600]; minus binary-family L-SML -0.996 [-1.307, -0.675]; minus its own evidence-average ablation -0.059 [-0.264, +0.151].
- `FEATURE_ACCURACY_DIAGNOSTIC.csv` (post hoc): the latent sensitivities/specificities are off from the true-error values by 14.91 / 8.57 points on average over 140 estimates (28 features x 5 folds), although their rank correlation with the measured balanced accuracy is 0.799. Example `q15_H1`: latent sensitivity 0.743 against a true 0.430.
- Three independent audits (`AUDIT_RECOMPUTE.json`, `AUDIT_POPULATION.json`, `AUDIT_MATH_NULL.json`, summary `RED_TEAM.md`) pass; shuffled-label null mean AUC 0.500292 over 20 seeds on 6,030 answers.

## What failed (implementation) vs what is still open (direction)

Failed, this implementation only (the negative result's verdict box ticks "closes this IMPLEMENTATION only"): the fixed 15-family partition, top-20% (q80) binary marks, EM fit of the two-level binary model, and a continuous-likelihood extension at application time. It differs from the family average (mean within-answer rank correlation 0.91465; 8,309 of 145,597 step decisions change) but is not better, and its accumulation adds nothing over averaging the same evidence.

Still open, as the experiment's own text says: confidence-aware group learning in other settings; a reliability estimator whose latent reliabilities are shown to track true errors. The simulation (30,000 test examples) confirms the mechanism works when its assumptions hold (log loss 0.154357 to 0.136218 with three complementary measurements), so the failure is attributed to the data's dependence structure and reliability estimation, not to the algebra.

Relation to Omri's 2026-10-01 decision: the decision replaces the plain average with a combination that uses label-free SML or method-of-moments estimates, starting from the Step 457 runner-up (hierarchical latent-group EM weights within groups, Dawid-Skene weights between groups). This experiment is the closest earlier test of a two-level latent reliability model on a fixed family bank, and its post hoc finding (latent reliabilities inflated relative to true-error reliabilities) matches the ssl line's Steps 450 and 464 findings. It is prior evidence for that new work, not a decision about it.

## Reopening conditions

From `results/lsml_group_confidence_v1/NEGATIVE_RESULT.md`: a separately declared source experiment with a defensible family-error structure and a more faithful reliability estimate that beats both matched equal controls and its own evidence-average ablation on PRMScore before any external testing. The Hebrew report adds: do not multiply a group weight by group size, and do not sweep temperatures or banks on the same development data and then call it independent confirmation.

## Dependencies other lines have on it

- None in code: no later line imports `spectral_utils/lsml_group_confidence*.py`.
- The family-tail handoff (`docs/experiments/FAMILY15_TAIL20_AGENT_HANDOFF_HE.md`) explicitly distinguishes this experiment (PRMScore 0.631878) from the packaged family15 tail20 candidate.
- Suggested input for the 2026-10-01 fusion direction (see above).

## Outside git but needed

- `results/lsml_group_confidence_v1/PREDICTIONS.npz` (11.3 MB), `CALIBRATION.npz` (11.1 MB) and `EVALUATION.npz` (0.3 MB) in the main checkout, git-ignored. `SEAL.json` records their hashes; the evaluate step checks the seals. The 2026-10-01 consolidation review lists these three files (22,786,417 bytes) among the omitted ignored files: they were not in the 2026-10-01 upload and no verified Drive copy was found in this check. Regenerable with `python scripts/run_lsml_group_confidence.py fit` then `evaluate` (about 78 s for fitting and scoring), but the frozen files and their hashes are the record.
