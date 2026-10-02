# Line status proposal: codex/lsml-external-generalization-v1

> **PROPOSAL by Claude, not an Omri decision.** Omri's decision (2026-10-01): write proposed statuses explicitly as proposals, with supporting evidence and reopening conditions; this is not blanket approval to close these directions; distinguish 'this implementation failed' from 'the research direction is closed'. The only explicit closure so far is Joint L-SML (discontinued 2026-10-01).
>
> How to read the status words: they describe the tested implementations and the branch, not the research direction. SATURATED = the implementations tested on current inputs gave no measured gain. IMPLEMENTATION-NEGATIVE = this implementation lost to its matched reference. SUPERSEDED = the branch's content is carried by a newer line. TOOL-ONLY = kept as code or data that other lines use. OUT OF SCOPE FOR NOW and DEFERRED are Omri's words (2026-10-02) and mean "not closed, not a next step". None of these words closes a research direction.
>
> Drafted 2026-10-01 by the consolidation session ae2dd164 (no owner session exists for this line); Omri's 2026-10-02 decisions applied on 2026-10-02.

Proposed status:
- The branch as a development line: SUPERSEDED (by `claude/ssl-pseudolabel-residual-v1`, which contains every commit of this branch and carried the external check forward in Step 459)
- External telemetry collection, frozen evaluation pipeline and the frozen bank11 L-SML bundle: TOOL-ONLY (the external-transfer reference and harness reused by later lines)
- Family/tail L-SML recipes evaluated externally (Codex family15 tail20 V1, Steps 444 and 446): SATURATED (same evidence as the ssl status file's family/tail entry)
- Answer-local token-level L-SML (registered arm of Step 433): IMPLEMENTATION-NEGATIVE (direction open)
- Follow-ups listed in `docs/HANDOFF_FAMILY_TAIL_LSML_2026-09-24.md` section 3, with Omri's decisions of 2026-10-02 (`docs/consolidation/DECISIONS_2026-10-01.md` items 16-18):
  - Source-side transfer proxy (item 1): APPROVED as one bounded exploratory test. Its scope, protocol and success criterion are written down before it runs. It is not a method sweep and does not block the SML/MoM direction. Hard2Verify and Socratic are exposed, so matching their ranking is exploratory evidence, not proof of transfer.
  - Why bank11 L-SML transfers (item 2): DEFERRED, not closed.
  - Independent audits of V2 and V3 (item 3): APPROVED as verification of existing work.
  - Published comparators: open and unfinished; no decision recorded.

Owner / last updated: no owner session. Tip bef0b1de3 (2026-09-24). This is the branch the main checkout has checked out; the local tip is ahead of `origin/codex/lsml-external-generalization-v1` but is contained in `origin/claude/ssl-pseudolabel-residual-v1`.

## Steps on this branch

- 2026-09-24 [Codex, untagged blocks] External collection: AIRCC teacher-forced telemetry for Hard2Verify / Qwen3-8B (200/200), Socratic-PRMBench / Qwen3-8B (2,995/2,995) and Socratic-PRMBench / QwQ-32B (2,995/2,995); 6,190 records, 53,970 steps, 0.382 allocated GPU-hours, private Drive archives verified. Source: HISTORY "Full external collection verified and archived"; `results/lsml_external_generalization_v1/FULL_ARCHIVES.json`.
- 2026-09-24 [Codex] Existing teacher-forcing repeatability check: identical reruns gave no observed diversity. Source: `docs/experiments/EXTERNAL_TEACHER_FORCING_REPEATABILITY_20260924.md`.
- Step 433 [Codex, untagged heading "Evaluate frozen and answer-local L-SML on external benchmarks"] Frozen step-level bank11 L-SML: Hard2Verify balanced F1 43.670, Socratic PRMScore 63.221 / 64.238, against CT7 37.751 / 58.753 / 60.166; answer-local token L-SML loses to its token-equal control on both Socratic backbones. Source: `docs/experiments/LSML_EXTERNAL_GENERALIZATION_RESULTS_20260924.md`; `results/lsml_external_generalization_v1/evaluation/{METRICS,CONTRASTS}.json`. (Collides with `Step 433 [Claude][ct7-levers]`; keep both.)
- 2026-09-24 [Codex] Family15 tail20 external V1: 41.020 / 59.921 / 62.690 against bank11 L-SML 43.670 / 63.221 / 64.238. Source: `results/family_tail_external_v1/METRICS.json`, `docs/experiments/FAMILY_TAIL_EXTERNAL_RESULTS_20260924.md`.
- Step 444 [Claude, family-tail external V2] The defect-corrected family-tail row reaches only the family-equal level externally (42.28 / 61.00 / 62.30), 1.4-2.2 points below bank11 L-SML. Source: `results/family_tail_external_v2/REPORT_HE.md`.
- Step 446 [Claude, family-tail external V3] Bank11 maximal-step tail gives the best Hard2Verify score (44.48, +0.81 over bank11 L-SML, not significant) but trails bank11 L-SML on both Socratic cells (-0.84 / -0.59, significant). Source: `results/family_tail_external_v3/REPORT_HE.md`.

## Evidence

- Frozen bank11 L-SML minus its matched step-equal control: Socratic / Qwen3-8B +2.430 PRMScore points, Bonferroni-corrected interval [+1.848, +3.027]; Socratic / QwQ-32B +2.735 [+2.115, +3.356]; Hard2Verify +2.787 [-0.222, +5.839] (inconclusive). 100,000 paired source-question draws, 18 contrasts. N = 6,190 records / 53,970 steps (200 Hard2Verify answers, 2,995 Socratic answers per backbone). File: `docs/experiments/LSML_EXTERNAL_GENERALIZATION_RESULTS_20260924.md` @ `codex/lsml-external-generalization-v1`.
- Answer-local token L-SML minus token equal: -1.285 [-1.894, -0.665] (Socratic / Qwen3-8B) and -1.087 [-1.854, -0.313] (Socratic / QwQ-32B); native fits 2,984 of 2,995 per backbone. Same file.
- Source rank did not predict external rank in three locks (V1, V2, V3): on source CT7 (64.62) and the family variants led; externally bank11 L-SML leads and CT7 is last. File: `docs/HANDOFF_FAMILY_TAIL_LSML_2026-09-24.md` section 2 item 4 @ `codex/lsml-external-generalization-v1`.
- Hard2Verify decision diagnostic: 41 of 42 entirely correct answers receive at least one error flag; with the observed per-answer flag counts even a label-using reassignment cannot exceed 52.673 balanced F1. File: `results/lsml_external_generalization_v1/evaluation/CALIBRATION_DIAGNOSTICS.json`.

## What failed (implementation) vs what is still open (direction)

Failed or exhausted:
- The family/tail L-SML recipes (15 families, tail marks at top 20% or top 1, common or per-family thresholds): below bank11 L-SML on Socratic in every lock. The family-tail handoff (section 3 item 6) lists them as "do not reopen without a new idea". That is a statement about these recipes, not about family or tail fusion as a whole.
- The answer-local token L-SML implementation (stride 8, at least 3 active channels): below its own equal control on both Socratic backbones. This is one implementation of runtime learning, not all CPU runtime learning (the results document says so explicitly).

Still open:
- A source-side split that predicts transfer (leave one benchmark out, leave one ProcessBench cell out, PRMBench category holdout), with any retrospective check against the known external ranking labelled as exposed. Approved by Omri as one bounded exploratory test (2026-10-02); protocol first.
- Why bank11 L-SML transfers (partition and weight stability across folds). DEFERRED, not closed (Omri, 2026-10-02). The ssl line's Step 438 note attributes its source edge to de-noising three anti-oriented channels.
- Independent audits of V2 and V3 (only V1 has the three audit scripts run). Approved by Omri as verification (2026-10-02).
- Answer-level information and decision calibration (the 41-of-42 finding).
- Cross-backbone fusion: Qwen3 and QwQ decisions disagree on 1,751 of 26,055 Socratic steps (6.72%); no cross-backbone fusion was fitted.
- Published comparators (Hard2Verify's Qwen3-8B critic 53.51, Qwen2.5-Math-PRM-7B 42.37; Socratic's Qwen2.5-Math-PRM-7B 68.0, QwQ-32B critic 73.8) are context only; reproduction is unfinished. No SOTA claim is supported.
- Untouched confirmation: MedPRMBench is deferred. Hard2Verify and Socratic are now exposed and can only give exploratory results.

How this sits with Omri's decisions: the frozen bank11 bundle is ordinary continuous L-SML, not Joint L-SML, so the 2026-10-01 Joint discontinuation does not by itself retire it. The 2026-10-01 decision to replace the plain average with an SML/method-of-moments combination applies to the newer label-free line; it does not decide what external reference a new candidate is compared against. Both questions are for Omri.

## Reopening conditions

- Family/tail recipes: only with a new idea that is not tail-threshold tuning, per-family thresholds, 16 families or within-family SML, token-level tail fusion, or FUSE-style consistency objectives (handoff section 3 item 6).
- Answer-local token L-SML: a source experiment in which a revised answer-local estimator beats its matched token-equal control before any external test.
- Any new external claim: a candidate frozen on source and tested on data not yet exposed.

## Dependencies other lines have on it

- `claude/ssl-pseudolabel-residual-v1` contains the whole branch. Its Step 459 (`results/algorithm_external_v1/`) reuses the external inputs and the sealing pipeline; its `scripts/fit_external_source_bundle.py` and `scripts/verify_external_banks_v4_source.py` descend from this branch.
- `claude/self-generated-step-labels-v1` Step 463 scores "the frozen bank11 L-SML sent to the external benchmarks" on self-generated answers (`results/self_generated_step_labels_v1/analysis/FROZEN_METHODS_V1.json`).
- The frozen modules `spectral_utils/external_generalization/` and `spectral_utils/external_generalization/_bank11/` are hash-locked; two `_bank11` files are committed with CRLF and must not be renormalized (collection handoff section 6).
- `lsml-ct7-levers-run` Step 447 compares against bank11 and family15 rows built from the same source pool.

## Outside git but needed

- `scratch/external_generalization_private/` in the main checkout (about 9.3 GB): external inputs, pinned official evaluator sources, raw telemetry archives. Restore locations and SHA256s: `results/lsml_external_generalization_v1/FULL_ARCHIVES.json`, `evaluation/SOURCE_DEPENDENCY_ARCHIVE.json`, `evaluation/EVALUATION_ARCHIVE.json`; Drive folder `gdrive:hallucination_detection/cluster_results/lsml_external_generalization_v1/`. The 2026-10-01 reconciliation lists this folder as "not checked" against Drive.
- Main checkout git-ignored outputs: `results/lsml_external_generalization_v1/` (8 files, about 23 MB), `results/family_tail_external_v1/` (about 6,200 files, 224 MB), `v2` (65 MB) and `v3` (82 MB). V1's large arrays and source pool are in a 190,131,044-byte archive with 6,312 verified members (`results/family_tail_external_v1/ARCHIVE.json`, restore steps in `RESTORE.md`); V2/V3 can be regenerated from V1 records in minutes (`scripts/run_family_external_v{2,3}.py`, then the evaluators with `--seal-only`). The 2026-10-01 upload streamed `lsml_external_generalization_v1.tar` to Drive with a matching MD5 but did not check its member count.
- Source pool `pool_z.npy` (sha256 d9abccfb...) and `pool_names.json` in session 2d14a8c9's temp scratchpad; hard-coded by the source scripts; archived as member `reproduction/source_pool/` of the family-tail V1 archive. A hash-verified copy also sits at `results/algorithm_decisions_v1/inputs_backup/pool_z.npy` in the ssl worktree.
- Population files `.worktrees/readout-quickest-detection-v1/results/step_evidence_v1/` (OOF_ANSWERS.csv, OOF_STEP_SCORES.npz, INPUT_FREEZE.json).
