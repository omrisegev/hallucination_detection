# Monday fusion experiments

Base: `54a2c2a8f463c1c48e8f943405cb1bb7f657c85b`.
Branch: `codex/monday-fusion-analysis-20261003`.
Frozen protocol commit: `bf7609cf57af34abb05e36713257b920ba34e41a`.

This is an isolated code export, not a checkout. The primary checkout and its
index are unchanged. `scripts/monday_fusion_workspace.py` commits only explicitly
mapped files using its own index and compare-and-swap branch updates.

Local entry point:

```powershell
python -B C:\Users\omris\TAU\hallucination_detection\results\monday_fusion_analysis_v1\code\scripts\experiments\monday_fusion_run.py audit
```

Stages are ordered: `audit`, `replay`, `zero`, `feasibility`, `level`,
`estimator-a`, `estimator-b`, then position/decision/LR, full combined recipe and
competitor analyses. Later stages must refuse incomplete prerequisites.

The full population is 13,769 model-answer records and 145,597 steps in 9 cells.
PRMBench has 6,969 answers, 6,030 eligible for within-answer AUC and 6,211
non-control answers. ProcessBench has 6,800 model-answer records, including 4,442
erroneous records. Source grouping contains 3,483 groups; q4/q8 records are paired,
not independent questions.

Input checks compare available historical hashes, original PRMB error annotations,
all v3 release rows and all v2 fold assignments. The pool backup uses a new
manifest; frozen manifests and scores are never edited.

Historical replay: 8 Step 457 BASE/GRP comparisons on the original fold-0 contract;
72 Step 462 BASE/GRP comparisons on 4 banks x 9 cells. The tolerance is `1e-9`.
The measured maximum errors were `8.881784197001252e-16` and
`2.220446049250313e-16`, respectively. Zero-removal checks all 72 bank/cell input
triples exactly and 72 content-score comparisons through the new runner.
Evidence: `results/monday_fusion_analysis_v1/{INPUT_AUDIT,REPLAY_GATE}.json`.

A centering arithmetic-layout difference initially stopped zero-removal at
pb_gsm8k_q4/B32: 40 centered-mark entries, maximum `1.1102230246251565e-16`;
features and binary marks were exact. The implementation now retains unchanged
columns and recomputes changed-sign column slices exactly as the historical
runner. The exact gate then passed; no tolerance was relaxed.

Feasibility used B54/remove_3 on the complete pb_gsm8k_q4 cell (400 answers,
2,082 steps). All ten content recipes produced finite scores in 30.16 seconds;
no quality metrics were computed. Multiplying by 360 gives about 3 hours, a rough
fit-only extrapolation that excludes larger-cell effects, metrics, LR and layers.
Evidence: `results/level_reduction_v1/run_20261003/FEASIBILITY.json`.

Each completed bank/cell/variant chunk has an artifact hash seal. Failed arms
remain explicit rows, never successful-cell averages. Only original-bank main
content comparisons retain step arrays; other scores reconstruct from saved
groups, within weights and between weights. Reports must retain failures and
coverage when selecting an 8-bank average.

Continue the approved sequence after the active LEVEL process finishes:

```powershell
python -B C:\Users\omris\TAU\hallucination_detection\results\monday_fusion_analysis_v1\code\scripts\experiments\monday_fusion_run.py continue --wait-for-level
```

The continuation waits before loading the full banks, then runs 2a, 2b,
primary contrasts, position replay and layers, nested LR, secondary contrasts,
main score packages, figures/reports and the combined full-population check.
It stops at `AWAITING_OMRI_DECISION_RULE`. Resume with the same command after
fixing an explicit failure; completed artifacts are hash-checked. Never start
two copies of a scoring stage at once. Live status is in WORKFLOW_STATUS.json
and each experiment's RUN_STATUS.json. A stopped stage is not a completed result.

The experiment directories contain frozen IMPLEMENTATION_NOTES_V1.json. These
clarify the unfiltered ALL_EQ position control, descriptive comparison scope,
raw digit channels in R2, content-primary combined selection and score retention.
The original PROTOCOL.json files stay byte-identical. The combined experiment
has its own committed protocol before its first fit. Code manifests identify
the exact commit used for each stage, including the earlier active LEVEL code.

The continuation does not contact Claude or make an external evaluation.
REVIEW_PACKET.json contains concrete files and hashes for independent review.
Only after Omri provides an explicit choice may `freeze-rule --rule ...
--authorization ...` lock the recipe. The separate `competitors` command then
produces F11/F12 from that same final score bundle. They are marked open before
this gate; code availability is not completion of those analyses.

The core score bundle STEP_SCORES.npz retains the provisional selected content
recipe and matched EQ for every bank. STEP_SCORES_MANIFEST.json also indexes
the original-bank main scores and experiment-2 DS/MoM comparisons in chunks.
All other scores reconstruct from fixed inputs and saved fit parameters.
LR OOF versus in-sample scope is explicit. Runtime columns include shared fitting
work and are not isolated method-speed measurements.

All development findings await Claude's independent recomputation, population
coverage review and label-null review. No unreviewed finding is a Monday-deck
conclusion. Omri chooses the final decision rule after comparison. Existing
external benchmark observations remain exploratory.


## Review-corrected continuation, 2026-10-04

Omri freed disk and explicitly requested continuing. RUNS now points to
`results/{level_reduction_v1,estimator_input_v1}/run_20261004_resume/`.
The original run_20261003 files remain unchanged. Sealed checkpoints are
reused by immutable hardlinks; mutable aggregate files are independent copies.
Each new run has committed REVIEW_CORRECTIONS_V2.json and byte-identical
original PROTOCOL.json/IMPLEMENTATION_NOTES_V1.json.

Selection keeps the registered eight-bank matched-EQ mean criterion, excludes
random-removal diagnostic controls, and no longer mixes in a plain baseline
with an artificial zero gain. Direct intervention contrasts are separately
labelled post-review exploratory. Continuation recomputes only core contrasts
and resumes the 26 sealed layer checkpoints; fitting/scoring algorithms are
unchanged. The original source hash inventory is retained in each revision file.

Omri explicitly requests highlighting HEM+MoM channel/cell versus group/answer
localization on original banks, alongside its q80 F1/clean-flag tradeoff.
Both score bundles are retained as main comparisons even when a different
recipe wins the registered criterion. Slide talking points and exact sources:
`docs/meetings/advisor_prep_2026-10-05/FUSION_GRAPH_TAKEAWAYS_20261004_HE.html`.
New interval claims remain awaiting Claude's independent review. Omri's final
decision-rule gate and the remaining original experiment coverage remain intact.
