Codex, request from Omri (2026-10-04): replace the automatic combined-recipe selection with a side-by-side candidate panel. Omri chooses the recipe and the decision rule himself; the script must not select a candidate.

WHY THE COMBINED STAGE ADDS NO INFORMATION NOW
After the review correction, `SELECTION.json` of level_reduction_v1 picks `original/MEAN_MOM`: no LEVEL removal won. `monday_fusion_combined.proposal()` then picks the input for that recipe: `channels_cell` (mean advantage 0.007438 vs 0.007246 for groups_answer). So LEVEL_ONLY equals BASE and COMBINED equals INPUT_ONLY, and both remaining conditions are exactly the experiment-2 content chunks (MEAN_MOM on groups_answer and on channels_cell), whose decision rules the estimator-input layers will also compute. The stage would recompute 288 stems (~3 h) for identical numbers, and `run_combined` then auto-selects `selected_condition` by mean advantage, which is the automatic choice Omri does not want. It also never shows HEM_MOM, which Omri wants beside MEAN_MOM.

WHAT TO DO WITH THE RUNNER
Do not stop it now. `combined` is the last workflow stage. When the workflow reaches `combined` (after input_report), stop it there, with Omri informed; nothing before it is lost. Do not run `freeze-rule` or `competitors` until Omri has chosen.

WHAT TO BUILD INSTEAD (read-only aggregation; a good job for a subagent)
A candidate panel built only from sealed outputs: experiment-2 content chunks, estimator-input layer checkpoints (when complete), LR checkpoints (original variant), and the comparator inputs. No refits.

Candidates, all on original banks, CONTENT position as the primary row:
- MEAN_MOM, groups_answer (the Step 457 input)
- MEAN_MOM, channels_cell (current registered winner)
- HEM_MOM, groups_answer
- HEM_MOM, channels_cell (Omri's highlighted finding)
References in every table: DS_FILTER_EQ, ALL_EQ, the matched HEM_EQ / MEAN_EQ, and LR (OOF, per bank, full and survivor rosters, with and without position) as the linear ceiling.

For each candidate x decision rule (ANSWER_Z_Q80, R2_COUNT, ENTROPY_GATE_Q80; TOP_1/2/3 as controls, TOP_2 marked post hoc):
- PRMBench: within-answer AUC, official PRMScore, clean-answer flag rate, answer-level AUROC.
- ProcessBench: argmax first-error accuracy, first-flag decision accuracy, official F1, clean-answer flag rate, answer-level AUROC.
- Per bank and equal 8-bank mean (PB also equal over the 8 cells), with N.
- Paired source-question bootstrap intervals for: each candidate minus DS_FILTER_EQ, HEM minus MEAN at the same input, channels_cell minus groups_answer at the same recipe. Mark these as post-review exploratory.
- Position variants (POS, PRIOR, PRIOR_CF) as a separate diagnostic table, with explicit failures kept in the denominators (example: PRMBench B32 original fails PRIOR/PRIOR_CF for all four MEAN recipes on the 1e-6 one-bin gate).

Comparator forensics for every candidate x the three rules, pre-selection and descriptive, using the same code as `monday_fusion_competitors.run_competitors` but without requiring a lock:
- F11 content: PRMBench error-step overlap with the cached supervised PRM per category (both / only ours / only PRM / neither), clean-answer flags, and category metrics native plus top-k for ours and the PRM.
- F12 content: ProcessBench first-error forensics against our Mind the Gap reproduction by error-position quartile and length bin; miss types (early, late, no flag); clean-answer flag rates.
Label all of it "pre-selection development comparison; not a locked recipe; not untouched confirmation".

CROSS-CHECK AVAILABLE NOW
Claude already computed the CONTENT decision table for these candidates in `results/monday_fusion_review_claude_20261004_round2/decision_rules_precompute/` (DECISION_TABLE.csv, DECISION_METRICS.csv, SUMMARY.json). Its gate passed against your sealed content (864 rows) and your sealed LEVEL `original` layers (2,556 rows, max difference 1.1e-16). Use it as a cross-check of the panel, not as a replacement. Main point it shows: the HEM_MOM PB F1 drop is specific to q80; under R2 and the entropy gate the same input change raises PB F1 by +0.011 [0.008, 0.014] and +0.014 [0.008, 0.020].

AFTER OMRI CHOOSES
Omri names one candidate and one rule. Then lock it (`STEP_SCORES.npz` + `LOCKED_RECIPE.json` with his authorization text) from the experiment-2 chunks, since `freeze_decision` currently expects the combined-stage files and needs a small adaptation; keep its replay and write-once checks. Then run the final F11/F12 on the locked recipe only. Report which candidates and rules were viewed before the choice.

Deliver the panel as one Hebrew HTML report plus CSVs, with every number naming its file and N. Do not send messages or edit the Monday deck automatically.
