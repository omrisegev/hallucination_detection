# Line status: claude/ssl-pseudolabel-residual-v1

Status:
- Label-free fusion line (Dawid-Skene filter + plain average, position, per-model-per-dataset fitting; Steps 450-462): PAUSED-FOR-DECISION
- L-SML variants on the step bank (Steps 438-446, 456): SATURATED
- Label-free switch between plain and grouped fusion (Step 458): SATURATED
- First-error product readout (Step 462): SATURATED

Owner / last updated: Claude session on worktree .worktrees/ssl-pseudolabel-residual-v1 (owns Steps 456-462; Steps 438-455 tagged
[Claude] were earlier Claude sessions on this same branch; Steps 447 and 448 tagged [Codex] came in with the merge a9c7d6f83),
2026-09-30.

Steps on this branch:
- Step 438 [Claude, bank20 L-SML] Extending the 11-channel bank to 20 under L-SML is unsupported; the bank11 edge is de-noising of one partition. Source: results/bank20_lsml_prmbench_v1/run_20260927_calfix/
- Step 439 [Claude, independent weak channels] Adding 21 weak channels makes L-SML fall 9 points. Source: results/indbank_lsml_prmbench_v1/run_20260927_calfix/
- Step 440 [Claude, declared partitions + Joint L-SML] Joint L-SML on a declared partition gains +0.0130 over block-equal. Source: results/declared_joint_prmbench_v1/run_20260927_calfix/
- Step 441 [Claude, family/tail development chain] Label-using development superseded by Step 442. Source: HISTORY.md Step 441 and docs/HANDOFF_FAMILY_TAIL_LSML_2026-09-24.md (the review HISTORY cites, docs/reviews/CLAUDE_TOKEN_TAIL_LSML_CORRECTION_HANDOFF_20260924_HE.md, is not on this branch)
- Step 442 [Claude, calfix] Calibration fix moves numbers at most 0.14 points; learned weights beat equal only on bank11. Source: docs/HANDOFF_FAMILY_TAIL_LSML_2026-09-24.md
- Step 443 [Claude, tail threshold calibration] The step-level K=2 was an input-scale artefact; no tail threshold beats matched equal. Source: results/tail_threshold_calibration_v1/REPORT_HE.md
- Step 444 [Claude, family-tail external V2] The corrected family-tail row reaches only the family-equal level externally, 1.4-2.2 points below bank11 L-SML. Source: results/family_tail_external_v2/REPORT_HE.md
- Step 445 [Claude, maximal-step tail on source] Maximal-step tail rows are below their alternatives on source. Source: results/tail1_transfer_v3/TRANSFER_LOCK_V3.json
- Step 446 [Claude, family-tail external V3] Bank11 maximal-step gives the best Hard2Verify score (44.48, not significant) but trails bank11 L-SML on Socratic. Source: results/family_tail_external_v3/REPORT_HE.md
- Step 447 [Claude, partition ceiling] Over all 237,018 block-equal partitions the declared one is in the bottom quartile and the whole space barely reaches CT7. Source: results/partition_ceiling_prmbench_v1/run_20260927_calfix/
- Step 447 [Codex, digit alternative probability] A numeric alternative carries signal but is not a consistently better standalone replacement. Source: results/digit_alternative_probability_v1/REPORT_HE.md
- Step 448 [Claude, Mind the Gap code audit] The released code has no step-level code and is not the published method. Source: docs/reviews/MIND_THE_GAP_CODE_AUDIT_20260927.md
- Step 448 [Codex, numeric-family extension] Three digit columns form a separate group without a gain over one column. Source: results/digit_family_extension_v1/REPORT_HE.md
- Step 449 [Claude, Mind the Gap reproduction] The published method is reproduced in pattern and within 3.3 SLA points; the released code is 18.4 points away. Source: results/mtg_reproduction_v1/SUMMARY.json
- Step 450 [Claude, expectation vs realization] The written-token block lifts bank11 by +0.0119 under L-SML; L-SML equals averaging on 13 channels. Source: results/expectation_realization_v1/run_20260927/RED_TEAM.md
- Step 451 [Claude, stage B] Label-free classifier properties select channels (the filter removes exactly the anti-oriented ones) but cannot weight them. Source: results/expectation_realization_v1/run_20260927_stage_b/RED_TEAM.md
- Step 452 [Claude, stage B2] Counting the level family once repairs the weights but they add nothing over plain averaging. Source: results/expectation_realization_v1/run_20260927_stage_b2/RED_TEAM.md
- Step 453 [Claude, fold-role fix of six runners] No cited conclusion changes (at most 0.0021 within-AUC). Source: docs/HANDOFF_EXPECTATION_REALIZATION_2026-09-27.md
- Step 454 [Claude, PRMScore decomposition] The filter's PRMScore gain is mostly position structure; the gap to Qwen PRM is flag-count allocation. Source: results/expectation_realization_v1/prmscore_decomposition_v1/RED_TEAM.md
- Step 455 [Claude, er_generality] The filter's channel selection transfers to 20/32/51-channel banks; its score gain, grouping and L-SML do not. Source: results/er_generality_v1/SUMMARY.md
- Step 456 [Claude, lsml_merge_step] One merge step repairs L-SML's partition, but L-SML still loses to plain averaging. Source: results/lsml_merge_step_v1/SUMMARY.md
- Step 457 [Claude, algorithm_decisions] The frozen rule selects the filter + plain average (8-bank mean 0.7656); L-SML loses on 8/8 banks; digits help everywhere. Source: results/algorithm_decisions_v1/SUMMARY.md
- Step 458 [Claude, partition_switch] The switch meets its criterion only through twin bank families and never switches on with the family held out. Source: results/partition_switch_v1/SUMMARY.md
- Step 459 [Claude, algorithm_external] Externally the candidate beats ct7 on Socratic mostly through position, and the step index alone beats every method. Source: results/algorithm_external_v1/SUMMARY.md
- Step 460 [Claude, position_channel] The step index as one more channel adds +0.004 to +0.009 within-AUC on 4 banks; its weight is set by bank size. Source: results/position_channel_v1/SUMMARY.md
- Step 461 [Claude, position_prior] A position prior inside the Dawid-Skene model is adopted for the grouped method (0.8056) but its weight is 2-5x below the optimum. Source: results/position_prior_v1/SUMMARY.md
- Step 462 [Claude, per_dataset_fit] Fitting per model per dataset is free on PRMBench and drops position on ProcessBench by itself; the per-dataset prior latches onto a step-0 artefact and the first-error readout fails. Source: results/per_dataset_fit_v1/SUMMARY.md

If PAUSED-FOR-DECISION (label-free fusion line):
- Resolved 2026-09-30: the digit features stay (Omri); recorded in CLAUDE.md.
- Open question for Omri (to be decided after the merge, among all options on the table): which next step? Options, in Claude's recommended order (Research_Directions.md, 2026-09-30 section):
  (1) use the filter's per-dataset decision on the position channel as the gate for the stronger position prior; (2) set the
  position weight with the cross-fitted slope while keeping the plain average unchanged; (3) remove the per-dataset position
  profile of the content score before any position term; (4) the external check (Hard2Verify, Socratic) under the per-dataset
  contract; (5) untouched confirmation on MedPRMBench (needs AIRCC).
- The advisor report is paused by Omri until findings from other conversations are collected.

If SATURATED:
- L-SML variants: tried bank extensions (438-439), declared/Joint partitions (440), family and tail marks with calibration
  (441-446), exhaustive partitions (447), level-family reduction (452), a merge step (456) and 8 banks (457). Evidence: L-SML
  below plain averaging on 8/8 banks (results/algorithm_decisions_v1/SUMMARY.md, 8 banks x 6,030 PRMBench answers); merge step no
  gain (results/lsml_merge_step_v1/run_20260928/RED_TEAM.md). Reopen only with channels that are conditionally independent given
  the label (e.g. a second model or internal layer states), since the current ones share one output distribution.
- Switch rule: evidence results/partition_switch_v1/run_20260928/RED_TEAM.md (0/80 switches with the bank family held out).
  Reopen only with banks that do not share channels.
- First-error product readout: evidence results/per_dataset_fit_v1/run_20260929/RED_TEAM.md (below the argmax in 64/64 cells,
  also without a prior; shrinking the slope makes it worse 8/8). Reopen only with a per-step error rate estimated at the true
  scale (about 0.14 on PRMBench), which the label-free latent class does not give (0.28-0.41).

Outside git but needed (about 2.3 GB, all git-ignored or untracked; listed in docs/HANDOFF_LABEL_FREE_ALGORITHM_2026-09-30.md):
- results/*/run_*/STEP_SCORES.npz of Steps 450-462 (largest: algorithm_decisions_v1 351 MB, expectation_realization_v1 stage_b2
  154 MB x2, lsml_merge_step_v1 103 MB, per_dataset_fit_v1 75 MB): regenerable by the runners (minutes each); Steps 461-462 read the
  457/460/461 files as replay references.
- results/*/run_*/BOOTSTRAP_DELTAS.npz (lsml_merge_step 142 MB, expectation_realization, er_generality): regenerable.
- results/algorithm_external_v1/run_20260929/*/PREDICTIONS_UNSEALED.json (1 + 16 + 16 MB): their digest is in SEAL.json;
  regenerable only with the private external inputs (main checkout scratch/external_generalization_private/inputs).
- results/external_banks_v4/*/FEATURES.npz and reference_arrays/*.npy (hashes in reference_arrays/HASHES.json): regenerable.
- results/algorithm_decisions_v1/inputs_backup/pool_z.npy (57 MB): hash-verified backup of the pool input that INPUT_MANIFEST.json
  points to in a temporary session scratchpad; regenerable by results/indbank_lsml_prmbench_v1/pool_structure.py.
- None is archived on Drive yet; suggested target gdrive:hallucination_detection/claude_label_free_line_2026-09/.
