# expectation_realization_v1: run log

Stage-by-stage management record. Protocol: `PROTOCOL.json` (frozen before any scoring).

| Date | Stage | What happened | Commit |
|---|---|---|---|
| 2026-09-27 | Design | Algorithm and experiment agreed with Omri in chat: 13 channels in two views (expectation: level + change; realization: written token vs forecast), CONT L-SML, Joint declared/own, tail L-SML, stage A accuracy check of SML sensitivity/specificity estimates, stop after stage A | |
| 2026-09-27 | Pre-freeze checks | CT7 within-AUC replays exactly (0.7723966); fam421 recomputed from CT7 profiles 0.78013 (reported 0.780120); realized_drv alone 0.79574; realized_z alone 0.73874 (NOT the 0.7434 quoted in chat from Step 417) | |
| 2026-09-27 | Protocol frozen | `PROTOCOL.json` | 73cca2308 |
| 2026-09-27 | Code + tests | Runner, stage-A estimators (6 known-result tests pass), digit diagnostic | 409497657 |
| 2026-09-27 | Implementation smoke | One fold, 2,000 draws: end to end; replays exact; Joint own partition did not converge in fold 0; discovery ~600 s per fold. Implementation check only | |
| 2026-09-27 | Independent review | NOT READY: blocking evaluation/calibration role overwrite (same defect exists in the Step 438-440 runners), non-convergence policy, discovery seed; plus digit-count basis and minor items | |
| 2026-09-27 | Amendment A1 | All review items fixed; protocol amended before the full run (see PROTOCOL.json amendments) | this commit |
| 2026-09-27 | Re-review of A1 | Independent reviewer: FIXES VERIFIED (two non-blocking reporting notes: no NOT_ESTIMABLE row in PB_CONTRASTS; empty groups resampled in the bootstrap) | |
| 2026-09-27 | Full run `run_20260927` | 5 folds, 100,000 draws, 43 min; all pre-evaluation hard stops passed (answer order, level bank and derivative reproduced from the token matrix to 0.0, B11 replay 0.0, ct7 exact, fam421 0.78013, write-once); Joint on its own partition did not converge in any fold (NOT_ESTIMABLE) | this commit |
| 2026-09-27 | Digit diagnostic | PRMBench, alignment exact (0.0) | this commit |
| 2026-09-27 | Red team | three independent agents launched (recompute / coverage / null+math); no interpretation before their consensus | |
| 2026-09-27 | Red team complete | A recomputation: all numbers within 5e-5; B coverage: full, PRMBench-only qualifiers; C null+math: claims survive, math sound; claim 2 weakened (post-hoc channel beats all fusion; block gain = down-weighting of anti-oriented channels; position prior), claim 4 range corrected. `run_20260927/RED_TEAM.md`; step index alone verified at 0.6617 | Step 450 commit |
| 2026-09-27 | Stage A done, STOP | Discussion with Omri before stage B | |
| 2026-09-27 | Stage B frozen | PROTOCOL_STAGE_B.json (2b3971383); pre-run code review PASS with non-blocking fixes | 2b3971383 |
| 2026-09-27 | Amendment B1 | after the fold-0 smoke: label-free partition diagnostic; G1 arms on the stage-A tail partition; re-review PASS | 7fa255e8f |
| 2026-09-27 | Stage B full run | run_20260927_stage_b: COMPLETE, 0 failed fits, replay 1.1e-16, stage-A metrics reproduced 0.0; 266 s | pending red team |
| 2026-09-27 | Stage B red team complete | A recomputation: all claims confirmed (own DS EM); B coverage: full population, 5/5 folds, class and PB qualifiers; C nulls: claim 1 weakened (whole-answer label swap reproduces the filter gain; +0.0043 survives position adjustment), weights claims confirmed with dependence mechanism. `run_20260927_stage_b/RED_TEAM.md` | Step 451 commit |
| 2026-09-27 | Stage B done, STOP | Decision with Omri | |
