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
