# Completed AIRCC position experiments: evidence snapshot

Fetched 2026-09-13. Full cached development benchmark: 13,769 answers, including
6,800 ProcessBench and 6,969 PRMBench. Within-answer AUC uses 6,030 mixed-label
answers. All compared rows have full scoring coverage. Frozen Top10, argmax,
labels, folds and entropy gate are unchanged. External fits use other answers
without labels; these are not strict answer-local methods.

| Method | PB macro % | PRMB within AUC | PRMB pooled AUC | PRMScore |
|---|---:|---:|---:|---:|
| Answer-local RBM12 Logit reference | 36.27 | 0.74520 | 0.70585 | 0.62222 |
| Entropy reference | 35.44 | 0.73011 | 0.70266 | 0.62543 |
| Varentropy50 reference | 35.68 | 0.74246 | 0.71578 | 0.63278 |
| Other-answer IU with whole-answer position | 35.52 | 0.76573 | 0.69528 | 0.61442 |
| Answer-local Shrinkage IU plus position covariance prior | 35.81 | 0.75690 | 0.66490 | 0.58991 |
| Same prior, baseline direction and scale change only | 32.14 | 0.74153 | 0.65637 | 0.58285 |
| Answer-local Shrinkage IU on RBM12 bank | 20.27 | 0.69059 | 0.63794 | 0.56861 |
| Other-answer Gaussian factor, loading-map rank 2 | 33.54 | 0.75591 | 0.69256 | 0.60514 |
| Other-answer H1 RBM, loading-map rank 1 | 35.72 | 0.74723 | 0.68612 | 0.60018 |
| Other-answer H1 RBM, loading-map rank 2 | 22.19 | 0.68793 | 0.65303 | 0.56679 |

## What the experiments establish

- Whole-answer IU improves within-answer AUC over frozen RBM12 by 0.02053
  (secondary 95% grouped CI [0.01738, 0.02374]). PB changes by -0.756 pp
  (CI [-1.908, +0.377] pp); PRMScore drops by 0.00780, CI excludes zero.
  It beats its own position-mean control and shuffled-position control.
- Conditional Shrinkage IU beats the scale-only control by +3.670 pp PB
  (primary 98.333% CI [+2.035, +5.281] pp) and +0.01537 within AUC
  (CI [0.01355, 0.01721]). Direction changes add benefit beyond scale changes
  under this fixed two-dimensional IU-subspace extension. This is not
  evidence that the learned covariance prior measures semantic error directly.
- Against RBM12, conditional Shrinkage IU improves within AUC but PB delta
  is -0.465 pp with CI including zero; PRMScore drops by 0.03230, CI excludes
  zero. The native IU baseline here is weak on the 12 moment features.
  Its recovery must not be presented as a gain over historical leading IU.
- Relative to RBM12, whole-answer IU gains 269 final PB successes and loses
  317: 70 early, 247 late, zero gate/invalid losses. Conditional IU gains
  202 and loses 228: 79 early, 149 late, zero gate/invalid losses.
- H1 RBM loading-map rank 2 loses 898 former PB successes, all early, and
  gains 267. All 55 selected fits report optimizer convergence by function
  tolerance. Thus an iteration-cap failure does not explain this run.
  This does not prove a global optimum or rule out other optimization problems.
- Gaussian factor rank 2 improves within AUC over RBM12 but worsens PB.
  It improves over rank 1 and shuffled positions. One of 55 selected shuffled
  factor fits reaches the iteration cap; all retained predictions count.
  Rank here describes the position-feature loading map, not hidden-unit count.

## Interpretation and next step

Position contains useful information for the tested fusion implementations.
No new candidate is a demonstrated overall replacement for RBM12 or the best
simple references. Within-answer ranking gains coexist with weaker first-error
localization and poorer cross-answer calibrated scores. The latter are measured
outcomes; a pure calibration explanation has not been isolated.
Prioritize the completed IU position candidates for failure/late-peak analysis
and finish the already running graph controls before selecting a new model.
Do not automatically increase rank or sweep regularization.
The two remaining model families (conditional-context RBM and convolution) were
not launched by this fetch.

## Verification and remaining work

Jobs 255722 and 255752 completed successfully. The 24 fetched files match
remote SHA256 values. Existing cluster RESULT_REVIEW files are PASS and
include independent endpoint recomputation, fold exclusions and reference replay.
Locally every reported PB macro was recomputed from its eight per-cell rows.
This fetch is not a new independent raw-score/bootstrap audit.
Graph-local job 255753 was still in bootstrap; GraphTV 255754 was scoring.
Remote SQLite/NPZ sources remain on AIRCC; only compact reports were fetched.
The development branch was successfully pushed at f3fa3cc34 after user approval.
Earlier progress text saying push was blocked is superseded.
