# partition_switch_v1: a label-free stopping rule for the grouped DS-estimate fusion

Omri, 2026-09-28: try a label-free rule that decides when not to use the grouping (falling back to the DS-filtered plain average),
so that the estimate-weighted grouped method (algorithm_decisions_v1 P0__EQ_DSM) stops losing where its partition is poor.

- Protocol `PROTOCOL.json` frozen at 37d41f181; runner `scripts/experiments/partition_switch_run.py` with pre-run review fixes
  (3dfb66511); run `run_20260928` (0c90aa4e4): no refit - stored scores recombined; BASE replay 8.9e-16, stored arms replay the
  runner METRICS exactly. Red team `run_20260928/RED_TEAM.md`.
- Rule: per bank and fold, use the grouped method iff a label-free statistic of the fit rows is at most a threshold; the statistic
  (S1 between-group dependence, S2 largest-group share, S3 rank-one misfit of the group-mark covariance) and threshold for bank b /
  fold k are chosen on cells with bank != b and fold != k.

## Results (PRMBench within-answer AUC, difference from the DS-filtered plain average)

| Arm | 13 | 13+d | 20 | 20+d | 32 | 32+d | 51 | 51+d | 8-bank mean [95%] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| grouped method, always (EQ_DSM) | +0.0016 | -0.0011 | -0.0013 | -0.0079 | +0.0129 | +0.0091 | -0.0133 | -0.0018 | -0.0002 [-0.0013, 0.0008] |
| **switch (leak-free, as frozen)** | 0 | 0 | 0 | 0 | +0.0129 | +0.0091 | 0 | 0 | **+0.0028 [0.0020, 0.0036]** |
| switch, hem version | 0 | 0 | 0 | 0 | +0.0145 | +0.0100 | 0 | 0 | +0.0031 [0.0022, 0.0040] |
| blend (no parameter) | +0.0009 | +0.0004 | -0.0003 | **-0.0020** | +0.0123 | +0.0082 | **-0.0053** | +0.0005 | +0.0018 [0.0012, 0.0025] |
| switch with the bank FAMILY held out (red team) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Reading
- By the frozen criterion the switch succeeds (no bank loss, 8-bank mean above 0), and its gains on 32/32+d are content (77% / 90%).
- It is not a transferable rule. The rule always picked S1 and a threshold that separates the 32-channel family (S1 ~0.15) from
  everything else (>= 0.17); with the whole family held out it never turns grouping on and the gain is 0. Outside that family the
  relation between S1 and the gain reverses. The mechanism "grouping pays where the groups are close to independent" is not
  supported by these data.
- The blend fails (losses on 20+d and 51, mostly positional).

## Consequence for the external test
The frozen protocol says a successful switch replaces the unconditional grouped method as the secondary external candidate. The red
team shows the success is a one-family effect; the recommendation to Omri is to keep the plain average as the primary candidate and
the unconditional grouped method as the secondary, and to report the switch only as a labelled diagnostic (Omri to decide).
