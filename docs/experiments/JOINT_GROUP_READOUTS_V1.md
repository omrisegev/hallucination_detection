# Joint L-SML within-group readouts v1 (Step 399 [Claude])

Development only on the frozen 13,769-answer population; no untouched confirmation.
Module: `spectral_utils/joint_group_readouts.py`. Runner: `scripts/run_joint_redundancy_robustness_v1.py`
(arms `joint_hier_u`, `joint_hier_sml`). Results: `results/joint_group_readouts_v1/RUN.json`.
Same contract as `JOINT_REDUNDANCY_ROBUSTNESS_V1.md` (rosters L08 / L11 / L14x / L24, five source
folds, Joint's own LOAO consensus groups with pairs, frozen incumbent gate, 2,000-draw paired
source-group bootstrap). Digit streams are still in these rosters: the test targets the readout
mechanism, and every digit-inclusive number is historical under the 2026-09-17 digit decision.

## Question

Step 398 found that `hierarchical_joint_weights` builds each group's virtual classifier from the
global loading v, so a foreign stream that lands in a group of complementary streams dominates it.
Does a within-group direction that does not depend on v remove that failure while keeping the
L08 and L24 results?

## Arms

* `joint_hier`      existing: virtual_g = x_g v_g.
* `joint_hier_u`    virtual_g = x_g u_g (fitted group factor; sign fixed by u_g . v_g >= 0).
* `joint_hier_sml`  virtual_g = x_g e_g, e_g the leading eigenvector of the within-group
                    off-diagonal covariance (Continuous L-SML's own within-group rule).
Second stage (cross-group SML on the virtual classifiers) unchanged for all three.

## Result (PB macro-F1 % / PRMB within-AUC)

| arm | L08 | L11 | L14x | L24 |
|---|---|---|---|---|
| continuous | 43.7402 / .7781 | 41.1865 / .7660 | 39.4707 / .7557 | 38.9074 / .7508 |
| joint_hier (v) | 43.5862 / .7769 | 39.7326 / .7589 | 40.3581 / .7644 | 43.1289 / .7754 |
| joint_hier_u | 43.4899 / .7734 | 42.4840 / .7553 | 39.3801 / .7391 | 42.1619 / .7472 |
| **joint_hier_sml** | 43.4825 / .7767 | **42.3668 / .7697** | **43.0245 / .7773** | **43.2194 / .7754** |
| equal (reference) | 42.5287 / .7743 | 42.3803 / .7746 | 41.5782 / .7726 | 40.2021 / .7651 |

Digit share of |w| (mean over folds): joint_hier_sml .34 / .30 / .32 / .34 across the four rosters;
joint_hier (v) .34 / .12 / .13 / .34; continuous .34 / .10 / .07 / .02.

Paired bootstrap, 95%: joint_hier_sml minus continuous on L11 +1.18pp [+0.14, +2.19],
L14x +3.55 [+2.32, +4.72], L24 +4.31 [+3.00, +5.62]; on L08 -0.26 [-0.73, +0.22].
Roster ladder for joint_hier_sml: L11 - L08 -1.12 [-1.83, -0.40]; L14x - L08 -0.46 [-0.95, +0.05];
L24 - L08 -0.26 [-0.68, +0.15]. The v-based readout loses -3.85 / -3.23 / -0.46 on the same steps.

## Reading

* The within-group SML direction fixes the impurity failure: on L11/L14x the digit vote is
  restored and the losses versus L08 shrink from about -3 to -0.5 points. All replay anchors exact.
* The group-factor direction u_g does not: for redundant families u_g captures nuisance
  co-movement, so its virtual classifiers are poor and within-AUC drops on every wider roster.
* joint_hier_sml is now flat within about half a point from 8 to 24 streams, where Continuous
  L-SML loses 4.8 points. The remaining gap to L08 Continuous (-0.26pp) has an interval including zero.
* No successor is declared: digit-inclusive rosters, development data, one variant tested.
  The mechanism (stability-selected small K, one vote per group, v-free within-group direction)
  is not digit-specific and is the candidate to carry into the digit-free bank.

## Direct contrast: within-SML readout minus v-based readout (same Joint fit, same groups)

| roster | PB | within |
|---|---|---|
| L08 | -0.10pp [-0.26, +0.04] | -.0002 [-.0005, +.0001] |
| L11 | **+2.63pp [+1.67, +3.57]** | **+.0109 [+.0088, +.0129]** |
| L14x | **+2.67pp [+1.64, +3.60]** | **+.0129 [+.0106, +.0152]** |
| L24 | +0.09pp [-0.03, +0.23] | -.0001 [-.0004, +.0002] |

Where the groups were pure (L08, L24) the two readouts coincide; where a foreign stream entered the
digit group (L11, L14x) the v-free direction recovers 2.6 points and .011-.013 within-AUC.
