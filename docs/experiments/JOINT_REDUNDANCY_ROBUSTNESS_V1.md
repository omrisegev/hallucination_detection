# Joint L-SML redundancy-robustness test v1

Development only on the frozen 13,769-answer localization population; no untouched
confirmation. Script: `scripts/run_joint_redundancy_robustness_v1.py`. Results:
`results/joint_redundancy_robustness_v1/{RUN,RUN_L24,RUN_COMBINED}.json` (OOF arrays
and logs are local caches, ignored by git).

## Question

When redundant streams are added to the eight-stream L08 locator roster, does the
Joint global+local factor model lose less localization quality than Continuous
L-SML, IU-PCR and the equal reference? Secondary: can Joint run on L08 at all
without hand-supplied groups, given that its automatic partition contains pairs?

## Contract

- Rosters: L08 (Step-397 diverse eight); L11 = L08 + digit top1, Renyi a8, logtail50;
  L14x = L11 + H0lim innovation, rank-3, rank-10; L14 exactly as Codex's
  `soft_joint_auto_v1` (replay anchor); L24 = every eligible step stream.
- Five outer source folds (`FOLDS_V2`); every weight, group discovery and Joint fit
  uses training answers only; the frozen incumbent gate scores all OOF locators.
- Arms: `continuous` (Continuous L-SML, its own residual-K groups);
  `joint_hier` (Joint's own LOAO consensus groups with minimum size two via
  `fit_joint_pairs_checked`, hierarchical readout); `joint_global_v` and
  `joint_inverse` (same fit, global-loading and lambda-0 model-inverse readouts);
  `joint_lsmlgroups_hier` (same Joint fit on the partition Continuous L-SML found);
  `iu` (frozen `IU_FIT_DEFAULTS`); `equal` (labelled reference row, not a method).
- No label enters any fit; no group is supplied by hand. Paired source-group
  bootstrap, 2,000 draws, for L11/L14x/L24 minus L08 per arm, arm minus continuous
  per roster, and the difference in differences against continuous.
- Replay gates: incumbent 43.2546% / .776036; L08 continuous 43.7402% / .778143;
  L14 continuous 39.3519% / .755577; L24 continuous 38.9074% / .750769; Codex's
  L14 Joint global 39.8716% and hierarchical 40.3024%. All replay exactly.

## Result (PB macro-F1 / PRMB within-AUC)

| arm | L08 | L11 | L14x | L14 (Codex) | L24 |
|---|---|---|---|---|---|
| continuous | 43.7402 / .7781 | 41.1865 / .7660 | 39.4707 / .7557 | 39.3519 / .7556 | 38.9074 / .7508 |
| joint_hier | 43.5862 / .7769 | 39.7326 / .7589 | 40.3581 / .7644 | 40.3024 / .7640 | **43.1289 / .7754** |
| joint_lsmlgroups_hier | **43.8135 / .7783** | 38.9615 / .7503 | 39.1874 / .7519 | 39.2059 / .7518 | 38.5442 / .7482 |
| joint_global_v | 39.9742 / .7602 | 39.5944 / .7589 | 39.8149 / .7578 | 39.8716 / .7576 | 39.2631 / .7543 |
| joint_inverse | 38.6445 / .7452 | 38.3046 / .7502 | 39.0329 / .7516 | 38.9524 / .7516 | 37.8510 / .7473 |
| iu | 38.5701 / .7486 | 39.5188 / .7580 | 38.7125 / .7468 | 38.7694 / .7465 | 26.0678 / .5415 |
| equal (reference) | 42.5287 / .7743 | 42.3803 / .7746 | 41.5782 / .7726 | 41.6539 / .7723 | 40.2021 / .7651 |

Paired bootstrap (95%): on L08, joint_hier minus continuous -0.15pp [-0.63, +0.34] and
joint_lsmlgroups_hier minus continuous +0.07pp [-0.10, +0.26] (ties). Continuous
loses -2.55 [-3.71, -1.36] on L11, -4.27 [-5.62, -2.88] on L14x and -4.83 [-6.25,
-3.37] on L24. joint_hier loses -3.85 on L11, -3.23 on L14x and only -0.46
[-0.85, -0.07] on L24; joint_hier minus continuous on L24 +4.22pp [+2.91, +5.54],
within +.0247 [+.0214, +.0278]. IU collapses on L24 (-12.5pp).

## Mechanism (from the per-fold audits)

- The pair route works: on L08 Joint's own consensus is 2/3/3 with one pair, all
  five folds PASS multistart, native-map and Jacobian audits (condition 8.4).
- The hierarchical readout is what carries the digit view: digit share 0.34 with it,
  0.03-0.07 for the global-loading and model-inverse readouts and for IU.
- Robustness is not a property of the shared factor. It comes from Joint's
  stability-based K selection (smallest stable K, here always 3) plus one vote per
  group: on L24 Joint isolates the three digit streams as one group (3/13/8) while
  Continuous L-SML's residual rule fragments the redundant families into K=7 and
  the digit vote falls to 2%.
- The failure on L11/L14x is group impurity: Joint's digit group there is
  {digit top2, token-clock, digit top1, Renyi a0.25}. `hierarchical_joint_weights`
  builds each group's virtual classifier from the global loading v, in which the
  digit entries are tiny, so the foreign stream dominates the group and the digit
  share drops to 0.13. Continuous L-SML uses a within-group SML eigenvector instead.

## Status

Development evidence. joint_hier on L24 (43.13%) is a research candidate below the
L08 continuous point (43.74%) and below the incumbent-plus-L08 numbers of Step 397;
no successor is declared. The identified next algorithmic step is a within-group
readout inside Joint that does not depend on v (group factor u_g or within-group
SML), tested as one variant on the same contract. Equal rows are reference only.
