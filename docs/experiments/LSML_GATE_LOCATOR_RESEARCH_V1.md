# L-SML gate/locator research v1

## Objective

Combine the completed Fusion Independence Atlas with the Step-396 gate and the
six-stream Continuous L-SML locator.  This is development-only evidence.  No
new model forward pass is permitted and no result is an untouched confirmation.

## Firewall

- Extraction, normalization, covariance, grouping, Joint fitting and fusion
  weights are label-free.
- Development labels may select a roster only inside nested source folds.
- Every held source fold is excluded from its fitted weights.  Inner selection
  excludes both the outer and inner folds.
- The ProcessBench gate keeps the frozen within-cell midrank threshold `.33`.
- ProcessBench and PRMBench metrics are reported separately.

## Studies

1. Locator ladders compare Continuous L-SML on bank6, the proposed diverse
   atlas roster, incumbent-plus, a structurally valid 14-stream Joint roster,
   and the full eligible step roster.  TCN is represented both as four
   primitive residual experts and as one label-free four-view compression.
2. Gate ladders compare six, ten, all eligible streams and a structurally valid
   four-family Joint model.  Since digit-rate is a singleton, the valid Joint
   groups first form virtual experts; Continuous L-SML then combines those
   experts with digit-rate.
3. One finalist per insertion point must recur in at least four of five nested
   outer folds.  Finalists enter a frozen 2x2 locator/gate interaction table.
4. Expensive 10,000-draw paired source bootstrap is restricted to frozen
   finalists.  Discovery uses point estimates, fold ledgers and diagnostics.

## Interpretation

Continuous L-SML is an exploratory continuous relaxation, not a theorem-level
independence certificate.  Joint is run only where `K>=3` and every structural
group contains at least three streams.  A failed stability or multistart audit
blocks promotion but remains a diagnostic result.
