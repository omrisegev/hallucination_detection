# Mechanism analysis: hard filtering before DUFS-LIU

**Date:** 2026-08-08
**Status:** Retrospective analysis of the registered 24 development cells.

## Short answer

Hard filtering did not improve the current method. The best row remained the
unfiltered mixed-v2 DUFS-LIU result: **0.776562 cell-macro AUROC**.

The default deployed filter reduced mixed-v2 DUFS-LIU to **0.774249**, a loss of
**0.231 AUROC points**. More aggressive filtering caused larger losses. At the
strictest tested setting, mixed-v2 DUFS-LIU reached **0.764153**, which is
**1.241 points below** the unfiltered result.

## Did filtering make DUFS itself more useful?

No. This is different from asking whether the final score changed.

Without filtering, mixed-v2 DUFS-LIU added **0.048 AUROC points** over IU-PCR on
the same full feature pool. Under the default deployed filter, it was **0.025
points worse** than filtered IU-PCR. The difference-in-difference was therefore
**-0.073 points**, with a 95% cell-bootstrap interval of **[-0.128, -0.019]**.

The fixed-stable contract gave the same conclusion. Its default-filter
difference-in-difference was **-0.053 points**, with interval
**[-0.106, -0.003]**.

The strictest fixed-stable setting produced a small positive DUFS contribution,
but its final DUFS-LIU AUROC was **1.079 points below** the unfiltered method.
That is not a rescue: the base method became much worse while the Laplacian
recovered only a small part of the loss.

## What the filter removed

The filter and DUFS already agreed strongly about feature importance:

| contract | median Spearman between estimated rho and full-pool DUFS gate | mean gate of kept features | mean gate of removed features |
|---|---:|---:|---:|
| fixed-stable | 0.792 | 0.787 | 0.306 |
| mixed-v2 | 0.794 | 0.767 | 0.212 |

Thus the hard filter mostly removed features that DUFS had already softened in
the graph metric. It did not reveal a cleaner set that DUFS had failed to find.

Under mixed-v2, the default filter reduced the pool from a mean of **28.4** to
**21.3** features. The mean DUFS effective count changed only from **19.4** to
**17.8**. This is consistent with removal of already-low-gate coordinates,
rather than discovery of a new graph geometry.

## Why the result is plausible

DUFS gates affect only construction of the sample-neighbourhood graph. Ordinary
IU-PCR and the final DUFS-LIU solve can still use a weak feature as complementary
information. Hard deletion removes that feature from the covariance equations,
the two-component subspace, and the final fusion at the same time.

The Laplacian modification was already small. The mean cosine between DUFS-LIU
and IU-PCR weights was **0.9980** under unfiltered mixed-v2 and became even closer
to one, **0.9991**, after the default filter. Filtering therefore made the final
Laplacian solution more similar to IU-PCR, not more distinct.

This suggests the following interpretation:

1. DUFS already performs the useful part of feature suppression softly when it
   constructs the graph.
2. Estimated rho is useful for fusion weights, but its hard threshold is too
   destructive as a preprocessing selector for the Laplacian method.
3. Some low-rho features may still contribute through covariance structure or
   local complementarity, even if they should have little influence on graph
   distance.

The third point is an inference from the observed degradation, not a directly
identified causal explanation.

## Strictness conclusion

Making the filter stricter did not help:

| mixed-v2 setting | mean features | DUFS-LIU AUROC | change from full |
|---|---:|---:|---:|
| no filter | 28.4 | 0.776562 | 0.000 pp |
| rho max / 3 | 21.3 | 0.774249 | -0.231 pp |
| rho max / 2.5 | 20.3 | 0.772695 | -0.387 pp |
| rho max / 2 | 18.6 | 0.768532 | -0.803 pp |
| rho max / 1.5 | 15.2 | 0.764153 | -1.241 pp |

The losses were not caused by a forced three-feature fallback. Even at the
strictest setting, mixed-v2 retained between 7 and 19 features across cells.

## Validation and leakage checks

- The fit stage did not read labels.
- All score files were hashed before the report opened labels.
- No score checkpoint contains a label array.
- The unfiltered fixed-stable IU-PCR scores reproduce the previous freeze in
  24/24 cells with maximum absolute error 0.
- The unfiltered fixed-stable DUFS-LIU scores reproduce the previous freeze in
  24/24 cells with maximum absolute error 0.
- The default fixed-stable deployed-U-PCR scores reproduce the previous freeze
  in 24/24 cells with maximum absolute error 0.

## Decision

Do not add the deployed hard filter before IU-PCR or DUFS-LIU. Keep unfiltered
mixed-v2 DUFS-LIU as the best row from this experiment. Do not spend another
cycle tuning the same rho threshold on these 24 cells.

If feature suppression is revisited, it should separate the two roles:

- soft gates for graph construction;
- fusion weights for the final score.

A feature should not be deleted from both roles merely because its global
estimated rho is below one threshold.

See [the complete report](REPORT.md),
[the AUROC plot](figures/auroc_by_filter.png), and
[the incremental-gain plot](figures/dufs_incremental_gain.png).
