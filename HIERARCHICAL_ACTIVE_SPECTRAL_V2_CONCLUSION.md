# Hierarchical-active spectral v2 — conclusion

## Decision

**Stop the cross-family shared-correction candidate.**  At the registered
20-target-label point, `hybrid_domain_active` is **-0.19pp** versus U-PCR, with
a cell-bootstrap 95% interval of **[-0.41, +0.02]** and 8 wins / 16 losses.
The result converges from -0.16pp after 10 repetitions to -0.19pp after 20, so
the failed contribution gate is not explained by an unfinished Monte Carlo run.

The rejected claim is that a correction vector learned from other dataset
families transfers to the target.  The experiment does not reject trusted
labels in general.

## What worked

### 1. Active acquisition rejects a harmful transferred prior

The cleanest new result is the paired comparison that holds the donor model,
target-label budget, and two-score head fixed:

- hybrid active minus hybrid uniform at 20 labels: **+0.78pp**,
  95% CI **[+0.53, +1.08]**, 23W/1L;
- the gain is positive from 5 through 80 labels and is 24W/0L at 40 labels;
- active labels reduce the mean transferred correction coefficient from its
  prior value 0.385 to 0.193 at 20 labels; uniform labels reduce it only to
  0.329.

Thus the D-optimal policy found rows that efficiently exposed a bad correction.
It is a **prior-rejection/safety mechanism**, not a detector improvement: after
the rescue, the score still remains 0.19pp below U-PCR.

### 2. The shared-correction implementation works under its stated mechanism

In the disjoint `shared_correction` meta-world, the combined method improves
U-PCR by **+47.63pp**.  It is approximately neutral when U-PCR is sufficient
(-0.06pp) and under the family-shift negative control (-0.08pp).  This verifies
the implementation and establishes a sharp boundary: cross-family supervision
is valuable when the correction is genuinely shared.

### 3. The conservative local two-score head remains the safest labelled head

At 20 labels, local uniform fitting is -0.05pp versus U-PCR, compared with
-0.15pp for the earlier controlled-stratified reproduction and -0.18pp for
local active acquisition.  At 80 labels, local active is nominally +0.07pp,
but its interval [-0.24, +0.43] crosses zero.  It remains a useful control, not
a demonstrated improvement.

## What failed, and why

The donor-only same-domain LOFO head loses **-1.44pp**, CI
**[-2.01, -0.92]**, in 23/24 cells.  Updating it with uniformly selected target
labels still loses -0.97pp at 20 labels.  Active selection repairs 0.78pp of
that damage but cannot manufacture transferable information that the donor
families did not contain.

Pooling both QA and math donors is less harmful (-0.96pp) than same-domain
pooling, but still loses in 21/24 cells.  Because it also uses more donor labels,
this is evidence that the QA/math hierarchy was not useful—not evidence that
cross-domain pooling is a better detector.

The cell-level oracle switch between U-PCR and the 20-label hybrid candidate is
only **+0.12pp**.  Even a perfect selector for the eight winning cells would not
produce the one-point contribution sought here.  A more elaborate safety gate
may be operationally sensible, but it is not the next research headline.

## Comparison with prior experiment families

The ranking of ideas is now consistent across several independent cycles:

1. **U-PCR's low-dimensional head is the strongest real-data default.**
2. **Small, target-local perturbations are safest**, but have not improved it.
3. **Active labels can invalidate a bad prior efficiently**, but do not add new
   information by themselves.
4. **Cross-family correction transfer is materially worse** than a local head.
5. **Extra spectral dimensions, inverse tails, and pseudo-label feedback are
   worse still**: anchored-6 lost 0.36pp; stable SDSF lost 2.91pp versus SU-PCR;
   the full inverse tail lost 3.32pp; pseudo+gold lost 3.97pp to anchored-6.

The repeated pattern is that methods work when they preserve U-PCR's leading
ranking and fail in proportion to how much unsupported information they add.
Synthetic dependency correction is real, but the correction is not stable
across the current hallucination cells.

## Consequence for the next research cycle

Do not run another linear correction, pooling, or self-training variant on this
feature bundle.  The remaining positive result in the history is in a different
channel: a label-handed feature-subset oracle has +2.25pp of room, and a
half-split oracle transfers 84% of that gain.  The next hypothesis should
therefore test **label-efficient, stability-selected feature/subset adaptation**
with U-PCR kept as the fusion rule, rather than learning another correction to
the U-PCR weights.

That experiment must compare against the +0.12pp selector ceiling measured here
and use nested train/selection/test splits so labels used to choose features do
not score the chosen subset.

Evidence:

- `results/hierarchical_active_spectral_v2/REPORT.md`
- `results/hierarchical_active_spectral_v2/contrasts.csv`
- `results/hierarchical_active_spectral_v2/convergence.csv`
- `results/hierarchical_active_spectral_v2/replicates.csv`
- `SPEC_HIERARCHICAL_ACTIVE_SPECTRAL_V2.md`
