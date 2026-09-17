# Refine the noise-aware Joint model with the unchanged95% rule

Step404's sparse Joint objective learned zero loadings for all15 noise channels
and exact model aliases for copies, producing identical base/copy/noise scores
with native coverage13769. It retained all51 genuine original channels, thus
replaying full-support group weighting38.0948%, below the known automatic38.73%.

Compose the learned noise-only membership with the EXISTING alternating Joint
information refinement. On the nonnull canonical coordinates, reuse the final
valid groups, five-start checked fits,95% initial per-factor information stop,
minimum2/group, maximum P-8 removals. No new retention threshold. Then use the
same model-derived group-reliability readout; final sign uses original H1,
which remains removable. An internal first-coordinate sign during path replay
has no influence on the information-based support; no correctness sign fitting.

This is one declared composition of two Joint membership mechanisms, not an
independent feature-type prefilter. Null rows arise from penalized global/local
loadings. Refinement need not protect a noise factor after that factor is gone.
Reuse Step404 noise/alias learning without new fitting choices. Identical active
training matrices, groups and seeds may reuse a computed refinement; all held
scores use the corresponding bank's own coordinates. Preserve all failures and
H1 fallbacks; never force keeping an original-bank feature.

Full13769 matched base/copy/noise benchmark and original fixed gate, source-fold
hybrid fitting, no digits. Five primary paired contrasts,99.5% intervals from
10000 source-group bootstrap draws: refined versus frozen Step403 automatic
group head in each bank, plus refined stress versus refined base. Full sparse,
all previous controls and strong innovation5/historicalBOCPD remain in table.
Same practical margins PB-1pp/within-.002 AND native13769. Explicit exact-copy
score/peak equality, retained features/noise and BOCPD. No parameter search.

Restoring the already observed38.73% is not a new accuracy breakthrough. The
decision question is whether this combined Joint selector preserves that
quality under additions while recovering the previously failed native fits.
Exact copies and iid noise do not cover approximate copies, structured nuisance
families or genuinely useful independent streams. Those remain explicit limits.
