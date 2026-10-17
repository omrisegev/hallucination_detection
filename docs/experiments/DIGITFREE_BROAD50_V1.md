# One broad50 experiment, frozen 2026-09-17 before scoring

Question: does equal weight per covariance block improve Joint localization on
a broad, fixed, non-digit bank? Full 13,769-answer development population, v3
labels / v2 source groups. No roster, readout, gate or hyperparameter search.

50 token streams: 15 probability rank risks (1-p1,p2,...,p15), provided-token
surprisal/top1 loggap/censored zero-based rank50/mass above, raw tail15/tail50,
Renyi {.25,.5,1,2,4,infinity,H0lim}, escort varentropy {0,.5,.75,1,2,4},
top2/top1 ratio, 13 strict prefix innovations of shape/varentropy, adjacent
top15 ID turnover, adjacent normalized top50 JS on the union of token IDs.
First token inactive for temporal features. Existing per-answer VE1 Pearson
orientation for VE channels precedes innovation and Top10 reduction. Natural
orientation otherwise. No digit matching, rates, gates or sign anchors.
Each step: mean of largest min(10,n_valid) values in each stream. Unavailable
steps are flagged; answer-standardization uses available entries, then missing
entries are neutral zero in standardized space. Report missing counts.

Five outer source folds: answer-local step standardization, training-source
pooled covariance/groups/weights. This is a HYBRID multi-answer fit, not the
strict answer-only objective. Probability shape H1 is the fixed sign anchor.

Registered arms: entropy H1 singleton, equal broad50 control, Continuous L-SML
with its automatic groups, ordinary Joint hierarchical, block-balanced Joint
hierarchical. Both Joint arms share exactly one partition in each fold and the
legacy readout with small-M averaging guard. No feature is removed/protected.
Grouping uses a bounded four-training-source-fold deletion stability check,
K={3,4}, min group size2 in each partition and consensus. Select highest median
ARI, mean ARI, minimum ARI, then smaller K. This is NOT the historical exhaustive
leave-one-source-out procedure. This first test changes only loss between Joint
arms; it does not claim to isolate bank size against historical L08 results.

Balanced loss: mean squared residual within each unordered group-pair block,
then equal total weight across eligible blocks (including within-group blocks).
Pair weights normalized to mean1 for optimizer tolerance. Same five initial
starts, max5000 sweeps, legacy tolerances; >=4 converged starts, model agreement
1e-5 and loading cosine .999, full profiled global Jacobian rank/condition<=1e8.
Pairs use canonical residual budgets and profiled pair-product identifiability.
No-admissible partition/invalid fit -> explicit entropy singleton fallback;
report native coverage and failure reasons, never silently call it a Joint fit.

PB gate: frozen historical tail15 answer-prominence raw score, within-cell
midrank >= .33 (transductive unlabeled gate). No digit gate is accessed. This
gate was historically developed on these data; not untouched confirmation.
Historical original4 and innovation5 locators replay under the SAME non-digit
gate as context; the old digit-driven 43% headline is ineligible.

Primary contrast balanced Joint minus ordinary Joint on PB eight-cell macro
harmonic clean/error exact accuracy and PRMB mean within-answer AUROC.
Paired 10,000 source-group bootstrap, 97.5% intervals for two endpoints.
Other arms and per-cell results descriptive; no winner promotion from ranking.
Save extraction availability, hashes, seeds, partitions, weights, diagnostics,
OOF scores, full metrics and running state. Small synthetic checks establish
implementation only, not quality. No new model inference, training or sweep.
