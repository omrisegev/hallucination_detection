# Global hallucination detection: moment fusion on the historical24 contract

Authorized2026-09-11. Separate codex/global-moment-fusion-24-v1 from75fdcb35.
This is a retrospective TRANSFER of feature definitions/fusion algorithms,
not a same-answer fitting experiment or untouched confirmation. No change to
the active localization job. One CPU thread, low priority background process.

Historical mixed_v2/full/iu_pcr complete-case population, exact24 cell roster,
row order, crop rules, labels and problem IDs. Keep the historical QA9/math15
partition, including its older QA task mix, rather than silently replacing it.
Data8.4GB already local. Match source SHA256 against the existing approved audit.

Within every answer: top10 token mean separately for each of six frozen moment
coordinates [H15,V15,m3_15,a,a^2,a^3]. Within each cell: N answers x6 summaries,
z-score across answers, fit without correctness labels, score those same rows.
This is the historical transductive unsupervised contract; no new train/test
split, supervised fitting, gate or ProcessBench readout. Ordinary answer AUROC.
No claim that these pooled weights were learned from one answer. No new tail.

Candidates: mean,IU-PCR,exact one-hidden-unit Gaussian RBM,B3, plus both energy
models before learning. The fit_matrix entrypoint extracts the exact existing
solver body; fit_all still calls it for localization and must replay unchanged.
Keep all frozen optimizer settings, six coordinates, high-risk mean orientation,
IU defaults, B3 one group and deterministic seed recipe. Same method names do
not imply identical weights across the two tasks. Short token traces are allowed
here because fitting observations are answers, not tokens.

Other current anchors: raw Varentropy15/50; normalized15 contribution mean/IU,
using coordinatewise top10 summaries and cell-local normalization. Raw V15 is
top10 of summed token contributions, distinct from sum of top10 contributions.
Retain saved direct probability17 mean/IU/Joint-shrinkage and entropy baselines,
even though their historical input contains tail. Explicitly label them saved
references. Replay Historical IU-PCR from its original feature bundle and require
the exact saved AUC; require all saved comparator AUCs to replay.

No ranking from partial cells. Publish complete per-cell rows as available and
all24/QA9/math15 means only once all24 cells complete. Pure fit failures remain
visible; no silent fallback/zero score and no mean over fewer successful cells.
Store score arrays, feature matrices, groups, normalization and model states.
All settings fixed before outcomes; this is exploratory transfer with no winner
threshold. Six paired contrasts: IU-historicalIU,RBM-IU,B3-RBM,IU-mean,
RBM-initialRBM,B3-initialB3.10000 paired problem-group draws within every cell,
then paired cell resampling for macros,95% exploratory intervals. These intervals
condition on fixed transductive fits, and do not include refitting or all prior
research choices. Same within-cell grouping convention as the prior24 experiment;
not a newly audited cross-cell source grouping. Do not claim simultaneous
significance or select a publication winner from this table.

Automatic separate arithmetic verifier checks scores, parameters, AUROC and
macro means. CSV/JSON first; no new HTML until results are discussed.
