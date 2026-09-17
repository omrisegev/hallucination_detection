# Step412: feature mass inside Joint group discovery

One frozen structural intervention relative to Step410, not the rejected
Step411 minimax-pruning rule. Replace group discovery everywhere it is used
inside staged membership and per-deletion regrouping. Keep Step410's current-
factor proposal ranking, top3 budget and95% initial-information constraint.

For each training covariance C, let K_ij=corr(i,j)^2. Solve
min_{m>=0} .5*m'K*m - sum(m), then normalize m to sum one. The projected KKT
residual must be<=1e-6; failure is explicit. L-BFGS-B settings and tolerances are
fixed in code. No positive floor or named feature protection is added.

Use the leading eigencomponent of diag(sqrt(m))*C*diag(sqrt(m)) for a complete-
covariance rank-one PROVISIONAL projection, with its covariance extension for
zero-mass features. The residual affinity is abs(C-vv'), INCLUDING its diagonal.
The diagonal matters to repetition: otherwise a repeated private-noise diagonal
becomes a newly counted off-diagonal edge. This changes only structure discovery;
the final checked Joint off-diagonal objective and feasibility guards are intact.

Use the mass-normalized symmetric affinity operator, extend eigenfunctions to
zero-mass coordinates, and row-normalize its topK positive eigenfunctions.
Deterministic farthest-point initialization (12-decimal distance ties), followed
by weighted Lloyd KMeans, clusters all coordinates. Do not drop zero-mass rows
merely because their grouping measure is zero. The existing sparse Joint still
decides row/group membership. Minimum-two admissibility remains unchanged.

Retain four inner source-fold deletion partitions and K={3,4}. Their consensus
uses mean coassignment and the full-training mass. Rank admissible candidates by
median/mean/minimum MASS-WEIGHTED NMI, then smaller K. This is explicitly a new
stability measure, not ARI. Record parts, masses, optimizer diagnostics and seeds;
this discovery is deterministic even though the public seed contract is retained.

The kernel energy, weighted projection and fixedK grouping have split-mass
repetition checks. These do NOT prove full-model near-copy robustness: final
Joint is still unweighted, groups/admissibility are discrete, and covariance
information is a surrogate for localization. Exact aliases remain canonicalized
before this module as in all prior arms. The historical block-pair-weighted loss
is a different previously evaluated control, not this intervention.

Six mechanism tests and five saved-base covariance feasibility checks precede
quality evaluation; they are not evidence of improvement. Freeze this protocol
and source hashes, then25 fresh fits on all13769 answers/145597 steps in five
banks: base50+BOCPD,15 exact copies,15 iid noise,15 near copies,15 correlated noise.
Same five outer source folds, hybrid other-answer training, fixed Tail15 Top10
q=.33 non-digit gate and H1 orientation. No digit inputs, new inference/downloads,
held-label feature/count selection or unreported fallback. This is studied
development data, not untouched confirmation.

Retain42 metric bundles: prior controls/history, Steps408/409/410/411, and five
new arms. Six primary contrasts: newnear vs Step410near (not the weaker411near),
newbase vs Step408base, and four new additions vs newbase.10000 paired source-
group draws, two endpoints, Bonferroni confidence1-.05/12. Each preservation
criterion requires native13769, PB lower bound>-.01, within lower bound>-.002.
Prior-base quality must pass separately from new-base addition preservation.
Over40 remains a hoped-for new result, not a claim from the historical comparator.

Independent audit: fixed inputs/hashes; sparse objective/calibration/membership;
mass KKT and normalization; weighted NMI from contingency tables; replay all
grouping proposals and native guards; fixed-reference information path/readout;
held scores/native failures;42 metrics and all12 intervals. Replay invalid fits
to verify failure accounting. At most25 fits,3 membership rounds, P-8 accepted
deletions with3 attempts each. Keep failed outcomes; no threshold changes or
new variants after seeing the held results. Finish this bounded stage and report.
