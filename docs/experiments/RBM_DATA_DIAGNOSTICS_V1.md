# Saved-data diagnosis before expanding the RBM

Approved by Omri, 2026-09-11. Parent 44f9bced8; dedicated worktree and branch.
No refit, restarts, CD, new architecture, hyperparameter search or DUFS changes.
Both original six-feature and twelve-feature (moments through order six) banks
receive the same diagnostics on the full corrected 13,769-answer v3 benchmark.

## Frozen readout and evaluation

Top-10 token mean per step, then Claude's exact first_near_max: all scores within
0.25 population SD of the answer maximum are raised above it, earliest first,
epsilon 1e-6 times max(SD,1e-12). Preserve original scores. No missing-value fill.
Use the saved common mean-entropy gate and source folds. Recompute PRMB ranking
and PRMScore with q0.8 from other-fold transformed steps; labels do not set q.

Eleven risk arms: learned and initial RBM6/RBM12; entropy; raw Varentropy15/50;
Varentropy15 normalized equal/IU; RBM weight shrinkage; shared diagonal RBM.
All old/new pairs; original length and random controls stay unchanged.
Full source-group paired bootstrap, 10,000 draws. Primary learned RBM6 and RBM12
new-minus-old intervals 97.5%; remaining contrasts 95%. Report PB all8/Q4/Q8,
each cell, within-answer and pooled PRMB AUC, PRMScore, failures and coverage.
Bootstrap intervals for PB and within-answer AUC condition on saved models and
calibration; historical research-selection uncertainty is not included.

## Diagnostics (no label-based fitting or orientation)

Token features are reconstructed using saved normalization and columns, and
saved trained/initial top10 step scores must replay exactly for every fit.
PRMB labels apply to step means of standardized features only. PB post-first-
error steps are NOT assumed erroneous. Means/variances need two steps per class;
AUC needs one of each. Sample variance uses ddof=1; <=1e-12 is explicitly counted
as numerically zero and excluded from variance ratios. Ratios use log(var1/var0).

Seven fixed strata: all, entropy <=/> within-answer median, token length <=/>
median, early ceil(S/2) steps / remaining late steps. Entropy stratum uses the
step mean of the existing token-entropy stream. Strata ties stay in lower group.
All diagnostic distributions retain answer/cell/group identities and coverage.

Residual = Z-a-P(h=1|Z)w BEFORE readout. Compare residual correlation spectrum
with four exact Gaussian/Bernoulli samples from the SAME saved RBM, same trace
length, hash(uid,bank,replica,purpose) seeds; no renormalization or refit. Hidden
prior sigmoid(b+a.w+||w||^2/2), visible conditional N(a+h*w,I). These are conditional
model checks, not formal p-values or proof of latent class recovery.

Chronological step-mean residual correlations at lags1-3 retain actual gaps.
For each stratum independently, permute residual vectors among its positions
four times. Require three eligible lag pairs and nonconstant columns. No metric
on artificial shuffled answers. Compare signed correlations to permutation means;
also retain per-feature values. Length/position strata address obvious confounding
but do not establish causal temporal information.

Feature reliability uses feature step means; fusion uses saved top10 scores.
This difference is declared: feature-versus-fusion AUC is not an aggregation-
controlled ablation. Assess regime reversals without selecting a label-driven
feature. Report task failures with exact/early/late/no-error gate, gained/lost,
and whether the true first error is the longest step. Saved initial-vs-learned
metrics and coefficient changes assess training; one fit cannot test restarts.

## Acceptance and stopping

Source SHA256 manifests; completed reviewed dependencies; full baseline metric
replay; Claude entropy/Varentropy50 exact full-vector replay; independent metric
and diagnostic checks; short/constant/tied/missing/unknown-label tests.
27-answer smoke checks mechanics/runtime only. Conclusions use full population.
Do not silently overwrite a frozen flawed result: identify corrections separately.
Return chat tables first, CSV/JSON and concise history/direction notes; no HTML.
Recommend only a smallest follow-up tied to measured task failures, considering
alternative explanations. No arbitrary gain threshold. Stop before model training.
