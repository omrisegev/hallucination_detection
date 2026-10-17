# DUFS moment selection before RBM - fixed development protocol

Authorized 2026-09-11. Parent codex/rbm-diagonal-variance-v1 at 6e8f46100.
Question: does answer-local feature selection improve localization, beyond
simple redundancy reduction and beyond the RBM initialization itself?

## Fixed design

Full corrected v3 benchmark: 13,769 answers; 6,800 ProcessBench in eight cells,
6,969 PRMBench. Same source groups, folds, annotations, failures and metrics.
No global24 run, new moments, graph-strength search or supervised selection.
All fitting, scaling and column selection use only the current answer.
The external frozen entropy q0.3 gate is explicitly external calibration;
its detector and q were previously chosen using development outcomes.

Rows are all tokens in ONE answer. Columns are the existing compact order6
bank: H15, V15, m3_15, selected surprisal a, a^2, a^3, m4_15, a^4,
m5_15, a^5, m6_15, a^6. m_j = sum_k q_k (-log(q_k))^j for the normalized
top15 probabilities (existing 1e-12 arithmetic unchanged). No tail channel.
Z-score columns on the answer, population std, original constant-column rule.

Four banks, each scored by the original fixed-unit-conditional-variance
one-hidden Gaussian RBM trained for at most100 L-BFGS-B iterations and by
its untrained initialization a=0,b=0,w=2/P (eight arms):

1. All12 varying columns, original order6 reference.
2. Original6 columns, original order3 reference.
3. Adapted DUFS hard-select6: existing spectral_utils/adapted_dufs.py,
   three seeds0,1,2,120 epochs each, Adam.02, batch min(256,T), self-tuning
   k7 sample affinity and two-step random-walk diffusion, STG sigma.5,
   original parameter-free trace/survival-sum objective. Rank average gate
   survival probabilities descending; keep exactly6. Ties use original
   column index. Restore original column order before fitting RBM.
   This is the historical adaptation, not the paper's original optimizer.
4. Low correlation hard-select6: squared Pearson correlations between
   varying standardized columns. Start with the column having smallest
   total squared correlation to the other columns. Greedily append the
   column with smallest summed squared correlation to those already kept.
   Ties use original index. Restore original order. No labels or tuning.

DUFS is a FEATURE selector: its graph nodes are tokens, but it selects
columns, not tokens. All tokens are then scored. Hard selection avoids the
cancellation of multiplicative soft gates by a second z-score. This does
not test a graph penalty in the RBM likelihood or temporal smoothing.

All eight arms use the SAME risk-orientation anchor: mean of the original6
standardized columns. Learned scores may be complemented if negatively
correlated with this anchor. A selector cannot redefine the risk direction.
Each step score is the mean of its top10 token posteriors, or all tokens
for a shorter step. Common gate/readout and earliest ties unchanged.
Untrained posterior is sigmoid(2*mean(Z)), not raw mean(Z); averaging its
token posteriors is not in general equivalent to averaging raw scores.

Fewer than6 tokens or6 varying full-bank columns, or fewer than3 varying
original columns: explicit all-arm failure. Selector errors fail that
selector's two arms; no hidden fallback. Nonconverged but finite fits remain
scored and flagged. No clipping beyond existing feature arithmetic.

## Evaluation and acceptance checks

Two primary contrasts: DUFS trained minus low-correlation trained;
DUFS trained minus all12 trained. Canonical source-group paired bootstrap,
10,000 draws,97.5% intervals for the two primary comparisons; other
contrasts descriptive95%. Report PB macro, Q4/Q8, every cell, PRMB pooled
and within-answer AUC, PRMScore under its existing group-held-out q.8 rule,
coverage, failures, gained/lost exact decisions. The bootstrap does not
account for earlier research choices. No automatic success threshold.

Original6 and all12 trained/untrained score arrays must replay exactly.
Import frozen entropy, Varentropy15/50 and existing fusion references with
metric and ordering replay. Retain historical/Mind-the-Gap scope caveats.
Record selection frequency, seed top6 Jaccard, weights, fitting time and
convergence. Across-seed stability is not perturbation robustness or accuracy.
Use synthetic tests and27 fixed short/median/95th-percentile real answers
for mechanics/runtime only, never for ranking or changing the recipe.
Full fit followed by separate metric arithmetic review and saved-state/
real-token reconstruction. Preserve input hashes and source freeze.

Return results in chat before another experiment. Orders8/10 and graph
regularization remain backlog. No new HTML or deletion of previous results.
