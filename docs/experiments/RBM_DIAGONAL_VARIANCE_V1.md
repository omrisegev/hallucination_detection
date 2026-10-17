# Conditional diagonal variance in the compact RBM

User authorized first auditing variance/covariance assumptions and, if the
restriction is material, testing learned shared diagonal conditional variance
with regularization and a floor. Dedicated codex/rbm-diagonal-variance-v1,
parent50d9ef17a. This is the original SIX-feature order3 bank only; no order
or feature selection, no DUFS, no labels in fitting, no global24 sweep.

## First: frozen original-model audit

Read original6 states from the completed higher-moment checkpoint and rebuild
same-answer standardized X. Report observed covariance C and model covariance
D+pi(1-pi)ww', plus mean discrepancy, diagonal RMSE, full/offdiagonal relative
Frobenius errors and eigenvalues of C-D. Original D=I while diag(C)=1 by
construction. These are descriptive misspecification diagnostics, not p-values
or proof of the cause of localization errors. No labels or refits in the audit.
Record all13769 answers and summary quantiles. Retain original files.

## One fixed candidate

Exact Gaussian RBM with one binary hidden unit:
x|h ~ N(a+h*w, D), with D diagonal and shared across h=0,1.
Posterior sigmoid(b+x'(w/D)); pi=sigmoid(b+sum((a*w+.5*w*w)/D)).
Fit exact normalized mean NLL +0.1*sum(log(D_j)^2).
Fixed lower bound D_j>=0.05 (variance, NOT standard deviation); no upper bound.
The penalty pulls conditional variance toward1, preserving the original
model as its reference. Coefficient0.1 and floor0.05 are declared engineering
choices, not optimized or established optima. No coefficient/floor sweep.
The partition function changes with D; do not only rescale the output scores.

Common original6mean orientation, normalization, constant-column handling,
top15 q convention, selected-token surprisal, external raw-entropy q0.3 gate,
Top10 token mean and first ties, corrected v3 labels/source folds preserved.

Arms:
- original RBM100 iterations (exact historical replay).
- original RBM before learning (exact historical replay).
- original RBM continued from its saved fitted parameters for100 additional
  iterations, still D=I: matched additional optimization control.
- diagonal RBM initialized from EXACTLY the same original fitted parameters
  and D=I,100 additional iterations with the new objective and bounds.
Thus continued vsdiagonal isolates the model/objective change without granting
only the new method another optimizer budget. No hidden retries/fallbacks.
All finite nonconverged fits retained and flagged; projected gradient and
floor hits recorded because constrained optima may have nonzero raw gradient.

Full13769 model-answer rows only for findings. Primary diagonal-vs-continued;
10000 canonical-source bootstrap draws,97.5% intervals on PB macro8 and PRMB
within-answer AUC. Diagonal-vs-original/initial, continued-vs-original, and
Varentropy/entropy contrasts exploratory95%. Include Q4/Q8, every cell,
PRMScore/pooled AUC, coverage/failures, matched mean/covariance diagnostics,
and explicit historical comparator scope. Retain higher-order RBM and weight
shrinkage as saved matched reference rows; no refitting those candidates.
No new winner claim from likelihood/covariance improvements alone.

Validation: analytic gradients against finite differences, D=I objective and
parameter-gradient bridge, independent two-Gaussian-mixture density, posterior
coefficient w/D and bound checks, original score replay,27-answer mechanical
smoke, separate full metric arithmetic verification and state audit.
Audit and experiment use immutable source/input signatures and isolated WAL
checkpoints. Chat-first, no HTML. Freeze before full candidate execution.

## Literature scope clarification requested by user

Shaham et al., ICML2016, A Deep Learning Approach to Unsupervised Ensemble
Learning, Lemma4.1 (https://proceedings.mlr.press/v48/shaham16.pdf) proves a
bijection between binary conditionally independent Dawid-Skene and a binary
RBM with one hidden unit. It does not prove that our continuous Gaussian RBM
or implemented L-SML/IU optimizer is equivalent. Historical Step141 wording
is too broad. Stacked RBM dependence reduction is an empirical finding, not
a universal guarantee. This experiment tests continuous-model assumptions.
