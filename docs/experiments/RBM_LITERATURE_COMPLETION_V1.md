# Complete the remaining literature-linked RBM experiments

Authorized continuation, 2026-09-12. Base de237a3622c776f2cfd866b39e0b0de3ca30fe90.
Dedicated code-only worktree/branch codex/rbm-literature-completion-v1.
User authorization supersedes the older stop-after-discussion instructions.

## Requirements and scope

1. Storage: preserve unique data, code, and results. The failed literature
   checkout's 104 SHA256-identical data copies have been removed (11.0705 GiB).
   Audit stays at source scratch/cleanup_20260912/verified_duplicates.jsonl.
2. Complete original DUFS6-of12 + RBM, unchanged frozen pipeline and review.
   It resumed from 9848 contiguous validated SQLite rows, PID18724. Its result
   may be incorporated only after full completion and review; no partial ranking.
3. Complete state-variance, capacity, CD, multi-start, second-layer, and token
   chronology tests below. No new moments, q, window-width or lambda search.
4. Correct the latest literature diagnostic's answer-bootstrap and old-readout
   attribution, preserve its source result, and maintain the full comparator ledger.
5. Return chat findings, CSV/JSON and short HISTORY/PROGRESS/Research_Directions
   updates. No HTML. Stop short of claiming untouched confirmation or a guaranteed
   winner. A negative mechanism test still counts as a completed experiment.

Both existing banks are used: original6 H15,V15,m3,a,a^2,a^3 and bank12 adding
m4,a^4,m5,a^5,m6,a^6. The cached answer-local normalized matrices are read from
rbm-supervision-matched-v1, and original states from higher-moment-fusion-v1.
The new fits receive NO labels, target positions, fold indices or other answers.
UID seeds are fixed. Original mean6 risk orientation remains the common anchor.
Baseline replay and source hashes bind caches, states, spans, identities and scores.

## Registered experiments

### V: shared versus state-specific variance

Two Gaussian components, initialized by the exact mapping from each saved
one-hidden RBM: mu0=a, mu1=a+w, mixture logit=b+a.w+||w||^2/2.
Both arms refit means and mixture mass with L-BFGS-B maxiter100, ftol1e-10,
gtol1e-6. Shared log-variance v has penalty .1||v||^2. Separate v0,v1 has
.1||mean(v0,v1)||^2+.1||v1-v0||^2. Both variances have floor .05 and no ceiling.
At equality their likelihood and penalty agree. These are unnamed latent
states, not known correctness classes. Score is their log density ratio plus
mixture log odds, oriented by correlation of its sigmoid with mean6.
This is a Gaussian mixture extension, not a claim of Bernoulli-RBM equivalence.

### C: exact versus CD, one versus four hidden units

E(x,h)=||x-a||^2/2-b.h-x.W.h, fixed unit conditional Gaussian variance.
Enumerate all 2^H states for exact likelihood, H=1 or4. Same cold initialization
for exact/CD: a=b=0; W=2/(P sqrt(H)), H4 gets fixed-seed N(0,.02^2) perturbations.
H1 exact replays the historical algorithm; maxiter100 and original tolerances.
CD uses real Bernoulli hidden and Gaussian visible samples: CD10,100 epochs,
batch128, learning rate .005/sqrt(1+epoch/20), gradient norm capped at5 with
counts reported. No epoch/rate selection from NLL or labels. Final epoch scored;
exact NLL/gradient are diagnostics, not a checkpoint selector.

Each hidden unit is oriented by its posterior's correlation with original mean6.
Fixed fusion is mean oriented hidden posterior. Its stable logit is computed
with logsumexp, avoiding inversion of rounded probabilities. H1 reduces to the
original oriented logit. Report posterior and logit Top10 separately: this does
not introduce another fit. Capacity comparison uses the same fixed readout.

### S: identifiable predictions and multiple starts

After C, fit two further exact starts for H1/H4 using seed1/2 perturbations.
Keep all start diagnostics, NLL, risk correlations, peak agreement and task
changes. The candidate chooses the lowest exact answer-only NLL among three
starts. No label/benchmark selection of a restart; no claim of global optimum.

### D: a second learned fusion layer

Use the SAME fixed H4 exact representation from C. Its four oriented posterior
streams are continuous visibles, answer-standardized, for a one-hidden Gaussian
RBM trained exact or CD10 under the same settings. Compare to fixed H4 mean
under both score conventions. This is layer-wise pretraining, NOT joint deep
likelihood optimization or a reproduction of Bernoulli RBMpaper. Original mean6
orients the final latent unit. No label-driven first-layer model selection.

### T: token chronology with the original RBM held fixed

Keep saved one-hidden RBM Gaussian emission parameters and risk orientation.
Fit a two-state Markov transition matrix within the answer with exact
forward/backward EM, max25 iterations, change tolerance1e-6. Initial transitions
are independent rows equal to the RBM prior; that model replays the RBM token
posteriors. Fixed Dirichlet(2,2) row priors supply one count per transition.
The latent prior at sequence starts stays the saved RBM prior. Offline smoothed
log odds and sigmoid are scored; no online-detection claim.

Three variants: actual full token order; reset at each distinct step start;
full within-answer random token permutation followed by inverse mapping before
unchanged step aggregation. No adjacency across answers. Overlapping original
step spans remain unchanged; distinct starts define reset boundaries without
duplicating tokens in the fitted chain. Record transition matrices, expected
pair counts, boundary effects, early/late moves and hidden correct peaks.
This tests chronological fusion of the RBM's evidence. It is distinct from the
failed two-half weight adaptation and from step-mean residual correlation.

## Evaluation and acceptance

Full13769 matched answers, existing v3 labels and canonical source groups/folds.
Original entropy q.3 gate thresholds unchanged. Top10 token mean and earliest
argmax, NOT first_near_max. Both posterior/logit are explicit readout comparisons.
PRMScore q.8 thresholds are recomputed on other-fold answer scores, no test groups.
Failures remain in PB denominators, invalid PRMB coverage and full-versus-
conditional PRMScore are disclosed. No hidden fallback for a failed method.

PB8 macro/Q4/Q8/per-cell, clean accuracy, raw exact, early/late, suppressed peaks.
PRMB within-answer and pooled AUC, PRMScore, all denominators. Reference table
includes both banks trained/initial, IU contribution fusion, equal contributions,
Entropy, Varentropy15/50, shrinkage and shared-diagonal models. Older near-max
rows stay in the separate historical registry; they are not adopted for RBM.
Do not silently omit B3 or other completed historical variants from the ledger.

10000 paired source-group bootstrap draws. Each mechanism's two bank contrasts
use97.5% intervals (V separate/shared; C exact4/exact1; S best/seed0; D learned/fixed;
T actual/shuffled); all additional contrasts descriptive95%. These are not a
family-wide correction for the entire research program or all prior selections.
No arbitrary effect-size promotion threshold. Statistical mismatch alone is
not proof of task improvement. Full cached results remain development evidence.

Before full scoring: finite-difference gradients, H1 likelihood equivalence,
enumerated Gaussian-mixture check for H4, actual Gibbs moment check, short/constant
inputs, exact brute-force Markov marginals/counts and boundary-reset identities.
Smoke answers check mechanics/runtime only. Review all saved-state score replays,
baseline metrics, independent PB/AUC arithmetic, calibration and failures.

## Literature scope

Shaham et al., ICML2016: https://proceedings.mlr.press/v48/shaham16.html
Author implementation: https://github.com/ushaham/RBMpaper
Their binary Dawid-Skene equivalence is not a theorem for our continuous moments.
The inspiration is modeling dependent inputs with multiple latent units/layers.
CoNAL/common-noise, instance-dependent reliability, and sequential/networked
ensemble papers motivate these measured-mismatch tests, not claimed reproductions
of those complete methods. Prior supervised and position diagnostics stay visible.
