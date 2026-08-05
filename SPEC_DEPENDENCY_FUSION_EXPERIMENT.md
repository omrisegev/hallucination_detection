# Dependency-aware fusion experiment

**Status:** implementation complete; no real-data result exists yet.  The data machine should
run the commands in §12 without changing an algorithm or choosing a hyperparameter from AUROC.

**Research question:** when hallucination-detector views share errors, can we improve label-free
fusion by (a) estimating reliability after separating sparse shared errors, (b) using the estimated
dependency structure in the weights, or (c) learning a nonlinear dependency-removing representation?

This experiment is in the **weights/estimation channel**.  It does not reopen feature ranking,
anti-redundancy pruning, or overlap with a hand-chosen good set; those questions were closed in
Steps 220–224.  The full canonical pool is primary.  The incumbent keep set is a secondary fixed
input arena, not a new selector.

---

## 1. Why this experiment exists

The current U-PCR arm estimates the unobserved reliability vector
`rho_i = Cov(f_i, Y)` from pairwise feature covariance.  Its additive equations are exact when
feature deviations have zero pairwise covariance.  Our data violates that premise: Step 223
measured normalized additive misfit 0.464 and concentrated residual mass in a minority of pairs.
Omri's proposed relaxation was recorded explicitly as

```text
C_ij = rho_i + rho_j - g² + Delta_ij,
```

with sparse `Delta`; see [HANDOFF_FEATURE_SELECTION_AND_FUSE.md](HANDOFF_FEATURE_SELECTION_AND_FUSE.md)
§6.  The first experiment used `Delta` to choose features and failed.  It never changed the
estimator or the final weights, which is the live question here.

The literature audit changes the novelty claim.  The actual Tenzer et al. paper already writes
`C = L + S`, permits sparse correlated errors, and calls the resulting method SU-PCR.  The local
file named `Tenzer2022_Crowdsourcing_Regression_Spectral.pdf` is not that paper: it contains the
older Dror et al. 2017 *Unsupervised Ensemble Regression*, as its digest correctly notes.  Hence
"add a sparse error term" is a required baseline, not our contribution.

The opening is narrower and testable: Tenzer uses the sparse component to clean the estimate of
`rho`, but its final Eq. (15) still projects onto the first two eigenvectors of the observed
covariance.  Our **Sparse Dependency-Structured Fusion (SDSF)** arm holds the decomposition and
`rho` fixed and replaces only that final step with a PSD, condition-controlled structured
covariance solve.  It asks whether shared errors
should affect not only how a feature is judged, but how many votes its evidence receives.

---

## 2. Paper provenance and what is borrowed

### Tenzer et al. 2022 — IU-PCR and SU-PCR

Yaniv Tenzer, Omer Dror, Boaz Nadler, Erhan Bilal, Yuval Kluger,
[*Crowdsourcing Regression: A Spectral Approach*](https://proceedings.mlr.press/v151/tenzer22a/tenzer22a.pdf),
AISTATS 2022.

The paper derives

```text
f_i(x) = g(x) + h_i(x)
C = L + S
L = g² 11ᵀ + a1ᵀ + 1aᵀ        rank(L) <= 2
S_ij = E[h_i h_j].
```

IU-PCR sets off-diagonal `S` to zero.  SU-PCR treats it as sparse and uses a low-rank-plus-sparse
decomposition before recovering `rho`.  Its exact uniqueness theorem requires
`||vec(S)||_0 < (m-1)/2`, which is fewer than 14 correlated pairs at `m=28`; our measured residual
concentration does **not** establish that strict regime.  Every run therefore reports sparse
density and whether the theorem's support condition happens to hold.  Failure of that diagnostic
does not invalidate an approximate method, but it prevents an exact-recovery claim.

The supplement says to apply the projected-gradient robust matrix completion method of
Cherapanamjeri, Gupta and Jain to all observed off-diagonal covariance entries, with threshold
`eta = mu ||M-S_t||_2 / m`.  [Their paper](https://arxiv.org/abs/1606.07315) provides the alternating
rank projection and hard-threshold mechanism.  No authors' U-PCR code was published.  Our output
is therefore named `su_pcr_reproduction`, never `official_su_pcr`.  It is an auditable small-matrix
reproduction of that mechanism and the two parameter rules reported by Tenzer, not a claim of
byte-for-byte equivalence to the authors' unreported PG-RMC implementation.

### Shaham et al. 2016 — the prior RBM-DNN

Uri Shaham et al., *A Deep Learning Approach to Unsupervised Ensemble Learning*,
arXiv:1602.02285; local digest:
[papers/digests/a-deep-learning-approach-to-unsupervised-ensemble-learning.md](papers/digests/a-deep-learning-approach-to-unsupervised-ensemble-learning.md).

It establishes the single-hidden-node RBM/Dawid–Skene equivalence and trains stacked binary RBMs
layer by layer to reduce dependence.  It is not a primary arm because it is binary-only, trained
with an older greedy contrastive-divergence procedure, uses an architecture heuristic, and has no
maintained package appropriate for a one-command reproducible run.  Reimplementing it would add
an implementation-age confound rather than answer the dependency question.

### Maymon et al. 2026 — DEEM

Ariel Maymon, Yanir Buznah, Uri Shaham,
[*Unsupervised Ensemble Learning Through Deep Energy-based Models*](https://arxiv.org/abs/2601.20556),
AISTATS 2026; [official code](https://github.com/shaham-lab/deem).

DEEM is the current nonlinear baseline from the same research line.  It uses an identifiable
multinomial iRBM endpoint and optional learned multinomial preprocessing layers intended to make
dependent learner outputs closer to conditional independence.  Its formal guarantee applies to
the conditionally independent iRBM case; dependency handling by the deep layers remains empirical.

The adapter is pinned to `deem==0.2.0`.  The package's probability output is unaligned, so our code
first obtains DEEM's majority-vote Hungarian class map, applies it to probability columns, and only
then reads the class-1 score.  No correctness label is used for this map.

---

## 3. Algorithms under test

### 3.1 The 2×2 spectral factorial

The cleanest way to locate an improvement is to cross two reliability models with two weight rules:

| arm | reliability estimate | final weight rule | interpretation |
|---|---|---|---|
| `iu_pcr` | independent errors | two-component PCR | published independent baseline |
| `iu_ridge` | independent errors | condition-controlled covariance solve | does ridge alone help? |
| `su_pcr_reproduction` | sparse correlated errors | two-component PCR | published sparse-error mechanism |
| `sdsf` | same sparse fit as SU-PCR | structured condition-controlled solve | proposed contribution |

All four receive the same oriented feature matrix.  `su_pcr_reproduction` and `sdsf` are computed
from one shared fit, so their only difference is the last weight equation.

For SDSF, let `C_structured` contain the recovered rank-two and sparse off-diagonal components and
the observed individual variances.  It is projected to the PSD cone and we solve

```text
w = (C_structured + gamma I)^(-1) rho.
```

`gamma` is not selected by AUROC.  It is the smallest value that caps the system condition number
at 100.  This is a stability rule, not supervised tuning.

### 3.2 DEEM arms

| arm | input | preprocessing | question |
|---|---|---|---|
| `deem_irbm_hard` | per-view median decisions | none | conditionally independent energy endpoint |
| `deem_deep_hard` | per-view median decisions | one sparsemax layer | value of nonlinear dependency processing |
| `deem_deep_soft` | rank pseudo-probabilities `[1-r,r]` | one sparsemax layer | value of retaining continuous ordering |

The soft values are explicitly called **rank pseudo-probabilities**, not calibrated classifier
probabilities.  Each DEEM arm runs seeds 0–4.  Its primary score is the average of the five aligned
probability vectors, formed before labels are read.  Fixed seed 0 and seed variance are also reported
so an apparent gain cannot hide training instability.

### 3.3 Deployed references

- `deployed.upcr_signrho`: exact maintained entry point and configuration from
  `scripts/labelfree_standing_report.py`, including current exclusion.
- `deployed.dufs_lsml`: stored label-free `a2.dufs_pf` choices, rescored through continuous L-SML.

These answer whether a new full system is practically better.  They are not used to attribute the
mechanism because their input sets differ.

---

## 4. Arenas

### `full` — primary

Every spectral arm and DEEM sees the entire canonical pool after the same two-pass `sign(rho)`
relative orientation.  There is no selection or exclusion.  This is the only arena where a new
weights method can recover information outside the incumbent keep set, and it is the primary
scientific comparison.

### `keep` — secondary fixed-input stress test

All arms receive exactly the features retained by the deployed U-PCR run.  No method may change
that set.  This asks whether a fusion improvement survives after the incumbent's selection, but it
cannot measure recovery of signal the keep rule discarded.

### `deployed` — practical references

Each incumbent runs as deployed.  Cross-arena comparisons are clearly prefixed `P*` and must not be
used to claim which internal mechanism caused a difference.

---

## 5. Orientation and information budget

1. Load cells with the canonical `prepare_cell` path and reproduce the GOOD_6 validity constant.
2. Undo the stored hand signs solely to reconstruct the raw relative feature directions.
3. Run the incumbent U-PCR probe and multiply every feature by `sign(rho_hat_i)`.
4. Fit every candidate without labels.
5. Resolve the single globally unidentifiable sign against the existing cell anchor.
6. Freeze the score.
7. Only now pass the score and correctness labels to `roc_auc_score`.

The estimator APIs accept no label argument.  The dataset-free test asserts this seam.  DEEM's
Hungarian alignment is against majority vote of feature predictions, then the same external global
anchor is applied as for the spectral arms.

---

## 6. Registered hypotheses and contrasts

The primary family has exactly three tests.  Holm's family size remains three even if DEEM crashes;
a missing arm must not make the remaining tests easier to declare significant.

| ID | contrast | what a positive result means |
|---|---|---|
| **H1** | `su_pcr_reproduction - iu_pcr` | sparse correction improves reliability estimation |
| **H2** | `sdsf - su_pcr_reproduction` | dependency-aware weights add value beyond published SU-PCR |
| **H3** | `deem_deep_soft_ensemble - iu_pcr` | nonlinear dependency modeling helps on continuous detector ranks |

Pre-registered ablations:

- `A1 = iu_ridge - iu_pcr`: improvement caused by a regularized solve without sparse modeling.
- `A2 = deem_deep_soft - deem_deep_hard`: value of continuous rank information.
- `A3 = deem_deep_hard - deem_irbm_hard`: value of the deep dependency-processing layer.
- `A4 = (sdsf-su_pcr) - (iu_ridge-iu_pcr)`: interaction; whether sparse structure makes the new
  weight rule more useful than ridge alone.
- `K1/K2`: H1/H2 repeated in the fixed keep arena.
- `P1–P3`: practical full-system comparisons against deployed U-PCR and DUFS+L-SML.

---

## 7. Metrics and statistical unit

Primary metric: macro AUROC across the 24 accepted dataset×model cells.

The cells are the primary unit because that is the project's fixed estimand, but they are not 24
independent datasets: GSM8K, MATH500, and TriviaQA occur under several models/method sources.  Every
contrast therefore also aggregates its cell deltas within source dataset and reports an
equal-dataset macro and dataset-bootstrap interval.  This is a pre-registered sensitivity, not a
replacement estimand selected after seeing the answer.

For every arm and contrast report:

- macro AUROC overall, QA, and math;
- paired mean and median delta in percentage points;
- wins/losses/ties by cell;
- paired Wilcoxon signed-rank p-value;
- 95% bootstrap interval over **cells**, not pooled samples;
- fixed-family Holm-adjusted p for H1–H3;
- runtime and failure rate;
- DEEM fixed-seed and between-seed variability;
- sparse support size, residual, convergence, largest dependency edges, and Tenzer-theorem flag.

The bootstrap seed is a stable hash of the contrast name, so rerunning or resuming cannot silently
change an interval.

---

## 8. Decision rules

### What counts as evidence for our contribution

SDSF advances only if **H2**, not merely H1, satisfies all of:

1. mean gain at least **+1.0 AUROC point**;
2. cell-bootstrap 95% lower bound greater than zero;
3. Holm-adjusted paired Wilcoxon `p < 0.05` in the fixed three-test family;
4. neither QA nor math macro delta is below **-0.5 AUROC point**;
5. the equal-dataset macro delta is positive;
6. at least 90% of primary-arena decompositions converge, with failures not concentrated in the
   winning/losing cells.

If H1 wins but H2 does not, the conclusion is "implement published SU-PCR," not "SDSF works."
If A1 explains H2 and A4 is null, the contribution is regularized weighting rather than sparse
dependency-weighted fusion.  If DEEM wins but is unstable, report both the gain and computational
reliability; do not select its best seed.

### Null results that remain informative

- H1 null + high sparse density: exact/approximate sparsity is a poor description.
- H1 positive, H2 null: dependency correction belongs in `rho`, not the final weights.
- DEEM hard null, soft positive: binarization discarded the gain.
- DEEM deep null relative to iRBM: extra nonlinear dependency processing did not help.
- SDSF positive only in `keep`: regularization helps but no discarded signal was recovered.

---

## 9. No tuning after labels

Registered fixed values:

- covariance/label scale ratio: 0.25, matching the incumbent comparison;
- low-rank rank: 2;
- projected sparse threshold multiplier: 1.0;
- no global sparse-support cap;
- PCR components: 2;
- g2 projection components: 1;
- structured solve target condition: 100;
- DEEM 0.2.0; 100 epochs; batch up to 1024; learning rate 0.001; momentum 0.9;
- DEEM seeds: 0,1,2,3,4;
- deep DEEM: one identity-initialized sparsemax preprocessing layer;
- weighted initialization enabled, AutoML hyperparameters disabled.

Any later sensitivity analysis gets a new output directory and is explicitly secondary.  Do not
replace the primary row with whichever sensitivity value has the best AUROC.

---

## 10. Implementation map

- `spectral_utils/dependency_fusion.py`
  - off-diagonal rank-two plus sparse projected decomposition;
  - SU-PCR `rho` recovery and PCR weights;
  - PSD/condition-controlled structured weights for SDSF.
- `spectral_utils/deem_adapter.py`
  - continuous-to-hard/soft transforms;
  - pinned DEEM configuration;
  - majority-vote class-map correction for continuous probabilities.
- `scripts/test_dependency_fusion.py`
  - planted support, low-rank recovery, clean-world, condition cap, transform, and label-seam tests.
- `scripts/run_dependency_fusion_experiment.py`
  - canonical data loading and validity gate;
  - full/keep/deployed arms;
  - arm/seed checkpointing and resume;
  - paired statistics and generated result report.

Generated under `results/dependency_fusion_study/`:

- `run_config.json` — immutable configuration hash;
- `records.jsonl` — append-only score/diagnostic checkpoints;
- `per_cell.csv`, `arm_summary.csv`, `contrasts.csv`;
- `deem_seeds.csv`, `deem_seed_summary.csv` — every seed and per-cell instability;
- `sparse_diagnostics.csv` — support size, residual, convergence, and theorem diagnostic;
- `summary.json`;
- `REPORT.md` — human-readable final table and gates.

---

## 11. Required data layout

Copy the complete cache directory to

```text
/path/to/hallucination_detection/local_cache/
```

including `derived_views.pkl`, `trace_cells.pkl`, and every source cache consumed by
`spectral_utils.subset_sweep.iter_cells`.  Partial CSV result tables are insufficient because the
new algorithms need the per-sample feature matrix.

---

## 12. Exact runbook for the data machine

From repository root on branch `master`:

```bash
python3 -m pip install -e ".[dependency-experiment]"
python3 scripts/test_dependency_fusion.py
python3 scripts/run_dependency_fusion_experiment.py \
  --data-dir local_cache \
  --device auto
```

Operational one-cell check before the complete run, without changing the registered method:

```bash
python3 scripts/run_dependency_fusion_experiment.py \
  --data-dir local_cache \
  --device auto \
  --max-cells 1
```

The complete command safely resumes the same output directory afterward.  To run spectral arms
first on a CPU machine:

```bash
python3 scripts/run_dependency_fusion_experiment.py \
  --data-dir local_cache \
  --skip-deem
```

Then rerun the complete command on the GPU machine; successful spectral checkpoints are reused and
only missing DEEM arm/seeds execute.  Never edit `records.jsonl` by hand.  A true configuration
change must use a new `--out-dir`; the runner refuses to mix configuration hashes.

No experiment command stages or commits repository files.
