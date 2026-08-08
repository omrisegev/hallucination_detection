# Frozen 24-cell unsupervised fusion experiment

## Goal

Run one apples-to-apples factorial comparison on the canonical 24 dataset/model cells.
Every method receives the same samples, the same `fixed_stable_v1` feature pool,
and the same fixed confidence direction. Labels are used only after every score
has been fitted, saved, and hashed.

This experiment answers a development question:

> Does a DUFS- or SpecRaGE-derived sample graph improve the final IU-PCR ranking,
> and is local reliability better represented by manual families, individual
> features, or fusion-aware micro-views?

It does not answer external generalization because these cells contributed to
earlier feature and method development.

## Terms and output metrics

- A **cell** is one dataset/model pair.
- **AUROC** measures correct-versus-incorrect ranking. 0.5 is random and 1.0 is
  perfect.
- **AUPRC** measures precision-recall quality and is needed for cells with very
  few correct or incorrect examples.
- A **percentage point (pp)** is 0.01 AUROC. For example, 0.750 to 0.755 is
  +0.5 pp.
- A **cell-macro** mean gives each cell equal weight.
- A **family-macro** mean prevents a dataset family with several models from
  dominating the result.
- A **paired bootstrap interval** resamples comparison units while keeping both
  methods from the same cell together.

## Registered roster

The runner imports `scripts/inscope_cells.py` and fails unless the bundle has
exactly 24 cells: 9 QA and 15 math. `inside_coqa_llama7b` remains excluded for
the documented generation defect; it is not removed because of performance.

There is no row-level train/validation split. Every method is unsupervised in
the 24 cells. There is also no label-based hyperparameter selection in this
experiment. The full lambda path is a sensitivity plot, not a selection rule.

## Headline methods

1. Deployed U-PCR.
2. Full-pool, two-component IU-PCR.
3. DUFS-LIU at the registered `lambda=0.1`.
4. CA-SpecRaGE-alpha-LIU with manual provenance views at `lambda=10`.
5. CA-SpecRaGE-alpha-LIU with duplicate-balanced atomic views at `lambda=10`.
6. CA-SpecRaGE-alpha-LIU with leave-one-cell-out micro-views at `lambda=10`.

For every view schema, secondary controls are adapted plain-loss Y, CA-trained
Y, prior-only uniform Y, the exact prior-alpha graph made from the CA base
graphs, global alpha, and a node-permuted alpha graph. The prior-alpha control
is the direct test of whether learned sample-specific alpha helped. A raw
uniform-feature graph is also included. All graph methods are evaluated on the
fixed path `0, 0.1, 0.3, 1, 3, 10, 30, 100`.

## Registered view definitions

- **Manual:** the existing semantic provenance families.
- **Atomic:** one feature per view. Features assigned to the same LOCO
  micro-cluster divide one cluster's prior mass.
- **Micro:** groups learned from pairwise, basis-invariant distances between
  normalized projected roughness matrices (U^\top R_jU).

For each held cell, micro groups are learned from the other 23 cells. Candidate
cluster counts 3--8 are selected without labels by a fixed combination of
silhouette, bootstrap adjusted-Rand stability, singleton fraction, and group-size
imbalance. The runner stores every candidate score and partition before fitting
SpecRaGE.

## Why lambda 10 remains the SpecRaGE headline

The synthetic mechanism study selected 10 before real labels were opened. A
later ten-cell execution pilot showed that 10 transfers poorly and that 0.1 may
look better. Changing the headline to 0.1 after that observation would turn the
24 cells into another tuning set. The report therefore:

- uses 10 for the SpecRaGE headline;
- uses 0.1 for the already frozen DUFS-LIU baseline;
- plots every registered lambda without selecting a real-data winner;
- may use the path to diagnose a mechanism, but not to claim a tuned gain.

## Two-stage leakage barrier

### Stage 1: fit and freeze scores

`scripts/frozen_24cell_benchmark.py` never reads the `__labels` arrays. It fits
each method, writes one score file and diagnostic file per cell, and records a
SHA-256 hash for each file. It checkpoints after every cell and supports resume.
Resume accepts a cell only if every registered score arm is present and no
label-like array exists. The run definition also records Python, NumPy, SciPy,
scikit-learn, PyTorch, and Matplotlib versions.

Some graph eigensolvers can fail to estimate a diagnostic such as algebraic
connectivity even when score fitting succeeds. Such a value is stored as JSON
`null`, never as zero. Each cell records the exact affected keys in
`nonfinite_diagnostic_paths`, so unavailable diagnostics remain visible to the
reviewer and cannot be mistaken for favorable measurements.

### Stage 2: evaluate

`scripts/frozen_24cell_report.py` verifies all 24 score hashes and exclusively
creates `SCORE_FREEZE_MANIFEST.json`. If that immutable file already exists,
the report requires an exact match instead of replacing it. Only then does it
read labels and compute metrics. It refuses debug or incomplete runs.

This barrier prevents accidental label use inside method fitting. It cannot
erase historical reuse of the same cells during earlier development.

## Installation on the data computer

From the repository root on branch `master`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[benchmark]"
```

The benchmark is CPU-only. A GPU is not required.

For an execution-only timing probe, `--debug-cell CELL` fits one named cell and
marks the output as non-scientific. The report refuses to evaluate it.

## Pre-run checks

```bash
git branch --show-current
git status --short
python scripts/test_feature_contract.py
python scripts/test_specrage_laplacian.py
python scripts/test_specrage_real_selection.py
python scripts/test_frozen_24cell_benchmark.py
python scripts/test_fusion_aware_views.py
```

The branch should be `master`. Tests must pass. The input bundle expected by
default is `results/dependency_fusion_raw/cells.npz`.

## Full run

```bash
python scripts/frozen_24cell_benchmark.py --resume
python scripts/frozen_24cell_report.py
```

`--resume` reuses a cell only when both its score and diagnostic checkpoint
exist. If the registered configuration or source hashes changed, use a new
`--out-dir`; do not mix results from two definitions.

The default output is `results/frozen_24cell_benchmark/`.

## Expected tables

1. **Headline summary:** cell-macro AUROC and AUPRC, 95% intervals, QA macro,
   math macro, and family macro.
2. **Paired comparisons:** mean/median AUROC change, cell and family bootstrap
   intervals, wins/ties/losses, worst case, Wilcoxon test, and Holm correction.
3. **Per-cell table:** AUROC, AUPRC, prevalence, and score variance for every
   method.
4. **Lambda paths:** all/QA/math macro, change versus IU-PCR, wins/losses, and
   worst change at every fixed lambda.
5. **View construction:** LOCO partitions, selected cluster count, silhouette,
   adjusted-Rand stability, singleton rate, group balance, graph overlap,
   effective rank, and projected-roughness distance.
6. **Diagnostics:** feature/view count, exclusions, DUFS effective features,
   alpha entropy and stability, graph health, projected condition, rank
   displacement, and runtime.

The Wilcoxon p-value is descriptive. The 24 cells are heterogeneous and reused
development data; a small p-value does not create external validation.

## Expected figures

- headline AUROC with cell-bootstrap intervals;
- per-cell paired changes versus IU-PCR;
- cell-by-method AUROC heatmap;
- fixed lambda paths for all, QA, and math;
- CA view-weight entropy and rank actuation versus performance change;
- manual/atomic/micro comparison and micro-partition stability;
- self-supervised training convergence;
- runtime versus sample count.

## Predeclared CA-SpecRaGE promotion gates

All must pass before asking for new-data confirmation:

1. at least +0.5 pp mean AUROC versus deployed U-PCR;
2. at least +0.5 pp versus IU-PCR;
3. at least +0.5 pp versus DUFS-LIU;
4. family-bootstrap lower bound versus IU-PCR above zero;
5. improvement versus IU-PCR in at least 14 of 24 cells;
6. worst loss versus IU-PCR no worse than -2 pp.
7. LOCO micro-views improve over manual views.
8. LOCO micro-views do not lose to balanced atomic views.

Passing means “worth confirming,” not “proved.” Failure means keep the method as
a mechanism result or revise it; do not choose a different lambda from this same
report and relabel that as confirmation.

## Independent second review

Give the reviewer the full repository state and specifically:

- `docs/methods/`;
- `RUN_DEFINITION.json`;
- `FIT_COMPLETE.json` and `SCORE_FREEZE_MANIFEST.json`;
- score checkpoints, diagnostics, CSV files, plots, and `REPORT.md`;
- `REVIEWER_GUIDE.md` generated beside the report.

Ask the reviewer to recompute metrics from raw scores, verify hashes and the
zero-lambda identity, inspect failures and controls, and state where our
interpretation is stronger than the evidence. Do not provide only the generated
summary or ask whether it “agrees.”

## Interpretation boundary

The manual families are only a baseline. Atomic views test whether reliability
is feature-local, while LOCO micro-views test whether stable fusion behavior is
a better grouping principle than semantic origin. If micro-views win only at a
real-data-selected lambda, that is a development clue rather than a confirmed
method. If their partitions are unstable, any AUROC gain must be treated as
fragile even when its mean is positive.
