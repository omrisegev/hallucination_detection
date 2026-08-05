# Raw-data + diagnostics handover — dependency-aware fusion experiment

Written by `scripts/export_dependency_fusion_raw.py` for the reviewing model that checked commit
`64f57cd` from the repo alone. Everything here is so that no claim in
`OPINION_DEPENDENCY_FUSION_RUN.md` has to be taken on trust.

**The DEEM sweep was still running when this was exported.** It asked for the current checkpoint to
be finished and pushed, so the run was left going; these DEEM tables are therefore a **snapshot**,
and the runner regenerates its own final versions when the sweep completes.

## Their list, item by item

| # | requested | here |
|---|---|---|
| 1 | regenerated `REPORT.md`, `summary.json`, `per_cell.csv`, `arm_summary.csv`, `contrasts.csv` | **pending the sweep** — the runner rewrites these itself; the committed ones are the spectral-only snapshot |
| 2 | `deem_seeds.csv`, `deem_seed_summary.csv` | present, **the runner's exact schema**, interim contents |
| 3 | every DEEM failure: cell, seed, error, score variance, training-history diagnostics | `deem_all_attempts.csv` + `records.jsonl.gz`. **See the caveat below — history is not recoverable for failures.** |
| 4 | final failures *and* earlier failed attempts that later succeeded | `deem_all_attempts.csv` — one row per append, with `superseded` and `attempt_order` |
| 5 | sample size `n` for every contrast | `contrast_sample_sizes.csv` (the runner's `REPORT.md` omits it; `contrasts.csv` always had it) |
| 6 | compressed `records.jsonl.gz` | present, gzip -9, verbatim, complete lines only |
| + | the fixed-SU-ρ 2×2 and its per-cell solver diagnostics | **already run** — `results/dependency_fusion_solver_matrix/` |

Added because their list cannot be checked without it:

| file | what it lets you do |
|---|---|
| `cells.npz` | re-derive **every arm from scratch, bit-exactly** — per cell: `V`, the oriented `F` every arm consumes, hand signs, derived polarity, anchor, labels, pool names |
| `cells_manifest.csv` | n, m, positive rate, GOOD_6 coverage, anchor name per cell |
| `deem_completion_coverage.csv` | per cell/arm: `seeds_ok`, `seeds_failed`, `ensemble_eligible` — the completion-coverage criterion their H3 objection asks for |

### CAVEAT on item 3 — training history for failures does not exist

The runner's failure path discards it. `save_method_record` catches the exception raised by
`orient_score` **after** `score_fn()` has already built DEEM's history dict, and the `except` branch
stores only `error_type`, `error` and `traceback`. So for exactly the collapsed fits the loss curve
is computed and thrown away. It is not in the checkpoint and cannot be exported.

What *does* identify a collapse: `score_std` and `score_n_unique` on the surviving seeds, plus the
error string `ValueError: method returned a non-finite or constant score`, raised when
`orient_score` sees `np.std(score) < 1e-12`. Full `history` JSON **is** exported for every
successful fit. Retaining `dynamic_diag` on failure is the change to make before DEEM is re-run.

## Array conventions in `cells.npz`

- `V` is `(n_samples, m_features)` — canonical `prepare_cell` output, z-scored over
  `CANONICAL_POOL`, with `ALL_SIGNS` **already applied**.
- `F` is `(m_features, n_samples)` — **transposed**, and this is what every arm consumes. It equals
  `(V * hand_signs * rho_polarity).T`: the hand signs are undone and replaced by the data-derived
  `sign(rho_hat)` polarity from the incumbent probe.
- `pool[j]` names column `j` of `V` / row `j` of `F`.
- `labels` is `int8`, 1 = correct. AUROC is raw, never `max(a, 1-a)`.
- `V`, `F`, `anchor` are float64 verbatim — this review turned on agreement at 1e−9.

## Reproduce the validity constant and two registered arms

```python
import numpy as np
from sklearn.metrics import roc_auc_score
from spectral_utils.fusion_utils import lsml_continuous
from spectral_utils.streaming_utils import anchor_orient
from spectral_utils.subset_sweep import GOOD_6
from spectral_utils.dependency_fusion import sparse_upcr_fit

d = np.load("cells.npz", allow_pickle=True)
cells = sorted({k.split("__")[0] for k in d.files})

SPARSE_FIT = dict(scale_ratio=0.25, rank=2, n_components=2, g2_projection_components=1,
                  threshold_multiplier=1.0, max_iter=100, inner_completion_iter=40,
                  decomposition_tol=1e-8, max_sparse_fraction=None, target_condition=100.0)

good6, su, sdsf = [], [], []
for ck in cells:
    V, F = d[f"{ck}__V"], d[f"{ck}__F"]
    y, anchor, pool = d[f"{ck}__labels"], d[f"{ck}__anchor"], list(d[f"{ck}__pool"])

    cols = [pool.index(f) for f in GOOD_6 if f in pool]          # validity anchor
    fused, _ = lsml_continuous(*[V[:, c] for c in sorted(set(cols))])
    good6.append(roc_auc_score(y, anchor_orient(np.asarray(fused, float), anchor)[0]))

    fit = sparse_upcr_fit(F, **SPARSE_FIT)                        # both arms, one fit
    for w, sink in ((fit.w_pcr, su), (fit.w_structured, sdsf)):
        sink.append(roc_auc_score(y, anchor_orient(w @ F, anchor)[0]))

print("GOOD_6 macro         ", round(float(np.mean(good6)), 7), "-> must be 0.7733442")
print("su_pcr_reproduction  ", round(float(np.mean(su)),    4), "-> must be 0.7668")
print("sdsf                 ", round(float(np.mean(sdsf)),  4), "-> must be 0.7104")
```

If GOOD_6 is not **0.7733442**, this bundle is not the data the committed numbers came from and
nothing downstream should be trusted. That is the project's standing rule and it applies to the
machine that produced this bundle as much as to anyone reading it.

## Audit the DEEM collapse yourself

```python
import gzip, json, collections, numpy as np
tally = collections.Counter()
for line in gzip.open("records.jsonl.gz", "rt", encoding="utf-8"):
    r = json.loads(line)
    if r.get("seed") is None:
        continue
    tally[(r["arm"], r["status"])] += 1
    if r["status"] == "ok" and r["arm"] == "deem_deep_soft":
        s = np.asarray(r["score"], float)
        print(r["cell"], "seed", r["seed"], "std", np.std(s), "unique", len(np.unique(s)))
print(tally)
```

`deem_completion_coverage.csv` answers the ensemble question directly: `collect_scores` builds a
seed ensemble only when **all five** registered seeds succeeded, so any cell with
`ensemble_eligible = 0` is silently absent from that arm's paired contrast rather than scored as a
loss.
