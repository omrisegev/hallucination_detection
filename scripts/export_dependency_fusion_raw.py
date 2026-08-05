#!/usr/bin/env python3
"""Hand the reviewing model the raw data and diagnostics it asked for.

It reviewed commit 64f57cd from the repo alone and listed six things it needed, plus one
follow-up diagnostic.  This script produces everything on that list that does not require the
DEEM sweep to have finished, and it is SAFE TO RUN WHILE THE SWEEP IS STILL RUNNING - it reads
only complete JSONL lines and tolerates a torn trailing line from a concurrent append.

Their request                                            -> delivered as
  1. regenerated REPORT/summary/per_cell/arm_summary/         PENDING: the runner rewrites these
     contrasts                                                itself when the sweep finishes
  2. deem_seeds.csv, deem_seed_summary.csv                 -> same names + the runner's exact
                                                              schema, interim contents
  3. every DEEM failure: cell, seed, error, score          -> deem_all_attempts.csv +
     variance, training-history diagnostics                   records.jsonl.gz  (see CAVEAT)
  4. final failures AND earlier failed attempts that       -> deem_all_attempts.csv, one row per
     later succeeded                                          append, with a `superseded` flag
  5. sample size n for every contrast                      -> contrast_sample_sizes.csv
  6. compressed records.jsonl.gz                           -> verbatim, complete lines, gzip -9
  follow-up: fixed-SU-rho 2x2 + per-cell solver            -> already run, committed separately in
     diagnostics                                              results/dependency_fusion_solver_matrix/

Beyond their list, because without it a repo-only machine still cannot check anything:
  cells.npz - the per-cell inputs every arm consumes, float64 verbatim, so the whole experiment
  is re-derivable off-machine and bit-exactly.  This review turned on agreement at 1e-9, so an
  approximate export would not have been good enough.

CAVEAT on item 3: the runner's failure path DISCARDS the training history.  `save_method_record`
catches the exception raised by `orient_score` *after* `score_fn()` has already built DEEM's
history dict, and the except branch stores only `error_type`, `error` and `traceback`.  So for
exactly the collapsed fits, the loss curve does not exist in the checkpoint and cannot be
exported.  What identifies a collapse is `score_std` / `score_n_unique` on the surviving seeds
plus the error string on the failing ones.  Fixing it means retaining `dynamic_diag` on failure.

Contains derived spectral features and correctness labels only.  No prompts, no generated text,
no model weights - the same scope as the result CSVs the repo already tracks.

Usage:
    python scripts/export_dependency_fusion_raw.py --data-dir local_cache
"""

import argparse
import csv
import gzip
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from inscope_cells import GROUP, INSCOPE                              # noqa: E402
from spectral_utils.subset_sweep import ALL_SIGNS, GOOD_6             # noqa: E402
from run_dependency_fusion_experiment import (                        # noqa: E402
    derive_oriented_matrix, load_cells,
)

STUDY = os.path.join(REPO, "results", "dependency_fusion_study")


# --------------------------------------------------------------------------- cells
def export_cells(data_dir, out_dir):
    cells = load_cells(os.path.abspath(data_dir))
    payload, manifest = {}, []
    for ck in [c for c in INSCOPE if c in cells]:
        cell = cells[ck]
        F, polarity, _ = derive_oriented_matrix(cell)
        pool = list(cell["pool"])
        payload[f"{ck}__V"] = np.asarray(cell["V"], dtype=np.float64)
        payload[f"{ck}__F"] = np.asarray(F, dtype=np.float64)
        payload[f"{ck}__labels"] = np.asarray(cell["labels"], dtype=np.int8)
        payload[f"{ck}__anchor"] = np.asarray(cell["anchor"], dtype=np.float64)
        payload[f"{ck}__pool"] = np.array(pool, dtype=object)
        payload[f"{ck}__hand_signs"] = np.array(
            [ALL_SIGNS.get(name, +1) for name in pool], dtype=np.int8)
        payload[f"{ck}__rho_polarity"] = np.asarray(polarity, dtype=np.int8)
        labels = np.asarray(cell["labels"], dtype=int)
        manifest.append({
            "cell": ck, "domain": GROUP.get(ck), "n": int(len(labels)), "m": len(pool),
            "n_positive": int(labels.sum()),
            "positive_rate": round(float(labels.mean()), 6),
            "good6_present": sum(1 for f in GOOD_6 if f in pool),
            "anchor_name": cells[ck]["unlabeled"].anchor_name,
        })
        print(f"  {ck:34s} V{payload[f'{ck}__V'].shape} F{payload[f'{ck}__F'].shape}", flush=True)

    npz = os.path.join(out_dir, "cells.npz")
    np.savez_compressed(npz, **payload)
    write_csv(os.path.join(out_dir, "cells_manifest.csv"), manifest)
    return len(manifest), os.path.getsize(npz)


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------- records
def read_complete_lines(path):
    """Every complete JSONL line. Tolerates a torn trailing line from a live append."""
    good, torn = [], 0
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.endswith("\n") or not line.strip():
                torn += 1 if line.strip() else 0
                continue
            try:
                good.append((line, json.loads(line)))
            except json.JSONDecodeError:
                torn += 1
    return good, torn


def export_records(out_dir):
    src = os.path.join(STUDY, "records.jsonl")
    lines, torn = read_complete_lines(src)

    # item 6 - verbatim, complete lines only so their parser cannot choke
    dst = os.path.join(out_dir, "records.jsonl.gz")
    with gzip.open(dst, "wt", encoding="utf-8", compresslevel=9) as fout:
        for raw, _ in lines:
            fout.write(raw)

    deem = [(i, rec) for i, (_, rec) in enumerate(lines) if rec.get("seed") is not None]
    # item 4 - which appends were later superseded by a retry of the same key
    last_index = {}
    for i, rec in deem:
        last_index[(rec["cell"], rec["arena"], rec["arm"], rec["seed"])] = i

    attempts, seeds_rows = [], []
    for i, rec in deem:
        key = (rec["cell"], rec["arena"], rec["arm"], rec["seed"])
        row = {
            "cell": rec["cell"], "domain": rec.get("domain"), "arena": rec["arena"],
            "arm": rec["arm"], "seed": rec["seed"], "status": rec["status"],
            "superseded": int(last_index[key] != i),
            "attempt_order": i,
            "runtime_seconds": round(float(rec.get("runtime_seconds", float("nan"))), 3),
            "auc": rec.get("auc", ""),
            "error_type": rec.get("error_type", ""), "error": rec.get("error", ""),
            "score_std": "", "score_min": "", "score_max": "", "score_n_unique": "",
            "global_anchor_flip": rec.get("global_anchor_flip", ""),
            "class_map": "", "history_keys": "", "history": "",
            "package_version": "",
        }
        if rec["status"] == "ok":
            score = np.asarray(rec["score"], dtype=float)
            row["score_std"] = f"{np.std(score):.6e}"
            row["score_min"] = f"{np.min(score):.6e}"
            row["score_max"] = f"{np.max(score):.6e}"
            row["score_n_unique"] = int(len(np.unique(score)))
            diag = rec.get("diagnostics", {}) or {}
            hist = diag.get("history") or {}
            row["class_map"] = json.dumps(diag.get("class_map", {}))
            row["history_keys"] = ",".join(sorted(hist.keys()))
            row["history"] = json.dumps(hist)
            row["package_version"] = diag.get("package_version", "")
            # item 2 - the runner's own deem_seeds.csv schema, latest attempt only
            if last_index[key] == i:
                seeds_rows.append({
                    "cell": rec["cell"], "domain": rec.get("domain"),
                    "arena": rec["arena"], "arm": rec["arm"], "seed": rec["seed"],
                    "auc": rec["auc"], "runtime_seconds": rec.get("runtime_seconds"),
                })
        attempts.append(row)

    write_csv(os.path.join(out_dir, "deem_all_attempts.csv"), attempts)
    write_csv(os.path.join(out_dir, "deem_seeds.csv"), seeds_rows)

    # item 2 - the runner's own deem_seed_summary.csv schema
    groups = {}
    for row in seeds_rows:
        groups.setdefault(
            (row["cell"], row["domain"], row["arena"], row["arm"]), []
        ).append(float(row["auc"]))
    summary = []
    for (cell, domain, arena, arm), vals in sorted(groups.items()):
        summary.append({
            "cell": cell, "domain": domain, "arena": arena, "arm": arm,
            "n_seeds": len(vals), "mean_auc": float(np.mean(vals)),
            "std_auc": float(np.std(vals)), "min_auc": float(np.min(vals)),
            "max_auc": float(np.max(vals)),
            "range_auc": float(np.max(vals) - np.min(vals)),
        })
    write_csv(os.path.join(out_dir, "deem_seed_summary.csv"), summary)

    # the completion-coverage table their H3 objection actually needs
    coverage = {}
    for i, rec in deem:
        key = (rec["cell"], rec["arena"], rec["arm"], rec["seed"])
        if last_index[key] != i:
            continue
        c = coverage.setdefault((rec["cell"], rec["arena"], rec["arm"]),
                                {"ok": 0, "failed": 0})
        c["ok" if rec["status"] == "ok" else "failed"] += 1
    cov_rows = [{"cell": k[0], "arena": k[1], "arm": k[2], "seeds_ok": v["ok"],
                 "seeds_failed": v["failed"],
                 "ensemble_eligible": int(v["failed"] == 0 and v["ok"] >= 5)}
                for k, v in sorted(coverage.items())]
    write_csv(os.path.join(out_dir, "deem_completion_coverage.csv"), cov_rows)

    return len(attempts), len(seeds_rows), torn, os.path.getsize(dst), os.path.getsize(src)


# --------------------------------------------------------------------------- item 5
def export_contrast_n(out_dir):
    """n per contrast, which REPORT.md's table omits."""
    src = os.path.join(STUDY, "contrasts.csv")
    if not os.path.exists(src):
        return 0
    rows = []
    with open(src, newline="", encoding="utf-8") as handle:
        for r in csv.DictReader(handle):
            rows.append({"contrast": r["contrast"], "reference": r["reference"],
                         "candidate": r["candidate"], "primary": r["primary"],
                         "n_cells": r["n"], "mean_delta_pp": r["mean_delta"],
                         "n_dataset_families": r.get("n_dataset_families", "")})
    write_csv(os.path.join(out_dir, "contrast_sample_sizes.csv"), rows)
    return len(rows)


README = """# Raw-data + diagnostics handover — dependency-aware fusion experiment

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
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.path.join(REPO, "local_cache"))
    parser.add_argument("--out-dir",
                        default=os.path.join(REPO, "results", "dependency_fusion_raw"))
    parser.add_argument("--skip-cells", action="store_true",
                        help="re-export only the DEEM tables and records")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    n_cells = npz_bytes = 0
    if not args.skip_cells:
        n_cells, npz_bytes = export_cells(args.data_dir, args.out_dir)
    n_att, n_seeds, torn, gz_bytes, raw_bytes = export_records(args.out_dir)
    n_contrast = export_contrast_n(args.out_dir)
    with open(os.path.join(args.out_dir, "RAW_DATA_README.md"), "w", encoding="utf-8") as handle:
        handle.write(README)

    def mb(b):
        return b / 1e6

    print()
    if n_cells:
        print(f"cells.npz                      {n_cells} cells, {mb(npz_bytes):.1f} MB")
    print(f"records.jsonl.gz               {mb(gz_bytes):.1f} MB "
          f"(from {mb(raw_bytes):.1f} MB, {100 * gz_bytes / max(raw_bytes, 1):.0f}%)")
    print(f"deem_all_attempts.csv          {n_att} appends"
          f"{'' if not torn else f' ({torn} torn line(s) skipped)'}")
    print(f"deem_seeds.csv                 {n_seeds} successful latest-attempt fits")
    print(f"contrast_sample_sizes.csv      {n_contrast} contrasts")
    print(f"\nwrote {args.out_dir}")
    if torn:
        print(f"NOTE: {torn} incomplete trailing line(s) skipped - the sweep is still appending.")


if __name__ == "__main__":
    main()
