#!/usr/bin/env python3
"""Addendum 2: trust-scale sweep for transported atomic directions.

Diagnostic only: the sweep itself is NOT a tuned method.  It measures the
gain = alignment*trust - dilution*trust^2 mechanism, and evaluates one
pre-stated label-free trust rule (self-consistency: mean pairwise cosine of
per-cell gamma3 estimates across source cells).

Directions evaluated (originals: LOFO pooling; pb: all-23 pooling):
  fisher  (supervised reference), g3 (label-free), codex frozen direction.
Trust values: 1/p, 1/6, 1/sqrt(p), plus c_selfcal (from g3 agreement).
Also saves per-cell arrays to npz for future iteration.
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np

SCRATCH = Path(__file__).resolve().parent
MW = SCRATCH / "mw"
REAL = Path(r"c:/Users/omris/TAU/hallucination_detection")
sys.path.insert(0, str(MW))

from sklearn.metrics import roc_auc_score  # noqa: E402

from scripts.hard_filter_dufs_liu_benchmark import (  # noqa: E402
    load_contract,
    family as original_family,
)
from scripts.leverage_balanced_processbench_transfer import (  # noqa: E402
    mixed_v2_matrix,
    resolve_data_path,
)
from scripts.atomic_nrm_structural_audit import SOURCE_CELLS  # noqa: E402
from spectral_utils.upcr import upcr_fit  # noqa: E402
from spectral_utils.laplacian_upcr import IU_FIT_DEFAULTS  # noqa: E402
from spectral_utils.atomic_neutral_residual import (  # noqa: E402
    atomic_contribution_space,
    fit_atomic_neutral_calibration,
)
from spectral_utils.contribution_subspace import (  # noqa: E402
    fit_contribution_transform,
)

OUT = SCRATCH / "atomic_orientation_diag"
OUT.mkdir(exist_ok=True)
BUNDLE = REAL / "results" / "dependency_fusion_raw" / "cells.npz"
TELEMETRY_KEYS = (
    "token_entropies",
    "token_spilled_energies",
    "token_logsumexp",
    "top_k_logprobs",
)
PROCESS_MODELS = ("qwen3_4b", "qwen3_8b")
PROCESS_SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")


def log(msg=""):
    print(msg, flush=True)


def normalize(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def cosv(u, v):
    du, dv = np.linalg.norm(u), np.linalg.norm(v)
    return float(u @ v / (du * dv)) if du > 0 and dv > 0 else float("nan")


def telemetry_only(row):
    return {name: row.get(name) for name in TELEMETRY_KEYS}


def process_items(path):
    with Path(resolve_data_path(Path(path))).open("rb") as handle:
        cache = pickle.load(handle)
    return [
        (str(key), cache[key])
        for key in sorted(cache)
        if not cache[key]["align_diag"]["problems"]
    ]


def make_cell(name, group, domain, F, names, y):
    y = np.asarray(y, dtype=int)
    w = upcr_fit(np.asarray(F, float), **IU_FIT_DEFAULTS).w
    aspace = atomic_contribution_space(F, names, w)
    transform = fit_contribution_transform(
        aspace, np.arange(F.shape[1], dtype=int)
    )
    b, res = transform.apply(aspace.baseline_score, aspace.contributions)
    return {
        "cell": name, "group": group, "domain": domain,
        "names": tuple(names), "b": b, "R_full": res, "y": y,
        "n": int(len(y)), "aspace": aspace,
    }


def restrict(c, atoms):
    lookup = {n: i for i, n in enumerate(c["names"])}
    return c["R_full"][:, [lookup[n] for n in atoms]]


def hermite3(R, b):
    phi = b ** 3 - 3.0 * b
    phi -= float(phi @ b / (b @ b)) * b
    phi -= float(np.mean(phi))
    return -(R.T @ phi) / len(b)


def fisher(R, y):
    return R[y == 1].mean(axis=0) - R[y == 0].mean(axis=0)


def score(c, d, trust):
    q = c["R"] @ d
    sd = float(np.std(q))
    if sd <= 1e-12:
        return 0.0
    s = c["b"] + (q / sd) * trust
    iu = float(roc_auc_score(c["y"], c["b"]))
    return 100.0 * (float(roc_auc_score(c["y"], s)) - iu)


def equal_group(rows, key):
    groups = sorted({r["group"] for r in rows})
    per_group = [
        float(np.mean([r[key] for r in rows if r["group"] == g]))
        for g in groups
    ]
    deltas = np.asarray([r[key] for r in rows])
    return (float(np.mean(per_group)), int((deltas > 0).sum()),
            int((deltas < 0).sum()))


log("loading originals...")
original = []
with np.load(BUNDLE, allow_pickle=True) as data:
    for name in SOURCE_CELLS:
        F, names = load_contract(data, name, "mixed_v2")
        y = np.asarray(data[f"{name}__labels"], dtype=int)
        original.append(make_cell(
            name, original_family(name), "original_23", F, names, y
        ))

log("calibrating (needed for codex direction)...")
cal = fit_atomic_neutral_calibration([c["aspace"] for c in original])
atoms = cal.feature_names
p = len(atoms)
for c in original:
    c["R"] = restrict(c, atoms)
    c["fisher"] = fisher(c["R"], c["y"])
    c["g3"] = hermite3(c["R"], c["b"])

# label-free self-consistency of g3 across source cells
G = np.stack([normalize(c["g3"]) for c in original])
gram = G @ G.T
off = gram[np.triu_indices(len(original), k=1)]
c_selfcal = float(max(np.mean(off), 0.0))
log(f"g3 self-consistency (mean pairwise cos): {c_selfcal:+.4f}")

TRUSTS = {
    "1/p": 1.0 / p,
    "1/6": 1.0 / 6.0,
    "1/sqrt(p)": 1.0 / np.sqrt(p),
    "selfcal": c_selfcal,
}

log("")
log("=== originals, LOFO pooling ===")
rows = []
for c in original:
    row = {"cell": c["cell"], "group": c["group"]}
    src = [cc for cc in original if cc["group"] != c["group"]]
    d_f = normalize(sum(cc["n"] * cc["fisher"] for cc in src))
    d_g = normalize(sum(cc["n"] * cc["g3"] for cc in src))
    for tname, t in TRUSTS.items():
        row[f"fisher@{tname}"] = score(c, d_f, t)
        row[f"g3@{tname}"] = score(c, d_g, t)
        row[f"codex@{tname}"] = score(c, cal.direction, t)
    rows.append(row)
for m in sorted(rows[0]) - {"cell", "group"} if False else [
    k for k in rows[0] if k not in ("cell", "group")
]:
    eg, w, l = equal_group(rows, m)
    log(f"  {m:>18}: {eg:+.3f}pp  W/L {w}/{l}")

sweep = {"original": rows, "c_selfcal": c_selfcal}

log("")
log("=== ProcessBench transfer, all-23 pooling ===")
d_f_all = normalize(sum(c["n"] * c["fisher"] for c in original))
d_g_all = normalize(sum(c["n"] * c["g3"] for c in original))
ext_rows = []
for model in PROCESS_MODELS:
    for subset in PROCESS_SUBSETS:
        path = (REAL / "dataset_cache" / "repgrid" / f"pb_{model}"
                / f"processbench_{subset}.pkl")
        items = process_items(path)
        telemetry = [telemetry_only(r) for _, r in items]
        y = [int(r["label"] == -1) for _, r in items]
        F, names, _, _ = mixed_v2_matrix(telemetry)
        c = make_cell(f"{model}__{subset}", subset, "processbench_qwen",
                      F, names, y)
        c["R"] = restrict(c, atoms)
        row = {"cell": c["cell"], "group": c["group"], "domain": c["domain"]}
        for tname, t in TRUSTS.items():
            row[f"fisher@{tname}"] = score(c, d_f_all, t)
            row[f"g3@{tname}"] = score(c, d_g_all, t)
            row[f"codex@{tname}"] = score(c, cal.direction, t)
        row["g3trans@1/6"] = score(c, hermite3(c["R"], c["b"]), 1.0 / 6.0)
        ext_rows.append(row)
        log(f"  loaded+scored {c['cell']}")
root = REAL / "dataset_cache" / "repgrid" / "pb_llama31_8b"
for subset in PROCESS_SUBSETS:
    items = process_items(root / f"processbench_{subset}.pkl")
    telemetry = [telemetry_only(r) for _, r in items]
    y = [int(r["label"] == -1) for _, r in items]
    F, names, _, _ = mixed_v2_matrix(telemetry)
    c = make_cell(f"llama31_8b__{subset}", subset, "processbench_llama",
                  F, names, y)
    c["R"] = restrict(c, atoms)
    row = {"cell": c["cell"], "group": c["group"], "domain": c["domain"]}
    for tname, t in TRUSTS.items():
        row[f"fisher@{tname}"] = score(c, d_f_all, t)
        row[f"g3@{tname}"] = score(c, d_g_all, t)
        row[f"codex@{tname}"] = score(c, cal.direction, t)
    row["g3trans@1/6"] = score(c, hermite3(c["R"], c["b"]), 1.0 / 6.0)
    ext_rows.append(row)
    log(f"  loaded+scored {c['cell']}")

log("")
for domain in ("processbench_qwen", "processbench_llama"):
    sel = [r for r in ext_rows if r["domain"] == domain]
    log(f"{domain}:")
    for m in [k for k in sel[0] if k not in ("cell", "group", "domain")]:
        eg, w, l = equal_group(sel, m)
        log(f"  {m:>18}: {eg:+.3f}pp  W/L {w}/{l}")
sweep["processbench"] = ext_rows

with (OUT / "TRUST_SWEEP.json").open("w", encoding="utf-8") as handle:
    json.dump(sweep, handle, indent=1, default=float)
log("")
log("saved TRUST_SWEEP.json")
