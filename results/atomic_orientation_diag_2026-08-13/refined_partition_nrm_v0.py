#!/usr/bin/env python3
"""Refined-partition NRM v0 — retrospective evaluation.

Candidate: refine the 6 provenance families along the label-free b-coupling
sign axis (each family splits into its positive-gamma3 and negative-gamma3
halves, if both are nonempty), then run the NRM machinery at the refined
granularity with the retained mode SELECTED and SIGNED by the pooled group-
level gamma3 witness instead of the all-ones anchor.

Constraints respected: gray-box (mixed-v2 telemetry only), one-pass at
inference (per-cell transductive IU + frozen calibration direction),
unsupervised (no correctness label enters partitioning, calibration,
selection, sign, or trust), built on IU-PCR.

Controls run through the SAME code:
  - family NRM reproduction (provenance partition, argmin|lambda-1|,
    all-ones sign) -- must reproduce +0.277/+0.557/+1.580 to validate the
    reimplementation;
  - no-band variant of the candidate (select among all modes by |gamma3|
    alignment) as a pre-stated secondary readout.

Labels are read ONLY for AUROC readouts.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
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
from spectral_utils.specrage_views import FEATURE_TO_VIEW  # noqa: E402

OUT = SCRATCH / "refined_partition_nrm_v0"
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
EPS = 1e-12
NULL_DRAWS = 500
NULL_ALPHA = 0.05
NULL_SEED = 20260813


def log(msg=""):
    print(msg, flush=True)


def standardize(x):
    m, s = float(np.mean(x)), float(np.std(x))
    return (x - m) / (s if s > EPS else 1.0), s


def residual_column(h, b):
    """standardize -> remove linear b component -> standardize; None if dead."""
    hz, s = standardize(h)
    if s <= EPS:
        return None
    beta = float(hz @ b / max(b @ b, EPS))
    r = hz - beta * b
    rz, s2 = standardize(r)
    if s2 <= EPS:
        return None
    return rz


def hermite3(col, b):
    phi = b ** 3 - 3.0 * b
    phi -= float(phi @ b / (b @ b)) * b
    phi -= float(np.mean(phi))
    return -float(col @ phi) / len(b)


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
    F = np.asarray(F, float)
    y = np.asarray(y, dtype=int)
    w = upcr_fit(F, **IU_FIT_DEFAULTS).w
    b_raw = w @ F
    b, _ = standardize(b_raw)
    # per-atom residuals + per-atom gamma3 (for the partition rule)
    atom_res = {}
    atom_g3 = {}
    for i, nm in enumerate(names):
        col = residual_column(w[i] * F[i], b)
        if col is not None:
            atom_res[nm] = col
            atom_g3[nm] = hermite3(col, b)
    return {
        "cell": name, "group": group, "domain": domain,
        "F": F, "names": tuple(names), "w": w, "b": b, "y": y,
        "n": int(len(y)), "atom_g3": atom_g3,
    }


def pooled_atom_g3(cells):
    """n-weighted pooled per-atom cubic coupling over cells where active."""
    total, weight = {}, {}
    for c in cells:
        for nm, v in c["atom_g3"].items():
            total[nm] = total.get(nm, 0.0) + c["n"] * v
            weight[nm] = weight.get(nm, 0) + c["n"]
    return {nm: total[nm] / weight[nm] for nm in total}


def refined_partition(g3_atoms):
    """Split each provenance family into its +/- pooled-gamma3 halves."""
    mapping = {}
    fams = sorted({FEATURE_TO_VIEW[nm] for nm in g3_atoms})
    for fam in fams:
        members = [nm for nm in g3_atoms if FEATURE_TO_VIEW[nm] == fam]
        pos = [nm for nm in members if g3_atoms[nm] > 0]
        neg = [nm for nm in members if g3_atoms[nm] <= 0]
        if pos and neg:
            for nm in pos:
                mapping[nm] = fam + "+"
            for nm in neg:
                mapping[nm] = fam + "-"
        else:
            for nm in members:
                mapping[nm] = fam
    return mapping


def family_partition(atom_names):
    return {nm: FEATURE_TO_VIEW[nm] for nm in atom_names}


def group_residuals(cell, mapping, group_order):
    """Group contribution residual columns for one cell (None if absent)."""
    cols = {}
    for g in group_order:
        idx = [i for i, nm in enumerate(cell["names"])
               if mapping.get(nm) == g]
        if not idx:
            continue
        h = cell["w"][idx] @ cell["F"][idx]
        col = residual_column(h, cell["b"])
        if col is not None:
            cols[g] = col
    return cols


def pairwise_cov(cell_cols, group_order, rng_children=None):
    G = len(group_order)
    cov = np.zeros((G, G))
    cnt = np.zeros((G, G), dtype=int)
    for k, cols in enumerate(cell_cols):
        present = [g for g in group_order if g in cols]
        li = [group_order.index(g) for g in present]
        V = np.column_stack([cols[g] for g in present])
        if rng_children is not None:
            rng = rng_children[k]
            V = np.column_stack([
                V[rng.permutation(len(V)), j] for j in range(V.shape[1])
            ])
        local = V.T @ V / len(V)
        cov[np.ix_(li, li)] += local
        cnt[np.ix_(li, li)] += 1
    if np.any(cnt == 0):
        raise ValueError("pair coverage gap: " + str(
            [(group_order[i], group_order[j]) for i, j in np.argwhere(cnt == 0)]
        ))
    cov = cov / cnt
    return 0.5 * (cov + cov.T), cnt


def calibrate(cells, mapping, *, selection, seed=NULL_SEED):
    """Label-free calibration. selection: 'g3_band', 'g3_all', 'family_nrm'."""
    group_order = sorted(set(mapping.values()))
    cell_cols = [group_residuals(c, mapping, group_order) for c in cells]
    cov, cnt = pairwise_cov(cell_cols, group_order)
    vals, vecs = np.linalg.eigh(cov)

    # permutation null band (simultaneous min/max), family-NRM skips it
    lo, hi = None, None
    if selection != "family_nrm":
        root = np.random.SeedSequence(seed)
        seqs = root.spawn(NULL_DRAWS)
        mins = np.empty(NULL_DRAWS)
        maxs = np.empty(NULL_DRAWS)
        for d, sq in enumerate(seqs):
            children = [np.random.default_rng(ch)
                        for ch in sq.spawn(len(cell_cols))]
            ncov, _ = pairwise_cov(cell_cols, group_order, children)
            ne = np.linalg.eigvalsh(ncov)
            mins[d], maxs[d] = ne[0], ne[-1]
        lo = float(np.quantile(mins, NULL_ALPHA / 2))
        hi = float(np.quantile(maxs, 1 - NULL_ALPHA / 2))

    # group-level gamma3 witness, n-weighted pooled
    g3_grp = np.zeros(len(group_order))
    g3_wt = np.zeros(len(group_order))
    for c, cols in zip(cells, cell_cols):
        for j, g in enumerate(group_order):
            if g in cols:
                g3_grp[j] += c["n"] * hermite3(cols[g], c["b"])
                g3_wt[j] += c["n"]
    g3_grp = g3_grp / np.maximum(g3_wt, 1)
    g3_unit = g3_grp / max(np.linalg.norm(g3_grp), EPS)

    info = {"group_order": group_order, "eigenvalues": vals.tolist(),
            "band": [lo, hi], "g3_group": g3_grp.tolist()}
    if selection == "family_nrm":
        j = int(np.argmin(np.abs(vals - 1.0)))
        v = vecs[:, j]
        sign = 1.0 if float(np.sum(v)) >= 0 else -1.0
        direction = sign * v
        info.update({"selected": j, "rule": "argmin|l-1|, all-ones sign"})
    else:
        if selection == "g3_band":
            cand = [j for j in range(len(vals)) if lo <= vals[j] <= hi]
            if not cand:
                cand = [int(np.argmin(np.abs(vals - 1.0)))]
        else:
            cand = list(range(len(vals)))
        scores = [abs(float(vecs[:, j] @ g3_unit)) for j in cand]
        j = cand[int(np.argmax(scores))]
        v = vecs[:, j]
        dot = float(v @ g3_unit)
        direction = (1.0 if dot >= 0 else -1.0) * v
        info.update({
            "selected": j, "rule": selection,
            "candidates": cand,
            "witness_alignment": abs(dot),
            "per_mode_alignment": [
                round(abs(float(vecs[:, k] @ g3_unit)), 3)
                for k in range(len(vals))
            ],
        })
    return {"mapping": mapping, "group_order": group_order,
            "direction": direction, "info": info}


def score_cell(cell, cal):
    cols = group_residuals(cell, cal["mapping"], cal["group_order"])
    present = [g for g in cal["group_order"] if g in cols]
    if len(present) < 2:
        return 0.0, float(roc_auc_score(cell["y"], cell["b"]))
    li = [cal["group_order"].index(g) for g in present]
    d = np.asarray(cal["direction"])[li]
    R = np.column_stack([cols[g] for g in present])
    q = R @ d
    sd = float(np.std(q))
    iu = float(roc_auc_score(cell["y"], cell["b"]))
    if sd <= EPS or float(np.linalg.norm(d)) <= EPS:
        return 0.0, iu
    s = cell["b"] + (q / sd) * (1.0 / len(present))
    return 100.0 * (float(roc_auc_score(cell["y"], s)) - iu), iu


def equal_group(rows, key):
    groups = sorted({r["group"] for r in rows})
    per_group = [
        float(np.mean([r[key] for r in rows if r["group"] == g]))
        for g in groups
    ]
    deltas = np.asarray([r[key] for r in rows])
    return (float(np.mean(per_group)), int((deltas > 0).sum()),
            int((deltas < 0).sum()), float(deltas.min()))


def main():
    t0 = time.time()
    log("loading originals...")
    original = []
    with np.load(BUNDLE, allow_pickle=True) as data:
        for name in SOURCE_CELLS:
            F, names = load_contract(data, name, "mixed_v2")
            y = np.asarray(data[f"{name}__labels"], dtype=int)
            original.append(make_cell(
                name, original_family(name), "original_23", F, names, y
            ))
    log(f"{len(original)} cells, {sum(c['n'] for c in original)} samples")

    log("")
    log("=== frozen (all-23) calibration: partition + diagnostics ===")
    g3_all = pooled_atom_g3(original)
    part_refined = refined_partition(g3_all)
    groups = sorted(set(part_refined.values()))
    log(f"refined partition ({len(groups)} groups):")
    for g in groups:
        members = sorted(nm for nm, gg in part_refined.items() if gg == g)
        log(f"  {g:>22}: {', '.join(members)}")

    METHODS = {
        "family_nrm": (family_partition(g3_all), "family_nrm"),
        "refined_g3_band": (part_refined, "g3_band"),
        "refined_g3_all": (part_refined, "g3_all"),
    }

    log("")
    log("=== originals: leave-one-dataset-family-out ===")
    rows = []
    fold_partitions = {}
    for c in original:
        row = {"cell": c["cell"], "group": c["group"]}
        src = [cc for cc in original if cc["group"] != c["group"]]
        g3_src = pooled_atom_g3(src)
        part_src = refined_partition(g3_src)
        fold_partitions[c["group"]] = part_src
        for mname, (mapping, sel) in METHODS.items():
            fold_mapping = part_src if mname.startswith("refined") else mapping
            cal = calibrate(src, fold_mapping, selection=sel)
            row[mname], row["iu_auroc"] = score_cell(c, cal)
        rows.append(row)
        log(f"  scored {c['cell']}")

    log("")
    log("originals LOFO, equal-group delta vs IU (pp):")
    for mname in METHODS:
        eg, w_, l_, worst = equal_group(rows, mname)
        log(f"  {mname:>18}: {eg:+.3f}pp  W/L {w_}/{l_}  worst {worst:+.2f}")
    log("  (published family NRM reference: +0.277pp)")

    # partition stability across folds
    base = set(part_refined.items())
    log("")
    log("partition stability across LOFO folds (vs all-23 partition):")
    for grp, part in sorted(fold_partitions.items()):
        diff = len(set(part.items()) ^ base) // 2
        log(f"  fold -{grp:<12}: {diff} atom reassignments")

    log("")
    log("=== frozen all-23 calibrations (for transfer) ===")
    cals = {}
    for mname, (mapping, sel) in METHODS.items():
        cals[mname] = calibrate(original, mapping, selection=sel)
        info = cals[mname]["info"]
        vals = np.asarray(info["eigenvalues"])
        log(f"{mname}: G={len(info['group_order'])}  "
            f"selected lambda={vals[info['selected']]:.4f}  band={info['band']}")
        if "witness_alignment" in info:
            log(f"   witness |cos|={info['witness_alignment']:.3f}  "
                f"per-mode: {info['per_mode_alignment']}")
        log(f"   direction: " + ", ".join(
            f"{g}:{d:+.3f}" for g, d in zip(
                info["group_order"], cals[mname]["direction"])
        ))

    log("")
    log("=== ProcessBench transfer ===")
    ext_rows = []
    for model in PROCESS_MODELS:
        for subset in PROCESS_SUBSETS:
            path = (REAL / "dataset_cache" / "repgrid" / f"pb_{model}"
                    / f"processbench_{subset}.pkl")
            items = process_items(path)
            telemetry = [telemetry_only(r) for _, r in items]
            y = [int(r["label"] == -1) for _, r in items]
            F, names, _, _ = mixed_v2_matrix(telemetry)
            c = make_cell(f"{model}__{subset}", subset,
                          "processbench_qwen", F, names, y)
            row = {"cell": c["cell"], "group": c["group"],
                   "domain": c["domain"]}
            for mname in METHODS:
                row[mname], _ = score_cell(c, cals[mname])
            ext_rows.append(row)
            log(f"  scored {c['cell']}")
    root = REAL / "dataset_cache" / "repgrid" / "pb_llama31_8b"
    for subset in PROCESS_SUBSETS:
        items = process_items(root / f"processbench_{subset}.pkl")
        telemetry = [telemetry_only(r) for _, r in items]
        y = [int(r["label"] == -1) for _, r in items]
        F, names, _, _ = mixed_v2_matrix(telemetry)
        c = make_cell(f"llama31_8b__{subset}", subset,
                      "processbench_llama", F, names, y)
        row = {"cell": c["cell"], "group": c["group"], "domain": c["domain"]}
        for mname in METHODS:
            row[mname], _ = score_cell(c, cals[mname])
        ext_rows.append(row)
        log(f"  scored {c['cell']}")

    log("")
    for domain, ref in (("processbench_qwen", "+0.557"),
                        ("processbench_llama", "+1.580")):
        sel_rows = [r for r in ext_rows if r["domain"] == domain]
        log(f"{domain} equal-group delta vs IU (pp)  "
            f"[published family NRM: {ref}]:")
        for mname in METHODS:
            eg, w_, l_, worst = equal_group(sel_rows, mname)
            log(f"  {mname:>18}: {eg:+.3f}pp  W/L {w_}/{l_}  worst {worst:+.2f}")

    payload = {
        "refined_partition_all23": part_refined,
        "pooled_atom_g3": {k: float(v) for k, v in g3_all.items()},
        "lofo_rows": rows,
        "transfer_rows": ext_rows,
        "calibrations": {m: cals[m]["info"] for m in cals},
    }
    with (OUT / "RESULT.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, default=float)
    log("")
    log(f"done in {time.time() - t0:.0f}s -> {OUT / 'RESULT.json'}")


if __name__ == "__main__":
    main()
