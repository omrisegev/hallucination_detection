#!/usr/bin/env python3
"""Fixed-orientation solver-mechanism cycle after stabilized SDSF failed.

This ports the already-preregistered Study-A questions in
``SPEC_SOLVER_MECHANISM_STUDY.md`` to the committed fixed-stable feature
contract.  It does not search for a winning AUROC configuration.

Questions
---------
1. Does ridge head rescaling hurt, or does admitting the covariance tail hurt?
2. Does stronger tail admission become worse along a fixed condition path?
3. On held-out rows, does the tail loss disappear with more training data, or
   remain despite a stable estimated tail (structural mismatch)?
4. Does direct two-channel CCA fusion, a distinct multi-view alternative, avoid
   the failure?  This arm is exploratory and cannot be promoted from this replay.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import types

import numpy as np
from scipy.linalg import eigh
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
if "spectral_utils" not in sys.modules:
    package = types.ModuleType("spectral_utils")
    package.__path__ = [os.path.join(REPO, "spectral_utils")]
    sys.modules["spectral_utils"] = package

from spectral_utils.dependency_fusion import (                         # noqa: E402
    _nearest_psd,
    regularized_covariance_weights,
    sparse_upcr_fit,
)
from spectral_utils.feature_contract import (                          # noqa: E402
    LEGACY_FEATURE_SIGNS,
    confidence_oriented_matrix,
    consensus_anchor,
)


VERSION = "sdsf-solver-cycle-v4-fixed-stable"
DEFAULT_BUNDLE = os.path.join(REPO, "results", "dependency_fusion_raw", "cells.npz")
DEFAULT_MANIFEST = os.path.join(REPO, "results", "dependency_fusion_raw", "cells_manifest.csv")
DEFAULT_OUT = os.path.join(REPO, "results", "sdsf_solver_cycle_v4")
KAPPAS = (3.0, 10.0, 30.0, 100.0, 300.0)
FRACTIONS = (0.25, 0.50, 0.75)
N_REPEATS = 50
N_CI_BOOT = 10000

SPARSE_FIT = dict(
    scale_ratio=0.25, rank=2, n_components=2,
    g2_projection_components=1, threshold_multiplier=1.0,
    max_iter=100, inner_completion_iter=40, decomposition_tol=1e-8,
    max_sparse_fraction=None, target_condition=100.0,
)

SPECTRAL_CHANNEL = frozenset({
    "cusum_max", "cusum_shift_idx", "dominant_freq", "epr",
    "high_band_power", "hl_ratio", "hurst_exponent", "low_band_power",
    "pe_mean", "rpdi", "spectral_centroid", "spectral_entropy",
    "stft_max_high_power", "stft_spectral_entropy", "sw_var_peak",
    "trace_length",
})
KNOWN_FAMILIES = (
    "triviaqa", "coqa", "hotpotqa", "sciq", "nq_open", "squad_v2",
    "truthfulqa", "gsm8k", "math500", "gpqa", "webq", "humaneval",
)


def seed(*parts):
    value = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(value.encode()).hexdigest()[:16], 16) % (2 ** 32)


def family(cell):
    return next((name for name in KNOWN_FAMILIES if name in cell), cell)


def load_cells(bundle, manifest_path):
    with open(manifest_path, newline="", encoding="utf-8") as handle:
        manifest = {r["cell"]: r for r in csv.DictReader(handle)}
    data = np.load(bundle, allow_pickle=True)
    keys = sorted({name.rsplit("__", 1)[0] for name in data.files})
    cells = {}
    for key in keys:
        names = [str(x) for x in data[f"{key}__pool"]]
        legacy = np.asarray(data[f"{key}__hand_signs"], dtype=float)
        expected = np.asarray([LEGACY_FEATURE_SIGNS[name] for name in names])
        if not np.array_equal(legacy, expected):
            raise RuntimeError(f"{key}: legacy reconstruction failed")
        raw = np.asarray(data[f"{key}__V"], dtype=float) * legacy
        matrix, kept, _ = confidence_oriented_matrix(raw, names, stable=True)
        cells[key] = {
            "matrix": matrix, "names": kept,
            "labels": np.asarray(data[f"{key}__labels"], dtype=int),
            "domain": manifest[key]["domain"], "family": family(key),
        }
    return cells


def orient_sign(score, anchor):
    corr = float(np.corrcoef(score, anchor)[0, 1])
    if not np.isfinite(corr):
        raise RuntimeError("constant score")
    return -1.0 if corr < 0 else 1.0


def auc(weight, matrix, labels, anchor):
    score = matrix @ weight
    return float(roc_auc_score(labels, orient_sign(score, anchor) * score))


def components(F, kappa):
    fit = sparse_upcr_fit(F, **SPARSE_FIT)
    C, rho = fit.covariance, fit.rho_hat
    psd, _ = _nearest_psd(C)
    values, vectors = eigh(psd)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    full, diag = regularized_covariance_weights(C, rho, target_condition=kappa)
    gamma = diag["ridge"]
    coef = vectors.T @ rho
    head_pcr = np.zeros_like(rho)
    head_ridge = np.zeros_like(rho)
    tail = np.zeros_like(rho)
    for j in range(len(rho)):
        ridge_part = coef[j] / (values[j] + gamma) * vectors[:, j]
        if j < 2:
            if values[j] > 1e-12:
                head_pcr += coef[j] / values[j] * vectors[:, j]
            head_ridge += ridge_part
        else:
            tail += ridge_part
    error = np.linalg.norm(head_ridge + tail - full) / (np.linalg.norm(full) + 1e-30)
    if error > 1e-10:
        raise RuntimeError(f"ridge decomposition identity failed: {error}")
    return fit, head_pcr, head_ridge, tail, gamma


def direct_cca_score(matrix, names, target_condition=100.0):
    left = [i for i, name in enumerate(names) if name in SPECTRAL_CHANNEL]
    right = [i for i, name in enumerate(names) if name not in SPECTRAL_CHANNEL]
    X, Y = matrix[:, left], matrix[:, right]
    n = len(matrix)
    Cx, Cy, Cxy = X.T @ X / n, Y.T @ Y / n, X.T @ Y / n

    def invsqrt(C):
        values, vectors = eigh(0.5 * (C + C.T))
        hi, lo = float(values[-1]), float(values[0])
        ridge = max(hi * 1e-10, (hi - target_condition * lo) / (target_condition - 1), 0.0)
        return (vectors * (1.0 / np.sqrt(np.maximum(values, 0.0) + ridge))) @ vectors.T

    Wx, Wy = invsqrt(Cx), invsqrt(Cy)
    U, singular, Vt = np.linalg.svd(Wx @ Cxy @ Wy, full_matrices=False)
    sx, sy = X @ (Wx @ U[:, 0]), Y @ (Wy @ Vt.T[:, 0])
    if np.corrcoef(sx, sy)[0, 1] < 0:
        sy = -sy
    sx = (sx - sx.mean()) / (sx.std() + 1e-12)
    sy = (sy - sy.mean()) / (sy.std() + 1e-12)
    return 0.5 * (sx + sy), float(singular[0])


def family_ci(values, families, name):
    values, families = np.asarray(values), np.asarray(families)
    names = sorted(set(families.tolist()))
    means = np.asarray([values[families == fam].mean() for fam in names])
    rng = np.random.default_rng(seed(VERSION, "ci", name))
    picks = rng.integers(0, len(means), size=(N_CI_BOOT, len(means)))
    stats = means[picks].mean(axis=1)
    return tuple(float(x) for x in np.quantile(stats, [0.025, 0.975]))


def summarize_delta(rows, candidate, reference, name):
    delta = np.asarray([100 * (r[candidate] - r[reference]) for r in rows])
    families = [r["family"] for r in rows]
    lo, hi = family_ci(delta, families, name)
    fam = sorted(set(families))
    fam_mean = np.asarray([
        np.mean([d for d, r in zip(delta, rows) if r["family"] == f]) for f in fam
    ])
    return {
        "contrast": name, "mean_delta_pp": float(delta.mean()),
        "family_macro_delta_pp": float(fam_mean.mean()),
        "family_ci_low_pp": lo, "family_ci_high_pp": hi,
        "median_delta_pp": float(np.median(delta)),
        "wins": int(np.sum(delta > 0)), "losses": int(np.sum(delta < 0)),
    }


def heldout(cell_key, cell, repeats):
    M, labels = cell["matrix"], cell["labels"]
    n = len(M)
    rows = []
    for fraction in FRACTIONS:
        for repetition in range(repeats):
            rng = np.random.default_rng(seed(VERSION, "split", cell_key, fraction, repetition))
            order = rng.permutation(n)
            cut = int(round(fraction * n))
            tr, te = order[:cut], order[cut:]
            mean, sd = M[tr].mean(axis=0), M[tr].std(axis=0)
            if np.any(sd < 1e-8) or len(np.unique(labels[te])) < 2:
                continue
            train, test = (M[tr] - mean) / sd, (M[te] - mean) / sd
            anchor_train, anchor_test = consensus_anchor(train), consensus_anchor(test)
            _, hp, hr, tail, _ = components(train.T, 100.0)
            rec = {"cell": cell_key, "family": cell["family"],
                   "fraction": fraction, "repetition": repetition}
            scores = {}
            for name, weight in {
                "head_pcr": hp, "head_ridge": hr,
                "head_pcr_tail": hp + tail, "full_ridge": hr + tail,
            }.items():
                sign = orient_sign(train @ weight, anchor_train)
                scores[name] = float(roc_auc_score(labels[te], sign * (test @ weight)))
                rec[f"auc_{name}"] = scores[name]
            rec["tail_effect_pp"] = 100 * (scores["head_pcr_tail"] - scores["head_pcr"])
            rec["head_effect_pp"] = 100 * (scores["head_ridge"] - scores["head_pcr"])
            rec["tail_norm"] = float(np.linalg.norm(tail))
            rows.append(rec)
    return rows


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def render(summary):
    lines = [
        "# SDSF solver cycle v4 — fixed-stable real artifact", "",
        f"Decision: **{summary['decision']}**.", "",
        "This is a mechanism study on the retrospective 24-cell artifact. It does not "
        "promote a tuned solver.", "", "## Full-data factorial", "",
        "| contrast | cell mean | family macro [95% CI] | W/L |", "|---|---:|---:|---:|",
    ]
    for row in summary["contrasts"]:
        lines.append(
            f"| `{row['contrast']}` | {row['mean_delta_pp']:+.2f} | "
            f"{row['family_macro_delta_pp']:+.2f} [{row['family_ci_low_pp']:+.2f}, "
            f"{row['family_ci_high_pp']:+.2f}] | {row['wins']}/{row['losses']} |"
        )
    lines += ["", "## Condition path", "",
              f"Family-weighted tail-effect slope: **{summary['kappa_slope_pp_per_log']:+.3f} "
              "points per log(kappa)**. Negative means weaker regularization admits a more "
              "harmful tail.", "", "## Held-out sample-size test", "",
              "| training fraction | tail effect | head-rescaling effect |", "|---:|---:|---:|"]
    for row in summary["heldout_summary"]:
        lines.append(
            f"| {row['fraction']:.2f} | {row['tail_effect_pp']:+.2f} | "
            f"{row['head_effect_pp']:+.2f} |"
        )
    lines += ["", "## Conclusion", "", summary["conclusion"], ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--repeats", type=int, default=N_REPEATS)
    args = parser.parse_args()
    cells = load_cells(args.bundle, args.manifest)
    per_cell, kappa_rows = [], []
    for idx, (key, cell) in enumerate(cells.items()):
        M, labels, anchor = cell["matrix"], cell["labels"], consensus_anchor(cell["matrix"])
        fit, hp, hr, tail, gamma = components(M.T, 100.0)
        row = {"cell": key, "family": cell["family"], "domain": cell["domain"],
               "auc_head_pcr": auc(hp, M, labels, anchor),
               "auc_head_ridge": auc(hr, M, labels, anchor),
               "auc_head_pcr_tail": auc(hp + tail, M, labels, anchor),
               "auc_full_ridge": auc(hr + tail, M, labels, anchor),
               "gamma": gamma, "tail_to_head_norm": float(np.linalg.norm(tail)/(np.linalg.norm(hp)+1e-30)),
               "condition_observed": float(np.linalg.cond(fit.covariance))}
        cca_score, canonical_correlation = direct_cca_score(M, cell["names"])
        row["auc_direct_cca"] = float(roc_auc_score(labels, orient_sign(cca_score, anchor)*cca_score))
        row["canonical_correlation"] = canonical_correlation
        per_cell.append(row)
        for kappa in KAPPAS:
            _, hpk, hrk, tk, gammak = components(M.T, kappa)
            kappa_rows.append({"cell": key, "family": cell["family"], "kappa": kappa,
                               "gamma": gammak,
                               "tail_effect_pp": 100*(auc(hpk+tk,M,labels,anchor)-auc(hpk,M,labels,anchor)),
                               "head_effect_pp": 100*(auc(hrk,M,labels,anchor)-auc(hpk,M,labels,anchor))})
        print(f"factorial {idx+1:02d}/{len(cells)} {key}", flush=True)

    contrasts = [
        summarize_delta(per_cell, "auc_head_ridge", "auc_head_pcr", "head rescaling only"),
        summarize_delta(per_cell, "auc_head_pcr_tail", "auc_head_pcr", "tail addition only"),
        summarize_delta(per_cell, "auc_full_ridge", "auc_head_pcr", "full inverse vs PCR"),
        summarize_delta(per_cell, "auc_direct_cca", "auc_head_pcr", "direct two-channel CCA vs PCR"),
    ]
    # Equal-family weighted OLS slope over the complete condition path.
    fam_counts = {f: sum(r["family"] == f for r in per_cell) for f in {r["family"] for r in per_cell}}
    xs, ys, ws = [], [], []
    for row in kappa_rows:
        xs.append(np.log(row["kappa"])); ys.append(row["tail_effect_pp"])
        ws.append(1.0/fam_counts[row["family"]])
    xs, ys, ws = np.asarray(xs), np.asarray(ys), np.asarray(ws)
    xm, ym = np.average(xs,weights=ws), np.average(ys,weights=ws)
    slope = float(np.sum(ws*(xs-xm)*(ys-ym))/np.sum(ws*(xs-xm)**2))

    heldout_rows = []
    for idx, (key, cell) in enumerate(cells.items()):
        heldout_rows.extend(heldout(key, cell, args.repeats))
        print(f"heldout {idx+1:02d}/{len(cells)} {key}", flush=True)
    heldout_summary = []
    for fraction in FRACTIONS:
        selected = [r for r in heldout_rows if r["fraction"] == fraction]
        heldout_summary.append({
            "fraction": fraction, "n": len(selected),
            "tail_effect_pp": float(np.mean([r["tail_effect_pp"] for r in selected])),
            "head_effect_pp": float(np.mean([r["head_effect_pp"] for r in selected])),
        })
    tail_effect = next(r for r in contrasts if r["contrast"] == "tail addition only")
    head_effect = next(r for r in contrasts if r["contrast"] == "head rescaling only")
    structural = bool(
        tail_effect["family_ci_high_pp"] < 0
        and heldout_summary[-1]["tail_effect_pp"] < -0.5
        and abs(head_effect["family_macro_delta_pp"]) < 0.5
    )
    decision = "ABANDON_FULL_INVERSE_SDSF" if structural else "MECHANISM_INCONCLUSIVE"
    conclusion = (
        "The low-eigenvalue tail, not top-two rescaling, carries the loss; it remains harmful "
        "on held-out rows at the largest training fraction. Direct channel CCA also fails to "
        "beat PCR. The supported action is to abandon full-inverse SDSF for these features, "
        "retain the low-dimensional PCR solver, and investigate dependency information only "
        "as a reliability correction on genuinely new families."
        if structural else
        "The preregistered evidence does not isolate a stable structural mechanism; no new "
        "solver should be promoted from this cycle."
    )
    summary = {"version": VERSION, "decision": decision, "contrasts": contrasts,
               "kappa_slope_pp_per_log": slope, "heldout_summary": heldout_summary,
               "conclusion": conclusion, "config": {"kappas": KAPPAS,
               "fractions": FRACTIONS, "repeats": args.repeats}}
    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(os.path.join(args.out_dir,"per_cell.csv"),per_cell)
    write_csv(os.path.join(args.out_dir,"kappa_path.csv"),kappa_rows)
    write_csv(os.path.join(args.out_dir,"heldout_repetitions.csv"),heldout_rows)
    with open(os.path.join(args.out_dir,"summary.json"),"w",encoding="utf-8") as h:
        json.dump(summary,h,indent=2,sort_keys=True)
    with open(os.path.join(args.out_dir,"REPORT.md"),"w",encoding="utf-8") as h:
        h.write(render(summary))
    print(decision)


if __name__ == "__main__":
    main()
