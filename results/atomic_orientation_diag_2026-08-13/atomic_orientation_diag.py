#!/usr/bin/env python3
"""Diagnosis: where does the atomic target direction live, and can the IU
score's own nonlinearity orient it without labels?

Everything is retrospective.  Correctness labels are used ONLY for
(a) reference directions (Fisher class-mean difference) and (b) AUROC
readouts.  No candidate direction estimator reads a label.

Candidate label-free orientation estimators, all built from the coupling
between the atomic residual matrix R and the standardized IU score b:
  gamma2  : quadratic Hermite moment   E[ r * (b^2 - skew*b - 1) ]
  gamma2s : gamma2 sign-harmonized across cells by sign(skew(b))
  gamma3  : cubic Hermite moment      -E[ r * orth(b^3 - 3b) ]
  sir     : sliced (decile) inverse regression of r on b (diagnostic only)
"""

from __future__ import annotations

import dataclasses
import hashlib
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
from spectral_utils.atomic_neutral_residual import (  # noqa: E402
    atomic_contribution_space,
    fit_atomic_neutral_calibration,
    atomic_neutral_score,
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

# Codex's frozen reference values (from the structural audit / message).
CODEX_SPECTRUM = [
    0.0016, 0.0058, 0.0258, 0.1289, 0.2215, 0.2675, 0.3330, 0.4675, 0.5536,
    0.6632, 0.8274, 0.9607, 1.0256, 1.1206, 1.2726, 2.0356, 7.0892,
]
CODEX_DIRECTION_SHA = (
    "d7de9faeb68825ac540cbaa70868aeb52dcf548d6b7e11e480664f446e952edb"
)
CODEX_COVARIANCE_SHA = (
    "9ef9010eb9f7969831603db2ff0a484c77c1b05b9455901e0e83449b707ca3ed"
)


def log(msg=""):
    print(msg, flush=True)


def sha256_array(values):
    values = np.ascontiguousarray(np.asarray(values, dtype=float))
    return hashlib.sha256(values.view(np.uint8)).hexdigest()


def normalize(v):
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def cos(u, v):
    u, v = np.asarray(u, float), np.asarray(v, float)
    du, dv = np.linalg.norm(u), np.linalg.norm(v)
    if du <= 0 or dv <= 0:
        return float("nan")
    return float(u @ v / (du * dv))


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


def full_residuals(aspace):
    """Per-atom standardized IU-orthogonal residuals (column-independent)."""
    transform = fit_contribution_transform(
        aspace, np.arange(len(aspace.baseline_score), dtype=int)
    )
    baseline, values = transform.apply(
        aspace.baseline_score, aspace.contributions
    )
    return baseline, values


def make_cell(name, group, domain, F, names, correctness):
    correctness = np.asarray(correctness, dtype=int)
    fitted = upcr_fit(np.asarray(F, float), **IU_FIT_DEFAULTS)
    aspace = atomic_contribution_space(F, names, fitted.w)
    baseline, residuals = full_residuals(aspace)
    return {
        "cell": name,
        "group": group,
        "domain": domain,
        "F": np.asarray(F, float),
        "names": tuple(names),
        "w": np.asarray(fitted.w, float),
        "y": correctness,
        "aspace": aspace,
        "b": baseline,                    # standardized IU score
        "R_full": residuals,              # n x p_c standardized residuals
        "n": int(len(correctness)),
        "pi": float(np.mean(correctness)),
    }


def restrict(cellrec, atom_names):
    lookup = {n: i for i, n in enumerate(cellrec["names"])}
    cols = [lookup[n] for n in atom_names]
    return cellrec["R_full"][:, cols]


def load_original():
    cells = []
    with np.load(BUNDLE, allow_pickle=True) as data:
        for name in SOURCE_CELLS:
            F, names = load_contract(data, name, "mixed_v2")
            y = np.asarray(data[f"{name}__labels"], dtype=int)
            cells.append(make_cell(
                name, original_family(name), "original_23", F, names, y
            ))
    return cells


def load_processbench():
    cells = []
    for model in PROCESS_MODELS:
        for subset in PROCESS_SUBSETS:
            path = (
                REAL / "dataset_cache" / "repgrid" / f"pb_{model}"
                / f"processbench_{subset}.pkl"
            )
            items = process_items(path)
            telemetry = [telemetry_only(row) for _, row in items]
            y = [int(row["label"] == -1) for _, row in items]
            F, names, _, _ = mixed_v2_matrix(telemetry)
            cells.append(make_cell(
                f"{model}__{subset}", subset, "processbench_qwen",
                F, names, y,
            ))
            log(f"  loaded {model}__{subset}: n={len(y)}")
    root = REAL / "dataset_cache" / "repgrid" / "pb_llama31_8b"
    for subset in PROCESS_SUBSETS:
        items = process_items(root / f"processbench_{subset}.pkl")
        telemetry = [telemetry_only(row) for _, row in items]
        y = [int(row["label"] == -1) for _, row in items]
        F, names, _, _ = mixed_v2_matrix(telemetry)
        cells.append(make_cell(
            f"llama31_8b__{subset}", subset, "processbench_llama",
            F, names, y,
        ))
        log(f"  loaded llama31_8b__{subset}: n={len(y)}")
    return cells


# ---------- label-free orientation estimators (never read y) ----------

def hermite2_moment(R, b):
    """E[r * phi2(b)], phi2 = b^2 - skew*b - 1 (orthogonal to b and 1)."""
    skew = float(np.mean(b ** 3))
    phi = b ** 2 - skew * b - 1.0
    phi -= float(np.mean(phi))          # exact in-sample centering
    return R.T @ phi / len(b), skew


def hermite3_moment(R, b):
    """-E[r * orth(b^3 - 3 b)] with numerical orthogonalization to b, 1."""
    phi = b ** 3 - 3.0 * b
    phi -= float(phi @ b / (b @ b)) * b
    phi -= float(np.mean(phi))
    return -(R.T @ phi) / len(b)


def sir_direction(R, b, n_slices=10, align_to=None):
    order = np.argsort(b)
    splits = np.array_split(order, n_slices)
    means = np.stack([R[idx].mean(axis=0) for idx in splits])
    wts = np.asarray([len(idx) for idx in splits], float) / len(b)
    M = (means * wts[:, None]).T @ means
    vals, vecs = np.linalg.eigh(M)
    v = vecs[:, -1]
    if align_to is not None and v @ align_to < 0:
        v = -v
    return v


def fisher_direction(R, y):
    """Supervised reference only: class-mean difference of residuals."""
    return R[y == 1].mean(axis=0) - R[y == 0].mean(axis=0)


def score_with_direction(cellrec, cal, direction):
    """Apply a direction through the exact frozen candidate machinery."""
    patched = dataclasses.replace(
        cal, direction=np.asarray(direction, float)
    )
    fitted = atomic_neutral_score(
        cellrec["aspace"], cellrec["w"], patched
    )
    y = cellrec["y"]
    iu = float(roc_auc_score(y, fitted.baseline_score))
    new = float(roc_auc_score(y, fitted.score))
    return iu, new, 100.0 * (new - iu)


def equal_group(rows, key):
    groups = sorted({r["group"] for r in rows})
    per_group = [
        float(np.mean([r[key] for r in rows if r["group"] == g]))
        for g in groups
    ]
    deltas = [r[key] for r in rows]
    return {
        "equal_group_delta_pp": float(np.mean(per_group)),
        "cell_macro_delta_pp": float(np.mean(deltas)),
        "wins": int(np.sum(np.asarray(deltas) > 0)),
        "losses": int(np.sum(np.asarray(deltas) < 0)),
        "worst_pp": float(np.min(deltas)),
        "best_pp": float(np.max(deltas)),
    }


def main():
    t0 = time.time()
    results = {}

    log("=== Stage L: load original 23 cells ===")
    original = load_original()
    log(f"loaded {len(original)} original cells "
        f"({sum(c['n'] for c in original)} samples)")

    log("")
    log("=== Stage 0: reproduce Codex's frozen atomic calibration ===")
    cal = fit_atomic_neutral_calibration(
        [c["aspace"] for c in original]
    )
    atoms = cal.feature_names
    p = len(atoms)
    log(f"eligible atoms ({p}): {', '.join(atoms)}")
    log("eigenvalues:")
    log("  mine : " + " ".join(f"{v:.4f}" for v in cal.eigenvalues))
    log("  codex: " + " ".join(f"{v:.4f}" for v in CODEX_SPECTRUM))
    spectrum_match = bool(np.allclose(
        cal.eigenvalues, CODEX_SPECTRUM, atol=5e-4
    ))
    dir_sha = sha256_array(cal.direction)
    cov_sha = sha256_array(cal.residual_covariance)
    log(f"spectrum match (5e-4): {spectrum_match}")
    log(f"direction sha match : {dir_sha == CODEX_DIRECTION_SHA}")
    log(f"covariance sha match: {cov_sha == CODEX_COVARIANCE_SHA}")
    log(f"null band: [{cal.null_lower:.6f}, {cal.null_upper:.6f}]  "
        f"neutral dim: {int(np.sum(cal.neutral_mask))}")
    results["stage0"] = {
        "atoms": list(atoms),
        "eigenvalues": cal.eigenvalues.tolist(),
        "spectrum_match": spectrum_match,
        "direction_sha_match": dir_sha == CODEX_DIRECTION_SHA,
        "covariance_sha_match": cov_sha == CODEX_COVARIANCE_SHA,
        "null_band": [cal.null_lower, cal.null_upper],
    }

    # Frozen-order residuals per cell.
    for c in original:
        c["R"] = restrict(c, atoms)

    log("")
    log("=== Stage 1: where does the target live? (labels: diagnosis) ===")
    # IU weight sign consistency across cells for the frozen atoms.
    sign_table = {}
    for a in atoms:
        signs = []
        for c in original:
            idx = c["names"].index(a)
            signs.append(1 if c["w"][idx] > 0 else -1)
        sign_table[a] = float(np.mean(np.asarray(signs) > 0))
    n_flippy = sum(1 for v in sign_table.values() if v < 1.0)
    log(f"atoms with any negative IU weight across cells: {n_flippy}/{p}")
    for a, frac in sorted(sign_table.items(), key=lambda kv: kv[1]):
        if frac < 1.0:
            log(f"  {a}: positive in {frac:.0%} of cells")
    results["w_sign_positive_fraction"] = sign_table

    # Fisher directions.
    for c in original:
        c["fisher"] = fisher_direction(c["R"], c["y"])
    g_star = normalize(sum(c["n"] * c["fisher"] for c in original))
    coherence = [cos(c["fisher"], g_star) for c in original]
    log("per-cell cos(fisher_c, g*): "
        f"min {min(coherence):+.3f}  median {np.median(coherence):+.3f}  "
        f"max {max(coherence):+.3f}  frac>0 "
        f"{np.mean(np.asarray(coherence) > 0):.2f}")
    results["fisher_coherence"] = coherence

    log("")
    log("global supervised direction g* (17 atoms):")
    for a, v in sorted(zip(atoms, g_star), key=lambda kv: -abs(kv[1])):
        log(f"  {a:>22} {v:+.4f}")

    # Eigenmass of g* across the calibration spectrum.
    vals, vecs = np.linalg.eigh(cal.residual_covariance)
    mass = (vecs.T @ g_star) ** 2
    mass /= mass.sum()
    band = (vals >= cal.null_lower) & (vals <= cal.null_upper)
    log("")
    log("eigenmass of g* per calibration mode (lambda: mass, band?):")
    for j in range(p):
        log(f"  {vals[j]:8.4f}: {mass[j]:.4f}"
            + ("   <-- in null band" if band[j] else ""))
    log(f"g* mass inside null band: {float(mass[band].sum()):.4f}")
    log(f"g* mass in top spike (lambda={vals[-1]:.3f}): {mass[-1]:.4f}")
    results["gstar_eigmass"] = {
        "eigenvalues": vals.tolist(),
        "mass": mass.tolist(),
        "band_mass": float(mass[band].sum()),
    }

    # Anchor and frozen-direction alignment with g*.
    ones = normalize(np.ones(p))
    P0 = vecs[:, band] @ vecs[:, band].T
    log("")
    log("alignment with g* (positive = points toward target):")
    log(f"  codex frozen direction (P0 @ invdep): {cos(cal.direction, g_star):+.4f}")
    log(f"  inverse-dependence anchor raw       : {cos(cal.anchor, g_star):+.4f}")
    log(f"  equal anchor raw                    : {cos(ones, g_star):+.4f}")
    log(f"  P0 @ equal anchor                   : {cos(P0 @ ones, g_star):+.4f}")
    log(f"  P0 @ g* (best in-band direction)    : {cos(P0 @ g_star, g_star):+.4f}"
        f"   (= sqrt(band mass))")
    results["alignment"] = {
        "codex_direction_vs_gstar": cos(cal.direction, g_star),
        "invdep_anchor_vs_gstar": cos(cal.anchor, g_star),
        "equal_anchor_vs_gstar": cos(ones, g_star),
        "P0_equal_vs_gstar": cos(P0 @ ones, g_star),
    }

    log("")
    log("=== Stage 2: label-free b-coupled orientation estimators ===")
    header = (f"{'cell':>32} {'n':>5} {'pi':>5} {'skew':>6} "
              f"{'c2':>6} {'c2s':>6} {'c3':>6} {'sir':>6}")
    log(header)
    for c in original:
        g2, skew = hermite2_moment(c["R"], c["b"])
        g3 = hermite3_moment(c["R"], c["b"])
        sirv = sir_direction(c["R"], c["b"], align_to=g3)
        c["g2"], c["skew"], c["g3"], c["sir"] = g2, skew, g3, sirv
        c["cos_g2"] = cos(g2, c["fisher"])
        c["cos_g2s"] = cos(np.sign(skew) * g2 if skew != 0 else g2,
                           c["fisher"])
        c["cos_g3"] = cos(g3, c["fisher"])
        c["cos_sir"] = cos(sirv, c["fisher"])
        log(f"{c['cell']:>32} {c['n']:>5} {c['pi']:.2f} {c['skew']:+.2f} "
            f"{c['cos_g2']:+.3f} {c['cos_g2s']:+.3f} "
            f"{c['cos_g3']:+.3f} {c['cos_sir']:+.3f}")
    for key in ("cos_g2", "cos_g2s", "cos_g3", "cos_sir"):
        arr = np.asarray([c[key] for c in original])
        log(f"  {key}: median {np.median(arr):+.3f}  frac>0 "
            f"{np.mean(arr > 0):.2f}")
    results["percell_cosines"] = {
        key: [float(c[key]) for c in original]
        for key in ("cos_g2", "cos_g2s", "cos_g3", "cos_sir")
    }

    pooled = {
        "g2": normalize(sum(c["n"] * c["g2"] for c in original)),
        "g2s": normalize(sum(
            c["n"] * np.sign(c["skew"]) * c["g2"] for c in original
        )),
        "g3": normalize(sum(c["n"] * c["g3"] for c in original)),
    }
    log("")
    for name, vec in pooled.items():
        agree = int(np.sum(np.sign(vec) == np.sign(g_star)))
        log(f"pooled {name}: cos with g* = {cos(vec, g_star):+.4f}   "
            f"sign agreement {agree}/{p}")
    results["pooled_cosines"] = {
        name: cos(vec, g_star) for name, vec in pooled.items()
    }
    log("")
    log("pooled g3 vs g* per atom:")
    for a, v1, v2 in sorted(
        zip(atoms, pooled["g3"], g_star), key=lambda kv: -abs(kv[2])
    ):
        log(f"  {a:>22} g3 {v1:+.4f}   g* {v2:+.4f}")

    log("")
    log("=== Stage 3: scoring under the frozen 1/sqrt(p) machinery ===")
    groups = sorted({c["group"] for c in original})

    def lofo_pool(maker, heldout):
        vecs_ = [
            maker(c) * c["n"] for c in original if c["group"] != heldout
        ]
        return normalize(sum(vecs_))

    makers = {
        "codex_frozen": None,  # handled separately
        "g2_lofo": lambda c: c["g2"],
        "g2s_lofo": lambda c: np.sign(c["skew"]) * c["g2"],
        "g3_lofo": lambda c: c["g3"],
        "fisher_lofo": lambda c: c["fisher"],   # supervised reference
    }
    rows = []
    for c in original:
        row = {"cell": c["cell"], "group": c["group"], "n": c["n"]}
        iu, _, d = score_with_direction(c, cal, cal.direction)
        row["iu_auroc"] = iu
        row["codex_frozen"] = d
        for name, maker in makers.items():
            if maker is None:
                continue
            direction = lofo_pool(maker, c["group"])
            row[name] = score_with_direction(c, cal, direction)[2]
        # transductive per-cell variants (label-free, same-batch like IU)
        row["g3_transductive"] = score_with_direction(c, cal, c["g3"])[2]
        row["fisher_incell"] = score_with_direction(c, cal, c["fisher"])[2]
        # supervised band-projection: is the null band itself the bottleneck?
        row["fisher_lofo_banded"] = score_with_direction(
            c, cal, P0 @ lofo_pool(lambda cc: cc["fisher"], c["group"])
        )[2]
        rows.append(row)
        log(f"  scored {c['cell']}")

    method_cols = [
        "codex_frozen", "g2_lofo", "g2s_lofo", "g3_lofo",
        "g3_transductive", "fisher_lofo", "fisher_lofo_banded",
        "fisher_incell",
    ]
    log("")
    log("original 23 cells, equal-group delta vs IU (pp):")
    orig_summary = {}
    for m in method_cols:
        s = equal_group(rows, m)
        orig_summary[m] = s
        log(f"  {m:>20}: {s['equal_group_delta_pp']:+.3f}pp  "
            f"W/L {s['wins']}/{s['losses']}  worst {s['worst_pp']:+.2f}")
    results["original_scoring"] = {"rows": rows, "summary": orig_summary}

    log("")
    log("=== Stage T: ProcessBench transfer (frozen source-23 pooling) ===")
    try:
        external = load_processbench()
    except Exception as exc:  # noqa: BLE001
        log(f"ProcessBench loading failed: {exc}")
        external = []
    if external:
        pooled_fisher = normalize(sum(
            c["n"] * c["fisher"] for c in original
        ))
        transfer_dirs = {
            "codex_frozen": cal.direction,
            "g2s_pooled": pooled["g2s"],
            "g3_pooled": pooled["g3"],
            "fisher_pooled": pooled_fisher,   # supervised reference
        }
        ext_rows = []
        for c in external:
            c["R"] = restrict(c, atoms)
            c["fisher"] = fisher_direction(c["R"], c["y"])
            c["g3"] = hermite3_moment(c["R"], c["b"])
            row = {"cell": c["cell"], "group": c["group"],
                   "domain": c["domain"], "n": c["n"]}
            row["iu_auroc"] = score_with_direction(c, cal, cal.direction)[0]
            for name, d in transfer_dirs.items():
                row[name] = score_with_direction(c, cal, d)[2]
            row["g3_transductive"] = score_with_direction(c, cal, c["g3"])[2]
            row["cos_g3_fisher"] = cos(c["g3"], c["fisher"])
            row["cos_pooledg3_fisher"] = cos(pooled["g3"], c["fisher"])
            row["cos_codex_fisher"] = cos(cal.direction, c["fisher"])
            ext_rows.append(row)
            log(f"  scored {c['cell']} (n={c['n']})")
        log("")
        for domain in ("processbench_qwen", "processbench_llama"):
            sel = [r for r in ext_rows if r["domain"] == domain]
            log(f"{domain} equal-group delta vs IU (pp):")
            for m in list(transfer_dirs) + ["g3_transductive"]:
                s = equal_group(sel, m)
                log(f"  {m:>20}: {s['equal_group_delta_pp']:+.3f}pp  "
                    f"W/L {s['wins']}/{s['losses']}")
            cs = [r["cos_g3_fisher"] for r in sel]
            log(f"  in-cell cos(g3, fisher): median {np.median(cs):+.3f}")
            cs = [r["cos_pooledg3_fisher"] for r in sel]
            log(f"  cos(pooled g3, fisher): median {np.median(cs):+.3f}")
            cs = [r["cos_codex_fisher"] for r in sel]
            log(f"  cos(codex dir, fisher): median {np.median(cs):+.3f}")
        results["transfer"] = ext_rows

    log("")
    log("=== Stage 4: does target variance track pi(1-pi)? ===")
    var_target = [float(np.var(c["R"] @ g_star)) for c in original]
    pis = np.asarray([c["pi"] for c in original])
    v = np.asarray(var_target)
    def _pearson(a, bb):
        return float(np.corrcoef(a, bb)[0, 1])
    def _spearman(a, bb):
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(bb)).astype(float)
        return _pearson(ra, rb)
    log(f"corr(var_along_g*, pi(1-pi)): pearson "
        f"{_pearson(v, pis * (1 - pis)):+.3f}  spearman "
        f"{_spearman(v, pis * (1 - pis)):+.3f}")
    log(f"corr(var_along_g*, pi)      : pearson {_pearson(v, pis):+.3f}  "
        f"spearman {_spearman(v, pis):+.3f}")
    results["stage4"] = {
        "var_along_gstar": var_target,
        "pi": pis.tolist(),
        "pearson_var_vs_pi1mpi": _pearson(v, pis * (1 - pis)),
        "spearman_var_vs_pi1mpi": _spearman(v, pis * (1 - pis)),
    }

    log("")
    log("=== Stage 5: per-mode variance dispersion across cells ===")
    log("(high CV for target-heavy modes = leverage for joint-diag)")
    tmat = np.stack([
        np.asarray([
            float(np.var(c["R"] @ vecs[:, j])) for j in range(p)
        ])
        for c in original
    ])
    cv = tmat.std(axis=0) / np.maximum(tmat.mean(axis=0), 1e-12)
    for j in range(p):
        log(f"  lambda {vals[j]:8.4f}: CV {cv[j]:.3f}  g*-mass {mass[j]:.3f}")
    tg = np.asarray([float(np.var(c["R"] @ g_star)) for c in original])
    log(f"  along g*: CV {tg.std() / tg.mean():.3f}")
    results["stage5_cv"] = cv.tolist()

    with (OUT / "RESULT.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=1, default=float)
    log("")
    log(f"done in {time.time() - t0:.0f}s -> {OUT / 'RESULT.json'}")


if __name__ == "__main__":
    main()
