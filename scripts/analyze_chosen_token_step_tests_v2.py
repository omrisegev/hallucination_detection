"""Step 416: does a length-free step readout rescue the entropy-free chosen-token statistic?

Readouts of the same per-token statistics (all answer-standardized afterwards, per contract):
  top10_std_excess, top10_pit_normal    the bank's Top10 mean (Step 415)
  z_std_excess, z_pit                   sum / sqrt(n)
  z_pooled                              sum(excess) / sqrt(sum varentropy + n*floor)
  z_pooled_pos                          z_pooled with the step-position trend removed, slope fitted on
                                        the training folds only (label-free)
Diagnostics use PRMB step labels for measurement only. Fusion contrasts follow the Step 413 contract.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.run_digitfree20_ladder_v1 as L  # noqa: E402
from scripts.run_lsml_gate_locator_research_v1 import score_locator  # noqa: E402
from spectral_utils.chosen_token_calibration import NAMES as CAL, SUFFICIENT, step_z_readouts  # noqa: E402
from spectral_utils.digitfree_broad50 import NAMES as BANK, masked_answer_standardize  # noqa: E402
from spectral_utils.laplacian_upcr import IU_FIT_DEFAULTS  # noqa: E402
from spectral_utils.lsml_gate_locator_research import FusionRecipe, _orient, fit_fusion_weights  # noqa: E402
from spectral_utils.upcr import upcr_fit  # noqa: E402

OUT = ROOT / "results/chosen_token_calibration_v1"
SEED = 416000


def load_folder(folder, offsets, width):
    x = np.full((offsets[-1], width), np.nan); done = np.zeros(len(offsets) - 1, bool)
    for path in sorted((OUT / folder).glob("*.npz")):
        with np.load(path) as z:
            idx = z["indexes"]; vals = z["values"]  # decompress once, not per answer
            cursor = 0
            for i in idx:
                a, b = offsets[i:i + 2]; x[a:b] = vals[cursor:cursor + b - a]; cursor += b - a; done[i] = True
    if not done.all():
        raise ValueError(f"incomplete {folder}")
    return x


def astd(v, offsets):
    return masked_answer_standardize(v[:, None], np.isfinite(v)[:, None], offsets)[:, 0]


def main():
    data = L.load_data(); offsets = data["offsets"]; cells = data["cells"].astype(str)
    n_steps = np.diff(offsets); folds = np.repeat(data["folds"], n_steps)
    top10 = load_folder("extracted", offsets, len(CAL))
    suff = load_folder("extracted_sufficient", offsets, len(SUFFICIENT))
    z = step_z_readouts(suff)
    rel_pos = np.concatenate([np.arange(n) / max(n - 1, 1) for n in n_steps])
    pos_std = astd(rel_pos, offsets)
    readouts = {
        "top10_std_excess": astd(top10[:, CAL.index("std_excess_surprisal")], offsets),
        "top10_pit_normal": astd(top10[:, CAL.index("pit_normal")], offsets),
        "z_std_excess": astd(z[:, 0], offsets),
        "z_pit": astd(z[:, 1], offsets),
        "z_pooled": astd(z[:, 2], offsets),
    }
    # label-free position removal: slope fitted on training folds, applied out of fold
    zp = readouts["z_pooled"]; resid = np.empty_like(zp)
    for outer in range(5):
        tr = folds != outer; te = ~tr
        slope = float(np.dot(zp[tr], pos_std[tr]) / max(np.dot(pos_std[tr], pos_std[tr]), 1e-12))
        resid[te] = zp[te] - slope * pos_std[te]
    readouts["z_pooled_pos"] = astd(resid, offsets)

    # ---- diagnostics
    x = data["x"]; step_labels = data["labels"]
    with np.load(ROOT / "results/joint_feature_selection_bocpd_v1/INPUTS.npz") as f:
        bocpd = f["bocpd"]; noreset = f["noreset"]
    col = lambda n: x[:, list(BANK).index(n)]  # noqa: E731
    fam_cols = {
        "rank_risk": [col(f"rank_{i}_risk") for i in range(1, 16)],
        "provided_token": [col(n) for n in ("surprisal", "top1_loggap", "censored_rank50", "mass_above")],
        "tail": [col("tail15"), col("tail50")],
        "entropy_shape": [col(n) for n in ("a0.25", "a0.5", "H1", "a2", "a4", "Hinf", "H0lim")],
        "varentropy": [col(n) for n in ("ve0", "ve0.5", "ve0.75", "ve1", "ve2", "ve4")],
        "top2_ratio": [col("top2_ratio")],
        "shape_innov": [col(f"{n}_prefix_innovation") for n in ("a0.25", "a0.5", "H1", "a2", "a4", "Hinf", "H0lim")],
        "ve_innov": [col(f"{n}_prefix_innovation") for n in ("ve0", "ve0.5", "ve0.75", "ve1", "ve2", "ve4")],
        "dynamics": [col("top15_turnover"), col("top50_truncated_js")],
        "bocpd": [bocpd], "noreset": [noreset], "position": [rel_pos],
    }
    prm = np.repeat(np.char.startswith(cells, "prmbench_"), n_steps)
    valid = prm & (step_labels >= 0); y = step_labels[valid] == 1

    def virtual(members):
        cols = []
        for m in members:
            v = m[valid]; s = 1.0 if roc_auc_score(y, v) >= .5 else -1.0
            cols.append(s * (v - v.mean()) / (v.std() + 1e-12))
        vv = np.column_stack(cols).mean(axis=1); return (vv - vv.mean()) / (vv.std() + 1e-12)

    fams = {f: virtual(m) for f, m in fam_cols.items()}
    base_names = list(fams); B = np.column_stack([fams[f] for f in base_names])

    def cond_corr(M):
        Mc = M.copy()
        for cls in (True, False):
            Mc[y == cls] -= Mc[y == cls].mean(axis=0)
        return np.corrcoef(Mc.T)

    def pr(c):
        lam = np.maximum(np.linalg.eigvalsh(c), 0); return float(lam.sum() ** 2 / (lam ** 2).sum())

    log_len = np.log(suff[:, 0])
    diag = {"base_effective_signals": pr(cond_corr(B))}
    for name, r in readouts.items():
        v = virtual([r]); c = cond_corr(np.column_stack([B, v]))
        dist = [abs(c[-1, j]) for j, f in enumerate(base_names) if f not in ("provided_token", "position")]
        diag[name] = {
            "auc": float(roc_auc_score(y, r[valid])),
            "corr_log_step_length_raw": float(np.corrcoef(z[:, 2] if name.startswith("z_pooled") else
                                                           (z[:, 0] if name == "z_std_excess" else
                                                            (z[:, 1] if name == "z_pit" else
                                                             top10[:, CAL.index("std_excess_surprisal" if "std" in name else "pit_normal")])),
                                                           log_len)[0, 1]),
            "corr_position": float(np.corrcoef(r, rel_pos)[0, 1]),
            "max_abs_cond_corr_distribution_families": float(max(dist)),
            "cond_corr_provided_token": float(c[-1, base_names.index("provided_token")]),
            "effective_signals_with_it": pr(c),
        }

    # ---- fusion contrasts (Step 413 contract)
    x20 = L.build_matrix(data); gate = data["gate"]
    six_idx = [list(L.ROSTER20).index(n) for n in ("H0lim", "ve0", "ve0.75", "ve1", "H0lim_prefix_innovation", "bocpd_residual")]
    # orientation of the added view is fixed by construction: higher = more surprising than expected
    add = {k: readouts[k] for k in ("top10_std_excess", "z_pooled", "z_pooled_pos", "z_pit")}
    oof = {"six_equal": None, "B20_equal": None}
    mats = {"six": (x20[:, six_idx], 0), "B20": (x20, L.ANCHOR)}
    for k, v in add.items():
        mats[f"six+{k}"] = (np.column_stack([x20[:, six_idx], v]), 0)
        mats[f"B20+{k}"] = (np.column_stack([x20, v]), L.ANCHOR)
    for bank, (M, anchor) in mats.items():
        arms = ("equal", "lsml", "iu") if bank in ("B20+z_pooled_pos", "six+z_pooled_pos") else ("equal",)
        for arm in arms:
            s = np.empty(len(M))
            for outer in range(5):
                te = folds == outer; tr = M[~te]
                if arm == "equal":
                    w = np.ones(M.shape[1]) / M.shape[1]
                elif arm == "iu":
                    w, _ = _orient(tr, upcr_fit(tr.T, **dict(IU_FIT_DEFAULTS)).w, anchor)
                else:
                    w, _ = fit_fusion_weights(tr, FusionRecipe(bank, tuple(f"s{j}" for j in range(M.shape[1])),
                                                                "continuous", anchor=anchor), seed=SEED)
                s[te] = M[te] @ w
            oof[f"{bank}_{arm}"] = s
    for k in ("top10_std_excess", "z_pooled", "z_pooled_pos"):
        oof[f"{k}_alone"] = readouts[k]
    oof = {k: v for k, v in oof.items() if v is not None}
    results = {a: score_locator(s, gate, data) for a, s in oof.items()}
    metrics = {a: {"pb": r["pb"], "within": r["within"]} for a, r in results.items()}
    arms_list, pb, within = L.bootstrap(data, results, 10000, SEED + 1)
    ix = {a: i for i, a in enumerate(arms_list)}
    pairs = [("z_pooled_alone", "top10_std_excess_alone"), ("z_pooled_pos_alone", "z_pooled_alone")]
    for k in add:
        pairs += [(f"six+{k}_equal", "six_equal"), (f"B20+{k}_equal", "B20_equal")]
    pairs += [("B20+z_pooled_pos_lsml", "B20+z_pooled_pos_equal"), ("B20+z_pooled_pos_iu", "B20+z_pooled_pos_equal"),
              ("six+z_pooled_pos_lsml", "six+z_pooled_pos_equal"), ("six+z_pooled_pos_iu", "six+z_pooled_pos_equal")]
    contrasts = {}
    for a, b in pairs:
        contrasts[f"{a} - {b}"] = {
            "pb": L.interval(pb[:, ix[a]] - pb[:, ix[b]], metrics[a]["pb"] - metrics[b]["pb"]),
            "within": L.interval(within[:, ix[a]] - within[:, ix[b]], metrics[a]["within"] - metrics[b]["within"])}
    report = {"diagnostics": diag, "metrics": metrics, "contrasts": contrasts}
    (OUT / "STEP_TESTS_V2.json").write_text(json.dumps(L.clean(report), indent=1, sort_keys=True) + "\n")

    print(f"base effective signals (12 within-answer families): {diag['base_effective_signals']:.2f}")
    print(f"{'readout':18s} {'AUC':>6s} {'r(logLen)':>9s} {'r(pos)':>7s} {'max|cond| dist':>14s} {'r(provided)':>11s} {'PR+it':>6s}")
    for name in readouts:
        d = diag[name]
        print(f"{name:18s} {d['auc']:6.3f} {d['corr_log_step_length_raw']:9.3f} {d['corr_position']:7.3f} "
              f"{d['max_abs_cond_corr_distribution_families']:14.2f} {d['cond_corr_provided_token']:11.2f} {d['effective_signals_with_it']:6.2f}")
    for a in sorted(metrics, key=lambda k: -metrics[k]["pb"]):
        print(f"{a:30s} PB {100 * metrics[a]['pb']:.2f}  within {metrics[a]['within']:.4f}")
    for k, v in contrasts.items():
        p, w = v["pb"], v["within"]
        flag = lambda i: "*" if i["low"] > 0 or i["high"] < 0 else " "  # noqa: E731
        print(f"{k:50s} PB {100*p['point']:+.2f} [{100*p['low']:+.2f},{100*p['high']:+.2f}]{flag(p)} "
              f"within {w['point']:+.4f} [{w['low']:+.4f},{w['high']:+.4f}]{flag(w)}")


if __name__ == "__main__":
    main()
