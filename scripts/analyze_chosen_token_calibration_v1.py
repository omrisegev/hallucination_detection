"""Chosen-token calibration: is it a new independent evidence source, and does it help?

Three parts, all on the frozen 13,769-answer development population:
1. Token level (no labels): correlation with top-50 entropy of raw surprisal versus the
   distribution-free statistics, per answer, averaged.
2. Step level diagnostic (labels for MEASUREMENT only, PRMB steps): conditional correlation of
   each new statistic with the 13 Step-414 evidence families, and the effective number of
   independent signals with the new family added.
3. Matched fusion contrasts under the Step-413 contract (five source folds, non-digit tail15
   gate, H1 anchor, 10,000-draw paired source-group bootstrap).
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
from spectral_utils.chosen_token_calibration import NAMES as CAL  # noqa: E402
from spectral_utils.digitfree_broad50 import NAMES as BANK, masked_answer_standardize  # noqa: E402
from spectral_utils.laplacian_upcr import IU_FIT_DEFAULTS  # noqa: E402
from spectral_utils.lsml_gate_locator_research import FusionRecipe, _orient, fit_fusion_weights  # noqa: E402
from spectral_utils.upcr import upcr_fit  # noqa: E402

OUT = ROOT / "results/chosen_token_calibration_v1"
SEED = 414500


def load_calibration(offsets):
    n = len(offsets) - 1
    x = np.full((offsets[-1], len(CAL)), np.nan); done = np.zeros(n, bool)
    tok_corr = np.full((n, 1 + len(CAL)), np.nan); tokens = np.zeros(n); censored = np.zeros(n)
    for path in sorted((OUT / "extracted").glob("*.npz")):
        with np.load(path) as z:
            cursor = 0
            for pos, i in enumerate(z["indexes"]):
                a, b = offsets[i:i + 2]
                x[a:b] = z["values"][cursor:cursor + b - a]; cursor += b - a
                tok_corr[i] = z["token_entropy_corr"][pos]; tokens[i] = z["tokens"][pos]
                censored[i] = z["censored"][pos]; done[i] = True
    if not done.all():
        raise ValueError("incomplete calibration extraction")
    return x, tok_corr, tokens, censored


def main():
    data = L.load_data(); offsets = data["offsets"]; cells = data["cells"].astype(str)
    raw, tok_corr, tokens, censored = load_calibration(offsets)
    cal = masked_answer_standardize(raw, np.isfinite(raw), offsets)
    report = {"censored_tokens": int(censored.sum()), "tokens": int(tokens.sum())}

    # ---- 1. token level
    labels_tok = ["surprisal"] + list(CAL)
    w = tokens / tokens.sum()
    ok = np.isfinite(tok_corr)
    report["token_level_corr_with_entropy"] = {
        name: {"answer_weighted_mean": float(np.nansum(tok_corr[:, j] * w) / w[ok[:, j]].sum()),
               "median": float(np.nanmedian(tok_corr[:, j]))} for j, name in enumerate(labels_tok)}

    # ---- 2. step-level family diagnostic on PRMB labelled steps
    x = data["x"]; n_steps = np.diff(offsets); step_labels = data["labels"]
    with np.load(ROOT / "results/joint_feature_selection_bocpd_v1/INPUTS.npz") as z:
        bocpd = z["bocpd"]; noreset = z["noreset"]
    rel_pos = np.concatenate([np.arange(n) / max(n - 1, 1) for n in n_steps])
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
    for j, name in enumerate(CAL):
        fam_cols[name] = [cal[:, j]]
    prm = np.repeat(np.char.startswith(cells, "prmbench_"), n_steps)
    valid = prm & (step_labels >= 0); y = step_labels[valid] == 1
    virt, aucs = {}, {}
    for f, members in fam_cols.items():
        cols = []
        for m in members:
            v = m[valid]; a = roc_auc_score(y, v)
            s = 1.0 if a >= .5 else -1.0          # diagnostic orientation only
            cols.append(s * (v - v.mean()) / (v.std() + 1e-12))
        vv = np.column_stack(cols).mean(axis=1); vv = (vv - vv.mean()) / (vv.std() + 1e-12)
        virt[f] = vv; aucs[f] = float(roc_auc_score(y, vv))
    names = list(virt); V = np.column_stack([virt[f] for f in names])
    Vc = V.copy()
    for cls in (True, False):
        Vc[y == cls] -= Vc[y == cls].mean(axis=0)
    cond = np.corrcoef(Vc.T)
    base = [names.index(f) for f in names if f not in CAL]

    def pr(ix):
        lam = np.maximum(np.linalg.eigvalsh(cond[np.ix_(ix, ix)]), 0)
        return float(lam.sum() ** 2 / (lam ** 2).sum())

    report["step_family_auc"] = aucs
    report["effective_signals_conditional"] = {"12_families_base": pr(base)}
    report["conditional_corr_new_vs_families"] = {}
    for c in CAL:
        k = names.index(c)
        report["effective_signals_conditional"][f"base_plus_{c}"] = pr(base + [k])
        report["conditional_corr_new_vs_families"][c] = {names[j]: float(cond[k, j]) for j in base}

    # ---- 3. matched fusion contrasts (Step 413 contract)
    x20 = L.build_matrix(data); gate = data["gate"]
    folds = np.repeat(data["folds"], n_steps)
    anchor20 = L.ANCHOR
    pitn = cal[:, CAL.index("pit_normal")]; stdx = cal[:, CAL.index("std_excess_surprisal")]
    six_idx = [list(L.ROSTER20).index(n) for n in ("H0lim", "ve0", "ve0.75", "ve1", "H0lim_prefix_innovation", "bocpd_residual")]
    mats = {
        "B20": (x20, anchor20),
        "B20_pit": (np.column_stack([x20, pitn]), anchor20),
        "B20_pit_std": (np.column_stack([x20, pitn, stdx]), anchor20),
        "six": (x20[:, six_idx], 0),
        "six_pit": (np.column_stack([x20[:, six_idx], pitn]), 0),
    }
    oof = {}
    for bank, (M, anchor) in mats.items():
        arms = ("equal",) if bank.startswith("six") else ("equal", "lsml", "iu")
        for arm in arms:
            s = np.empty(len(M))
            for outer in range(5):
                te = folds == outer; tr = M[~te]
                if arm == "equal":
                    wv = np.ones(M.shape[1]) / M.shape[1]
                elif arm == "iu":
                    wv, _ = _orient(tr, upcr_fit(tr.T, **dict(IU_FIT_DEFAULTS)).w, anchor)
                else:
                    names_b = tuple(f"s{j}" for j in range(M.shape[1]))
                    wv, _ = fit_fusion_weights(tr, FusionRecipe(bank, names_b, "continuous", anchor=anchor), seed=SEED)
                s[te] = M[te] @ wv
            oof[f"{bank}_{arm}"] = s
    oof["pit_normal_alone"] = pitn
    oof["std_excess_alone"] = stdx
    oof["H1_alone"] = x20[:, anchor20]
    results = {a: score_locator(s, gate, data) for a, s in oof.items()}
    metrics = {a: {"pb": r["pb"], "within": r["within"]} for a, r in results.items()}
    arms, pb, within = L.bootstrap(data, results, 10000, SEED + 1)
    col_ix = {a: i for i, a in enumerate(arms)}
    contrasts = {}
    for a, b in (("B20_pit_equal", "B20_equal"), ("B20_pit_std_equal", "B20_equal"),
                 ("six_pit_equal", "six_equal"), ("B20_pit_lsml", "B20_pit_equal"),
                 ("B20_pit_iu", "B20_pit_equal"), ("B20_lsml", "B20_equal"),
                 ("pit_normal_alone", "H1_alone")):
        contrasts[f"{a} - {b}"] = {
            "pb": L.interval(pb[:, col_ix[a]] - pb[:, col_ix[b]], metrics[a]["pb"] - metrics[b]["pb"]),
            "within": L.interval(within[:, col_ix[a]] - within[:, col_ix[b]], metrics[a]["within"] - metrics[b]["within"])}
    report["metrics"] = metrics; report["contrasts"] = contrasts
    (OUT / "RESULTS.json").write_text(json.dumps(L.clean(report), indent=1, sort_keys=True) + "\n")
    np.savez_compressed(OUT / "OOF.npz", **oof)

    print(json.dumps(L.clean({k: report[k] for k in ("censored_tokens", "tokens", "token_level_corr_with_entropy",
                                                    "step_family_auc", "effective_signals_conditional")}), indent=1))
    for c in CAL:
        row = report["conditional_corr_new_vs_families"][c]
        print(c, {k: round(v, 2) for k, v in row.items()})
    for a in sorted(metrics, key=lambda k: -metrics[k]["pb"]):
        print(f"{a:22s} PB {100 * metrics[a]['pb']:.2f}  within {metrics[a]['within']:.4f}")
    for k, v in contrasts.items():
        p, wv = v["pb"], v["within"]
        print(f"{k:34s} PB {100*p['point']:+.2f} [{100*p['low']:+.2f},{100*p['high']:+.2f}]  "
              f"within {wv['point']:+.4f} [{wv['low']:+.4f},{wv['high']:+.4f}]")


if __name__ == "__main__":
    main()
