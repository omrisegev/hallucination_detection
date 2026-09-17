"""Step 420 analysis, exactly as pre-registered in docs/experiments/LENGTH_EXPLICIT_CT7_V1.md.

Stops before scoring any new arm unless all three exactness gates pass.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
TEMPORAL = ROOT.parents[1] / ".worktrees/temporal-research-20260915"
sys.path.insert(0, str(ROOT))
import scripts.run_digitfree20_ladder_v1 as L  # noqa: E402
import scripts.analyze_chosen_token_step_tests_v2 as S  # noqa: E402
from scripts.run_lsml_gate_locator_research_v1 import score_locator  # noqa: E402
from scripts.run_length_calibrated_streams_v1 import COLS, OUT  # noqa: E402
from spectral_utils import frozen_locator_ct7 as C  # noqa: E402
from spectral_utils.chosen_token_calibration import SUFFICIENT  # noqa: E402
from spectral_utils.digitfree_broad50 import masked_answer_standardize  # noqa: E402

SEED = 420000


def mstd(v, ok, offsets):
    return masked_answer_standardize(v[:, None] if v.ndim == 1 else v, (ok[:, None] if ok.ndim == 1 else ok), offsets)


def main():
    data = L.load_data(); offsets = data["offsets"]; cells = data["cells"].astype(str)
    target = np.asarray(data["target"]); gate = np.asarray(data["gate"], bool); n_steps = np.diff(offsets)
    S_total = offsets[-1]

    # ---- load extraction
    top = np.full((S_total, 5), np.nan); cal = np.zeros((S_total, 5)); cnt = np.zeros((S_total, 5))
    codex = np.full((S_total, 5), np.nan); codex_ok = np.zeros((S_total, 5), bool); done = np.zeros(len(n_steps), bool)
    for path in sorted((OUT / "bank").glob("*.npz")):
        with np.load(path) as z, np.load(ROOT / "results/digitfree_broad50_v1/extracted" / path.name) as cz:
            assert np.array_equal(z["indexes"], cz["indexes"])
            cur = 0
            for i in z["indexes"]:
                a, b = offsets[i:i + 2]; n = b - a
                top[a:b] = z["top10"][cur:cur + n]; cal[a:b] = z["calibrated"][cur:cur + n]; cnt[a:b] = z["counts"][cur:cur + n]
                codex[a:b] = cz["values"][cur:cur + n][:, COLS]; codex_ok[a:b] = cz["available"][cur:cur + n][:, COLS]
                cur += n; done[i] = True
    assert done.all(), "bank extraction incomplete"
    with np.load(OUT / "bocpd.npz") as z:
        aux_top = z["top10"]; aux_cal = z["calibrated"]
    with np.load(ROOT / "results/joint_feature_selection_bocpd_v1/INPUTS.npz") as z:
        ct7_bocpd_view = z["bocpd"]
    gates = {}

    # ---- gate 1: bank Top10 replay
    ok = np.isfinite(top) & codex_ok
    assert np.array_equal(np.isfinite(top), codex_ok), "availability differs from the frozen extraction"
    # Amendment 1: the frozen extraction is stored as float32; require EXACT equality after the cast.
    gates["bank_top10_float64_max_abs_diff"] = float(np.max(np.abs(top[ok] - codex[ok])))
    gates["bank_top10_exact_after_float32_cast"] = bool(np.array_equal(top.astype(np.float32)[ok], codex[ok]))
    assert gates["bank_top10_exact_after_float32_cast"], "gate 1 failed: not exact after float32 cast"

    # ---- gate 2: BOCPD replay
    aux_std = mstd(aux_top, np.ones(S_total, bool), offsets)[:, 0]
    d2 = float(np.max(np.abs(aux_std - ct7_bocpd_view))); gates["bocpd_view_max_abs_diff"] = d2
    with np.load(TEMPORAL / "results/aligned_context_predictors_v1/SCORES_FROZEN.npz") as z:
        hist = z["bocpd"]
    with np.load(ROOT / "results/fusion_independence_atlas_v1/baseline_replay/SCORES_FROZEN.npz") as z:
        base = z["steps__append_innovation__H0lim"]
    rebuilt = base.copy()
    for a, b in zip(offsets[:-1], offsets[1:]):
        x = aux_top[a:b]
        if x.std() > 1e-12:
            rebuilt[a:b] = base[a:b] + .25 * base[a:b].std() * (x - x.mean()) / x.std()
    d2b = float(np.max(np.abs(rebuilt - hist))); gates["bocpd_historical_score_max_abs_diff"] = d2b
    assert d2 < 1e-8 and d2b < 1e-10, f"gate 2 failed: {d2} {d2b}"

    # ---- gate 3: CT7 rebuilt from the re-extracted readouts
    suff = S.load_folder("extracted_sufficient", offsets, len(SUFFICIENT))
    token_view = C.despiked_chosen_token_z(suff, offsets)
    bank_std = masked_answer_standardize(np.nan_to_num(top), np.isfinite(top), offsets)
    ct7_views = np.column_stack([bank_std, aux_std, token_view])
    with np.load(ROOT / "results/chosen_token_calibration_v1/CT7_DEV_SCORES.npz") as z:
        ct7_frozen = z["step_scores"]
    gates["ct7_rebuild_float64_max_abs_diff"] = float(np.max(np.abs(ct7_views.mean(1) - ct7_frozen)))
    bank32 = masked_answer_standardize(np.nan_to_num(top.astype(np.float32).astype(float)), np.isfinite(top), offsets)
    d3 = float(np.max(np.abs(np.column_stack([bank32, aux_std, token_view]).mean(1) - ct7_frozen)))
    gates["ct7_rebuild_exact_after_float32_cast"] = d3 == 0.0
    assert d3 == 0.0, f"gate 3 failed: {d3}"
    print("exactness gates PASS", gates, flush=True)

    # ---- new arms
    cal_ok = cnt > 0
    cal_std = masked_answer_standardize(cal, cal_ok, offsets)
    aux_cal_std = mstd(aux_cal, np.ones(S_total, bool), offsets)[:, 0]
    loglen = mstd(np.log(suff[:, 0]), np.ones(S_total, bool), offsets)[:, 0]
    lx7_views = np.column_stack([cal_std, aux_cal_std, token_view])
    arms = {"CT7": ct7_frozen, "LX7": lx7_views.mean(1), "LX8": np.column_stack([lx7_views, loglen]).mean(1),
            "LEN": loglen, "CT7+LEN": np.column_stack([ct7_views, loglen]).mean(1)}
    results = {a: score_locator(s, gate, data) for a, s in arms.items()}
    metrics = {a: {"pb": r["pb"], "within": r["within"]} for a, r in results.items()}
    names, pbd, wd = L.bootstrap(data, results, 10000, SEED)
    ix = {a: i for i, a in enumerate(names)}
    contrasts = {}
    for a, b in (("LX7", "CT7"), ("LX8", "LX7"), ("LX8", "CT7"), ("CT7+LEN", "CT7")):
        contrasts[f"{a} - {b}"] = {"pb": L.interval(pbd[:, ix[a]] - pbd[:, ix[b]], metrics[a]["pb"] - metrics[b]["pb"]),
                                   "within": L.interval(wd[:, ix[a]] - wd[:, ix[b]], metrics[a]["within"] - metrics[b]["within"])}

    # ---- diagnostics
    log_n = np.log(suff[:, 0])
    pb = np.char.startswith(cells, "pb_"); err = pb & (target >= 0)
    longest = np.array([int(np.argmax(suff[a:b, 0])) for a, b in zip(offsets[:-1], offsets[1:])])
    diag = {"truth_is_longest_step_pb_errors": float(np.mean(longest[err] == target[err]))}
    for a, s in arms.items():
        wc = [np.corrcoef(s[x:y], log_n[x:y])[0, 1] for x, y in zip(offsets[:-1], offsets[1:])
              if y - x > 2 and s[x:y].std() > 0 and log_n[x:y].std() > 0]
        peaks = np.asarray(results[a]["peaks"])
        diag[a] = {"within_answer_corr_log_length": float(np.nanmean(wc)),
                   "peak_is_longest_step_pb_errors": float(np.mean(peaks[err] == longest[err])),
                   "peak_at_step0_pb_errors": float(np.mean(peaks[err] == 0))}
    prm = np.repeat(np.char.startswith(cells, "prmbench_"), n_steps); lab = data["labels"]
    valid = prm & (lab >= 0); y = lab[valid] == 1

    def eff(V):
        M = V[valid].copy()
        for cls in (True, False):
            M[y == cls] -= M[y == cls].mean(0)
        lam = np.maximum(np.linalg.eigvalsh(np.corrcoef(M.T)), 0); return float(lam.sum() ** 2 / (lam ** 2).sum())

    diag["effective_views"] = {"CT7": eff(ct7_views), "LX7": eff(lx7_views), "LX8": eff(np.column_stack([lx7_views, loglen]))}
    diag["view_auc_prmb"] = {nm: float(roc_auc_score(y, v[valid])) for nm, v in
                             zip(["H0lim", "ve0", "ve0.75", "ve1", "H0lim_innov", "bocpd", "token"], lx7_views.T)} | {"loglen": float(roc_auc_score(y, loglen[valid]))}
    idx = np.concatenate([np.arange(k) for k in n_steps])
    diag["mean_calibrated_six_by_step_index"] = {str(k): float(lx7_views[:, :6].mean(1)[np.minimum(idx, 8) == k].mean()) for k in range(9)}

    report = {"gates": gates, "metrics": metrics, "contrasts": contrasts, "diagnostics": diag}
    (OUT / "RESULTS.json").write_text(json.dumps(L.clean(report), indent=1, sort_keys=True) + "\n")
    for a in arms:
        print(f"{a:8s} PB {100 * metrics[a]['pb']:.2f}  within {metrics[a]['within']:.4f}  "
              f"corr(logLen) {diag[a]['within_answer_corr_log_length']:+.2f}  peak=longest {diag[a]['peak_is_longest_step_pb_errors']:.2f}  "
              f"peak@0 {diag[a]['peak_at_step0_pb_errors']:.2f}")
    print("truth is longest step:", round(diag["truth_is_longest_step_pb_errors"], 3))
    for k, v in contrasts.items():
        p, w = v["pb"], v["within"]
        flag = lambda i: "*" if i["low"] > 0 or i["high"] < 0 else " "  # noqa: E731
        print(f"{k:16s} PB {100*p['point']:+.2f} [{100*p['low']:+.2f},{100*p['high']:+.2f}]{flag(p)}  "
              f"within {w['point']:+.4f} [{w['low']:+.4f},{w['high']:+.4f}]{flag(w)}")
    print("effective views", {k: round(v, 2) for k, v in diag["effective_views"].items()})
    print("view AUC", {k: round(v, 3) for k, v in diag["view_auc_prmb"].items()})
    print("mean calibrated six by step index", {k: round(v, 2) for k, v in diag["mean_calibrated_six_by_step_index"].items()})


if __name__ == "__main__":
    main()
