"""
Step 148 streaming pilot — prefix-AUROC, baseline shoot-out, online monitor.

Runs entirely locally (CPU) on raw entropy-trace caches:

    E1  prefix-AUROC curves: 16-feature suite on h[:n], continuous L-SML per
        (cell, budget), AUROC vs token budget (absolute) and trace fraction.
    E2  baseline shoot-out at every budget: DeepConf-style lowest group
        confidence (w=32/64/128), mean entropy, max entropy, tail confidence.
    E3  online monitor: causal per-token risk trajectories, threshold sweep
        -> detection rate vs false-alarm rate + flag earliness.
    E4  early-exit value: fraction of wrong-trace tokens saved at ~5%/10% FA.

Label protocol: final-answer correctness only (label 1 = correct). Unsupervised
throughout — labels touch nothing but the AUROC evaluation.

Results checkpoint to --out after every (cell, experiment) unit, so a killed
run loses at most one unit.

Usage:
    python scripts/streaming_pilot.py                      # full run
    python scripts/streaming_pilot.py --quick              # fewer budgets/boot
    python scripts/streaming_pilot.py --cells p4_math500_qwen7b_k10
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spectral_utils import (  # noqa: E402
    FEAT_NAMES, FEATURE_SIGNS, boot_auc,
    causal_trajectories, deepconf_lowest_group_conf, deepconf_tail_conf,
    iter_entropy_traces, lsml_continuous_pipeline, online_flag_curve,
    prefix_feature_matrix,
)

H16 = FEAT_NAMES[:16]
GOOD_5 = ["epr", "low_band_power", "sw_var_peak", "cusum_max", "spectral_entropy"]

ABS_BUDGETS = [16, 32, 64, 128, 256, 512, 1024, None]  # None = full trace
FRAC_BUDGETS = [0.1, 0.25, 0.5, 0.75, 1.0]
DEEPCONF_WINDOWS = [32, 64, 128]

# filename stem -> readable cell name (fallback: the stem itself)
CELL_NAMES = {
    "p1_gsm8k_llama8b": "gsm8k/Llama-3.1-8B",
    "p1_gsm8k_llama8b_k10": "gsm8k/Llama-3.1-8B_K10",
    "p4_math500_qwen7b_k10": "math500/Qwen2.5-Math-7B_K10",
    "p2c_gpqa_deepseek_r1_7b_inference": "gpqa/DeepSeek-R1-7B",
    "math500_T1.0": "math500/Qwen2.5-Math-1.5B_T1.0",
    "deepseek_r1_8b_gpqa_k2": "gpqa/DeepSeek-R1-8B_K2_TRUNC",
}


def load_cells(data_dirs, only=None):
    """Load every recognisable raw-trace pkl under the given dirs."""
    cells = {}
    for d in data_dirs:
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".pkl"):
                continue
            stem = fn[:-4]
            if only and stem not in only:
                continue
            path = os.path.join(d, fn)
            try:
                with open(path, "rb") as f:
                    obj = pickle.load(f)
                pairs = list(iter_entropy_traces(obj))
            except Exception as exc:
                print(f"  [skip] {fn}: {type(exc).__name__}: {exc}")
                continue
            pairs = [(tr, y) for tr, y in pairs if len(tr) >= 8]
            if len(pairs) < 30:
                print(f"  [skip] {fn}: only {len(pairs)} usable traces")
                continue
            name = CELL_NAMES.get(stem, stem)
            traces = [p[0] for p in pairs]
            labels = np.array([p[1] for p in pairs], dtype=int)
            if labels.min() == labels.max():
                print(f"  [skip] {fn}: single-class labels")
                continue
            cells[name] = (traces, labels)
            lens = np.array([len(t) for t in traces])
            print(f"  [cell] {name}: n={len(traces)} frac_correct={labels.mean():.3f} "
                  f"len med={int(np.median(lens))} max={lens.max()}")
    return cells


def scores_at_budget(traces, n_tokens):
    """All method scores for one budget. Returns (score_dict, valid_mask, meta)."""
    prefixes = [tr[: n_tokens] if n_tokens else tr for tr in traces]
    fd, valid = prefix_feature_matrix(prefixes, 10 ** 9, H16)
    pref_valid = [p for p, v in zip(prefixes, valid) if v]

    scores, meta = {}, {}
    for tag, feats in (("lsml16", H16), ("lsml5", GOOD_5)):
        fused, m = lsml_continuous_pipeline(fd, feats, FEATURE_SIGNS)
        scores[tag] = fused
        meta[tag] = {"K": m["K"], "residual": m["residual"]}
    scores["epr"] = -fd["epr"]
    scores["max_ent"] = np.array([-p.max() for p in pref_valid])
    scores["tail_conf"] = np.array([deepconf_tail_conf(p) for p in pref_valid])
    for w in DEEPCONF_WINDOWS:
        scores[f"deepconf_w{w}"] = np.array(
            [deepconf_lowest_group_conf(p, w) for p in pref_valid])
    # individual features (sign-oriented) for the per-feature table
    for f in H16:
        scores[f"feat_{f}"] = FEATURE_SIGNS[f] * fd[f]
    return scores, valid, meta


def frac_prefix(tr, f):
    return tr[: max(8, int(round(f * len(tr))))]


def run_e1_e2(cells, res, out_path, n_boot):
    for cell, (traces, labels) in cells.items():
        cell_res = res.setdefault(cell, {})
        e12 = cell_res.setdefault("e1_e2", {"abs": {}, "frac": {}})

        for kind, budgets in (("abs", ABS_BUDGETS), ("frac", FRAC_BUDGETS)):
            for b in budgets:
                key = "full" if (kind == "abs" and b is None) else b
                if key in e12[kind]:
                    continue  # resume
                t0 = time.time()
                if kind == "abs":
                    scores, valid, meta = scores_at_budget(traces, b)
                else:
                    prefs = [frac_prefix(tr, b) for tr in traces]
                    scores, valid, meta = scores_at_budget(prefs, None)
                y = labels[valid]
                entry = {"n_valid": int(valid.sum()), "meta": meta, "auc": {}}
                for tag, sc in scores.items():
                    a, lo, hi = boot_auc(y, sc, n=n_boot)
                    entry["auc"][tag] = (float(a), float(lo), float(hi))
                e12[kind][key] = entry
                save(res, out_path)
                a16 = entry["auc"]["lsml16"][0]
                dc = entry["auc"]["deepconf_w64"][0]
                print(f"  {cell} {kind}={key}: lsml16={a16:.3f} deepconf64={dc:.3f} "
                      f"({time.time() - t0:.0f}s)")


RISK_KEYS = ["cusum", "sw_var_sofar", "neg_group_conf", "run_mean_ent", "run_max_ent"]


def run_e3_e4(cells, res, out_path):
    for cell, (traces, labels) in cells.items():
        cell_res = res.setdefault(cell, {})
        if "e3" in cell_res:
            continue
        trajs_all = [causal_trajectories(tr) for tr in traces]
        e3 = {}
        for rk in RISK_KEYS:
            curve = online_flag_curve([t[rk] for t in trajs_all], labels)
            e3[rk] = curve
        cell_res["e3"] = e3
        # E4: interpolate tokens-saved at FA targets
        e4 = {}
        for rk, curve in e3.items():
            fas = np.array([c["false_alarm_rate"] for c in curve])
            det = np.array([c["detection_rate"] for c in curve])
            saved = np.array([c["frac_tokens_saved"] for c in curve])
            order = np.argsort(fas)
            e4[rk] = {
                f"fa{int(t * 100)}": {
                    "detection_rate": float(np.interp(t, fas[order], det[order])),
                    "frac_tokens_saved": float(np.interp(t, fas[order], saved[order])),
                }
                for t in (0.05, 0.10)
            }
        cell_res["e4"] = e4
        save(res, out_path)
        best = max(e4, key=lambda k: e4[k]["fa10"]["detection_rate"])
        print(f"  {cell} E3/E4: best monitor={best} "
              f"det@FA10={e4[best]['fa10']['detection_rate']:.2f} "
              f"saved={e4[best]['fa10']['frac_tokens_saved']:.2f}")


def save(res, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(res, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", nargs="*", default=["local_cache/raw_traces", "local_cache"])
    ap.add_argument("--out", default="results/streaming_pilot.pkl")
    ap.add_argument("--cells", nargs="*", default=None,
                    help="filename stems to include (default: all recognisable)")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--force", action="store_true", help="ignore existing results")
    args = ap.parse_args()

    if args.quick:
        args.boot = 200
        global ABS_BUDGETS, FRAC_BUDGETS
        ABS_BUDGETS = [32, 128, 512, None]
        FRAC_BUDGETS = [0.25, 0.5, 1.0]

    print("Loading cells...")
    cells = load_cells(args.data_dir, only=set(args.cells) if args.cells else None)
    if not cells:
        print("No usable cells found. Put raw trace pkls in local_cache/raw_traces/.")
        return 1

    res = {}
    if not args.force and os.path.exists(args.out):
        with open(args.out, "rb") as f:
            res = pickle.load(f)
        print(f"Resuming from {args.out} ({len(res)} cells present)")

    print("\nE1+E2: prefix-AUROC + baselines")
    run_e1_e2(cells, res, args.out, args.boot)
    print("\nE3+E4: online monitor")
    run_e3_e4(cells, res, args.out)
    print(f"\nDone. Results in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
