"""B16 (13 level/realization + 3 digit channels), DS filter + plain average (BASE) and DS filter +
grouped DS-MLE weights (GRP), fitted PER MODEL PER DATASET on the self-generated answers.

Mirrors `per_dataset_fit_run.py::fit_arms` (ssl-pseudolabel-residual-v1, Step 462) for the two
content arms without position, on four new cells (GSM8K / MATH x Qwen3-4B / Qwen3-8B own greedy
answers). Each cell is fitted on ALL its non-truncated answers, label-free, and scores the same
answers. Reads no label.

Hard checks before the new cells are fitted:
  1. feature replay: the 16 per-step channels recomputed from the raw ProcessBench Qwen3-8B GSM8K
     telemetry equal the stored source arrays (level bank, CT7 profile, digit features);
  2. fit replay: the fit function below, on the stored source bank, reproduces the Step 462
     per-dataset B16 BASE and GRP scores of pb_gsm8k_q8 and pb_math_q4.
Outputs results/self_generated_step_labels_v1/b16/{FEATURES.npz, STEP_SCORES.npz, FIT.json, REPLAY.json}.
"""
import hashlib
import importlib.util
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

MAIN = Path(os.environ.get("HD_MAIN_CHECKOUT", r"C:\Users\omris\TAU\hallucination_detection"))
SSL = MAIN / ".worktrees/ssl-pseudolabel-residual-v1"
HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SSL / "scripts/experiments"))
sys.path.insert(0, str(SSL))
from spectral_utils.external_banks_v4 import realized_z, realized_drv, digit_step_features  # noqa: E402
from spectral_utils.external_generalization.fusion import validate_telemetry, top10  # noqa: E402
import er_stage_a as SA, er_stage_b as SB, tail_calib_common as TC, lsml_merge_step as MS, ds_group_weights as GW  # noqa: E402
from calfix_common import tail_marks  # noqa: E402

_seg = importlib.util.spec_from_file_location("sgs", HERE / "spectral_utils/self_generated_steps.py")
SGS = importlib.util.module_from_spec(_seg); _seg.loader.exec_module(SGS)

ROOT = HERE / "results/self_generated_step_labels_v1"
OUT = ROOT / "b16"
EPS = 1e-12
CELLS = {"evdrop_gsm8k_qwen3_4b": "raw_gsm8k_T0.0.pkl", "evdrop_gsm8k_qwen3_8b": "raw_gsm8k_T0.0.pkl",
         "evdrop_math_qwen3_4b": "raw_math_T0.0.pkl", "evdrop_math_qwen3_8b": "raw_math_T0.0.pkl"}
DIG = ["digit_alternative", "digit_spread", "digit_alternative_innovation"]


def answer_standardize(values, offsets):
    """Verbatim copy of depth-feature-fusion-v1 spectral_utils/lsml_gate_locator_research.py:45."""
    x = np.asarray(values, dtype=np.float64)
    out = np.zeros_like(x, dtype=np.float64)
    for i in range(len(offsets) - 1):
        sl = slice(int(offsets[i]), int(offsets[i + 1]))
        block = x[sl]
        mean = block.mean(axis=0)
        sd = block.std(axis=0)
        out[sl] = np.divide(block - mean, sd, out=np.zeros_like(block), where=sd > EPS)
    return out


def step_channels(row, spans):
    """Raw per-step [steps, 13] (level 11, realized_z, realized_drv) and digit values/active [steps, 3]."""
    spans = np.asarray(spans, int)
    level = top10(validate_telemetry(row), spans)
    rz = realized_z(row, spans)
    rd = realized_drv(row, spans)
    dv, da = digit_step_features(row, spans)
    return np.column_stack([level, rz, rd]), np.asarray(dv, float), np.asarray(da, bool)


def build_bank(raw13, dvals, dact, off):
    """B16 matrix exactly as per_dataset_fit_run.py (values + masked digit z-scores)."""
    values = answer_standardize(raw13, off)
    D3 = np.zeros((len(raw13), 3))
    for a, b in zip(off[:-1], off[1:]):
        for j in range(3):
            v = dact[a:b, j]
            y = dvals[a:b, j][v]
            if len(y) and y.std() > 1e-12:
                D3[a:b, j][v] = (y - y.mean()) / y.std()
    return np.column_stack([values, D3])


def fit_cell(V, off, rc, keyb, keyg):
    """per_dataset_fit_run.fit_arms restricted to BASE and GRP, dsr = ptr = the cell's rows."""
    MK = SB.random_tie_marks(V, off, .2, keyb)
    TT = tail_marks(V, off, .2, tie_aware=True, centred=True)[0]
    est = SA.em_estimate(MK[rc], "ds")
    surv = np.flatnonzero(est["pi"] > 0.5)
    if len(surv) < 6:
        raise RuntimeError(f"only {len(surv)} survivors")
    anchor_ch = 0 if 0 in surv else int(surv[np.argmax(est["pi"][surv])])
    X = V[:, surv]
    base = X @ np.full(len(surv), 1 / len(surv))
    T = TT[:, surv]
    anchor = int(np.flatnonzero(surv == anchor_ch)[0])
    gb = MS.canon(TC.lsml_fit_scaled(T[rc], anchor, X[rc], standardize=True, loading_scale="unit")["groups"])
    gbm, _ = MS.absorb_merge(np.corrcoef(T[rc], rowvar=False), gb)
    G = int(gbm.max()) + 1
    Z = GW.group_matrix(X, off, gbm, None, answer_standardize)
    gv = SB.random_tie_marks(Z, off, .2, keyg[:, :G])
    eg = SA.em_estimate(gv[rc], "ds")
    w = SB.mle_weights(eg["psi"], eg["eta"])
    if w.sum() <= 0:
        raise RuntimeError("GRP weights all 0")
    grp = GW.weighted_group_score(Z, w)
    diag = {"survivors": surv.tolist(), "pi": np.asarray(est["pi"]).tolist(), "anchor_fallback": anchor_ch != 0,
            "groups": gbm.tolist(), "K": G, "group_weights": np.asarray(w).tolist()}
    return base, grp, diag


def keys(S):
    return np.random.default_rng(20260928).random((S, 16)), np.random.default_rng(20260929).random((S, 13))


def source_population():
    """Stored source B16 inputs in per_dataset_fit_run order (OOF_ANSWERS / JOINED)."""
    import pandas as pd
    mi = json.loads((SSL / "results/algorithm_decisions_v1/run_20260928/INPUT_MANIFEST.json").read_text(encoding="utf8"))
    p = {k: Path(v["path"]) for k, v in mi.items() if isinstance(v, dict) and "path" in v}
    R = MAIN / ".worktrees/readout-quickest-detection-v1/results/step_evidence_v1"
    ans = pd.read_csv(R / "OOF_ANSWERS.csv", encoding="utf-8-sig")
    off = np.load(R / "OOF_STEP_SCORES.npz")["offsets"]
    lv = np.load(p["level_bank"]); names11 = list(map(str, lv["channels"]))
    prof = np.load(p["ct7_profiles"]).astype(float)
    pn = json.loads(p["ct7_profile_validation"].read_text(encoding="utf8"))["channels"]
    raw13 = np.column_stack([lv["level"].astype(float), prof[:, pn.index("chosen_token_z_despiked")],
                             lv["derivative"][:, names11.index("chosen_surprisal")].astype(float)])
    DF = np.load(p["digit_features"])
    return ans, off, raw13, DF["values"].astype(float), DF["active"].astype(bool), names11


def replay(ans, off, raw13, dv, da):
    out = {}
    # 1. feature replay on pb_gsm8k_q8 from raw teacher-forced telemetry
    pk = pickle.load(open(MAIN / "dataset_cache/repgrid/pb_qwen3_8b/processbench_gsm8k.pkl", "rb"))
    by_id = {r["id"]: r for r in pk.values()}
    worst = {"level": 0.0, "realized_z": 0.0, "realized_drv": 0.0, "digits": 0.0, "digit_active_mismatch": 0}
    n = 0
    for i in np.flatnonzero(ans.cell.to_numpy() == "pb_gsm8k_q8"):
        rid = str(ans.id.iloc[i]).split("::")[-1]
        row = by_id[rid]
        spans = np.asarray(row["step_token_spans"], int)
        if (spans[:, 1] <= spans[:, 0]).any():
            continue
        a, b = off[i], off[i + 1]
        r13, v, act = step_channels(row, spans)
        worst["level"] = max(worst["level"], float(np.abs(r13[:, :11] - raw13[a:b, :11]).max()))
        worst["realized_z"] = max(worst["realized_z"], float(np.abs(r13[:, 11] - raw13[a:b, 11]).max()))
        worst["realized_drv"] = max(worst["realized_drv"], float(np.abs(r13[:, 12] - raw13[a:b, 12]).max()))
        worst["digits"] = max(worst["digits"], float(np.abs(v - dv[a:b]).max()))
        worst["digit_active_mismatch"] += int((act != da[a:b]).sum())
        n += 1
    out["feature_replay_pb_gsm8k_q8"] = {"answers": n, **worst}
    if not (worst["level"] < 1e-5 and worst["realized_z"] < 1e-6 and worst["realized_drv"] < 1e-5
            and worst["digits"] < 1e-6 and worst["digit_active_mismatch"] == 0):
        raise SystemExit("HARD STOP feature replay: " + json.dumps(worst))
    # 2. fit replay against the Step 462 per-dataset scores
    V = build_bank(raw13, dv, da, off)
    kb, kg = keys(int(off[-1]))
    ref = np.load(SSL / "results/per_dataset_fit_v1/run_20260929/STEP_SCORES.npz")
    step_cell = np.repeat(ans.cell.to_numpy(), np.diff(off))
    for ce in ("pb_gsm8k_q8", "pb_math_q4"):
        rc = np.flatnonzero(step_cell == ce)
        base, grp, _ = fit_cell(V, off, rc, kb, kg)
        out[f"fit_replay_{ce}"] = {"BASE": float(np.abs(base[rc] - ref["B16__BASE"][rc]).max()),
                                   "GRP": float(np.abs(grp[rc] - ref["B16__GRP"][rc]).max())}
        if max(out[f"fit_replay_{ce}"].values()) > 1e-9:
            raise SystemExit("HARD STOP fit replay: " + json.dumps(out))
    return out


def new_population():
    rows = [json.loads(l) for l in open(ROOT / "private/own_answers.jsonl", encoding="utf-8")]
    key = {(k["cell"], k["src_idx"]): k["item_id"] for k in map(json.loads, open(ROOT / "private/ITEM_KEY.jsonl", encoding="utf-8"))
           if k["source"] == "own"}
    meta, raw, dvl, dal, lens = [], [], [], [], []
    for cell, pkl in CELLS.items():
        d = pickle.load(open(MAIN / "dataset_cache/repgrid" / cell / pkl, "rb"))
        for r in (r for r in rows if r["cell"] == cell and r["n_tokens"] < r["max_new"]):
            c = d[r["src_idx"]]["candidates"][0]
            _, cspans = SGS.segment_answer(c["full_text"])
            starts = np.asarray([a for a, _ in c["token_offsets"]])
            spans = []
            for s, e in cspans:
                idx = np.flatnonzero((starts >= s) & (starts < e))
                spans.append([int(idx[0]), int(idx[-1]) + 1] if len(idx) else [0, 0])
            spans = np.asarray(spans, int)
            if (spans[:, 1] <= spans[:, 0]).any():
                raise SystemExit(f"empty step in {cell} {r['src_idx']}")
            row = {"gen_token_ids": c["gen_token_ids"], "top_k_logprobs": c["top_k_logprobs_raw"],
                   "token_entropies": c["token_entropies"], "token_spilled_energies": c["token_spilled_energies"],
                   "token_logsumexp": c["token_logsumexp"]}
            r13, v, act = step_channels(row, spans)
            raw.append(r13); dvl.append(v); dal.append(act); lens.append(len(spans))
            meta.append({"cell": cell, "src_idx": r["src_idx"], "item_id": key.get((cell, r["src_idx"])),
                         "n_steps": len(spans), "step_token_spans": spans.tolist()})
        print(cell, sum(m["cell"] == cell for m in meta), flush=True)
        del d
    off = np.concatenate([[0], np.cumsum(lens)])
    return meta, off, np.vstack(raw), np.vstack(dvl), np.vstack(dal)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ans, soff, sraw, sdv, sda, names11 = source_population()
    rep = replay(ans, soff, sraw, sdv, sda)
    json.dump(rep, open(OUT / "REPLAY.json", "w"), indent=1)
    print("replay PASS", json.dumps(rep), f"{time.time() - t0:.0f}s", flush=True)
    del ans, sraw, sdv, sda
    meta, off, raw13, dv, da = new_population()
    np.savez_compressed(OUT / "FEATURES.npz", raw13=raw13, digit_values=dv, digit_active=da, offsets=off)
    V = build_bank(raw13, dv, da, off)
    kb, kg = keys(int(off[-1]))
    step_cell = np.repeat([m["cell"] for m in meta], np.diff(off))
    base, grp = np.full(len(V), np.nan), np.full(len(V), np.nan)
    fits = {}
    for cell in CELLS:
        rc = np.flatnonzero(step_cell == cell)
        b, g, d = fit_cell(V, off, rc, kb, kg)
        base[rc], grp[rc] = b[rc], g[rc]
        fits[cell] = {"answers": sum(m["cell"] == cell for m in meta), "steps": len(rc), **d}
        print(cell, "survivors", len(d["survivors"]), "K", d["K"], flush=True)
    if not (np.isfinite(base).all() and np.isfinite(grp).all()):
        raise SystemExit("non-finite scores")
    np.savez_compressed(OUT / "STEP_SCORES.npz", B16__BASE=base, B16__GRP=grp, offsets=off)
    json.dump({"answers": meta, "fits": fits, "channels": names11 + ["realized_z", "realized_drv"] + DIG,
               "ssl_git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=SSL, capture_output=True, text=True).stdout.strip(),
               "scores_sha256": hashlib.sha256((OUT / "STEP_SCORES.npz").read_bytes()).hexdigest(),
               "elapsed_s": round(time.time() - t0, 1), "labels_read": False}, open(OUT / "FIT.json", "w"))
    print("done", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
