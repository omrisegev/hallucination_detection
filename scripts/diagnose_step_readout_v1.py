"""Window-to-step readout variants on frozen dual__iu window risks (all ProcessBench rows).

Motivation: in fusion_shrinkage_iu_v1 the peaks both IU and the shrinkage candidate miss are
adjacent to the true step in 938/2561 cases and share an 8-token window with it in 836/2561.
The frozen readout spreads each window's risk onto its tokens and takes the max token inside
each step, so a window straddling a step boundary hands its peak to both steps and ties or
near-ties are resolved by position. This script re-reads the SAME frozen window risks with
alternative rules and scores ProcessBench with the saved entropy-q0.3 fold gate. No fusion
change, no new fit. Development diagnostic.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "local_cache" / "short_cycle01_code"))
import spectral_utils  # noqa: E402
spectral_utils.__path__.append(str(ROOT / "spectral_utils"))
from spectral_utils.historical_fusion_evaluation import pb_metrics  # noqa: E402
from spectral_utils.answer_localization_v2 import moment_plan  # noqa: E402
from spectral_utils.window_localization import windows_to_tokens  # noqa: E402

BENCH = ROOT / "results" / "localization_full_benchmark_v3"
GATE = ROOT / "results" / "fusion_fixed_gate_v1"
FOLDS = ROOT / "results" / "localization_source_group_audit_v1" / "FOLDS_V2.json"
OUT = ROOT / "results" / "fusion_step_readout_v1"


def readouts(w, ws, we, ss, se, tokens):
    """Return {name: step scores} from window risks w over windows [ws,we) and steps [ss,se)."""
    tok = np.asarray(windows_to_tokens(moment_plan(int(tokens), 8), w), float)   # frozen spreading rule
    out = {}
    out["max_token"] = np.array([np.nanmax(tok[a:b]) for a, b in zip(ss, se)])
    cent = (ws + we) / 2.0
    for name, rule in (("center_max", "max"), ("center_mean", "mean")):
        v = []
        for a, b in zip(ss, se):
            inside = w[(cent >= a) & (cent < b)]
            if inside.size == 0:                    # step shorter than a window: fall back to token max
                v.append(np.nanmax(tok[a:b]))
            else:
                v.append(inside.max() if rule == "max" else inside.mean())
        out[name] = np.asarray(v, float)
    v = []
    for a, b in zip(ss, se):                        # windows fully inside the step only
        inside = w[(ws >= a) & (we <= b)]
        v.append(inside.max() if inside.size else np.nanmax(tok[a:b]))
    out["inside_max"] = np.asarray(v, float)
    for k in (3, 10):
        v = []
        for a, b in zip(ss, se):
            t = np.sort(tok[a:b][np.isfinite(tok[a:b])])[::-1]
            v.append(t[:min(k, len(t))].mean() if len(t) else np.nan)
        out[f"top{k}_token_mean"] = np.asarray(v, float)
    return out


def main():
    j = json.loads((BENCH / "evaluation" / "JOINED.json").read_text(encoding="utf-8"))
    z = np.load(BENCH / "evaluation" / "JOINED.npz")
    recs = j["records"]; target = z["target"]
    cells = np.array([r["cell"] for r in recs]); pb = np.array([c.startswith("pb_") for c in cells])
    folds = json.loads(FOLDS.read_text(encoding="utf-8"))
    gate = json.load(open(GATE / "METRICS.json"))
    thr = {int(k): v for k, v in gate["arms"]["dual__iu"]["rows"]["entropy_mean|quantile_0.3"]["thresholds"].items()}
    det = np.load(GATE / "DETECTORS.npz")["entropy_mean"]
    fthr = np.array([thr.get(int(folds["outer"].get(r["group_id"], -1)), np.nan) for r in recs])
    names = None; peaks = {}; valid = np.zeros(len(recs), bool)
    for i in np.flatnonzero(pb):
        p = BENCH / "scores" / f"{recs[i]['uid']}.npz"
        with np.load(p) as f:
            if "dual__iu__window" not in f.files:
                continue
            w, ws, we, ss, se = f["dual__iu__window"], f["window_starts"], f["window_ends"], f["step_starts"], f["step_ends"]
            frozen = f["dual__iu__risk"]
        ro = readouts(w, ws, we, ss, se, recs[i]["tokens"])
        assert np.allclose(ro["max_token"], frozen, atol=1e-9), recs[i]["uid"]
        if names is None:
            names = list(ro); peaks = {n: np.full(len(recs), -1) for n in names}
        for n in names:
            s = ro[n]; peaks[n][i] = int(np.nanargmax(s)) if np.isfinite(s).any() else -1
        valid[i] = True
    print("PB rows with frozen windows:", int(valid.sum()))
    OUT.mkdir(exist_ok=True)
    res = {}
    T, C = target[pb], cells[pb]
    for n in names:
        pk = peaks[n]; pv = valid & pb & (pk >= 0) & np.isfinite(det) & np.isfinite(fthr)
        pred = np.where(det >= fthr, pk, -1); pred[~pv] = -1
        m = pb_metrics(T, pred[pb], pv[pb], C)
        err = pb & (target >= 0)
        raw_exact = float(np.mean(pk[err] == target[err])); adj = float(np.mean(np.abs(pk[err] - target[err]) <= 1))
        res[n] = dict(pb_all8=m["macros"]["all"], pb_q8=m["macros"]["q8"], raw_peak_exact=raw_exact, within_one=adj,
                      cells={c: x["f1"] for c, x in m["cells"].items()})
        print(f"{n:18s} PB all8 {m['macros']['all']*100:6.2f}  Q8 {m['macros']['q8']*100:6.2f}  raw-peak exact {raw_exact*100:5.1f}  within-1 {adj*100:5.1f}")
    json.dump(res, open(OUT / "METRICS.json", "w"), indent=1)


if __name__ == "__main__":
    main()
