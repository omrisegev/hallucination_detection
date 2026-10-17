#!/usr/bin/env python
"""Tail-fitted token L-SML on CT7's seven streams (Omri, 2026-09-24).

Per answer, each stream marks its highest tokens as 1 (top 20% of the answer's valid tokens, or the
top 30, or the top 10), the rest 0, centred within the answer.  Continuous L-SML is fitted on those
indicators over the training answers (no labels); the weights are then applied to the CONTINUOUS
answer-local token streams, and each step is read out with the Top10 or Top30 token mean.

Everything else is the unchanged ct7_token_lsml_v1 (Step 436) pipeline: same token matrices, CT7
step-0 despike, answer-local axis, donor standardizer, outer folds, evaluation and CT7 replay.
Protocol: results/ct7_token_tail_lsml_v1/PROTOCOL.json (written before any number).

    python -B scripts/experiments/ct7_token_tail_lsml_v1.py --config configs/ct7_token_tail_lsml_v1.json
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ct7_levers_common as L  # noqa: E402
import ct7_token_lsml_v1 as T  # noqa: E402

from spectral_utils.claude_feature_bank_v1 import fit_l_sml_weights, fit_token_standardizer, fuse_token_matrix  # noqa: E402
from spectral_utils.ct7_token_streams import STREAMS, masked_answer_local, masked_step_top10  # noqa: E402

SCHEMA = "ct7-token-tail-lsml-v1"
MARKS = {"top20pct": ("frac", 0.20), "top30tok": ("count", 30), "top10tok": ("count", 10)}
READOUTS = {"r10": 10, "r30": 30}


def tail_indicator(x: np.ndarray, rows: np.ndarray, mark) -> np.ndarray:
    """Per column: 1 on the answer's top-k valid tokens, 0 elsewhere, centred over valid rows."""
    kind, val = mark; out = np.zeros_like(x); idx = np.flatnonzero(rows)
    if len(idx) == 0:
        return out
    k = max(1, int(np.ceil(val * len(idx)))) if kind == "frac" else min(int(val), len(idx))
    for j in range(x.shape[1]):
        top = idx[np.argsort(-x[idx, j], kind="stable")[:k]]
        out[top, j] = 1.0
        out[idx, j] -= k / len(idx)
    return out


def build_methods(td):
    d = td.d; total = int(d.off[-1]); methods, fits = {}, {}
    if "ct7" in d.references:
        methods["ct7"] = L.method_from_scores(d, d.references["ct7"])
    prep = T.prepared_matrices(td, despike=True, cols=T.ALL)
    s, _z = T.ct7_style_arm(td, prep); methods["ct7_top10_equal7"] = L.method_from_scores(d, L.answer_z(s, d.off))
    mats = [masked_answer_local(x, v) for x, v, _ in prep]            # answer-local axis, as T_C2 / T_E2
    ind = {m: [tail_indicator(mats[i], prep[i][2], spec) for i in range(d.n)] for m, spec in MARKS.items()}
    print("indicators built", flush=True)
    rules = ["equal", "cov_lsml"] + [f"tail_{m}_lsml" for m in MARKS]
    step = {(r, ro): np.full(total, np.nan) for r in rules for ro in READOUTS}
    for fold in range(5):
        t0 = time.perf_counter()
        train = np.flatnonzero(d.fold != fold); test = np.flatnonzero(d.fold == fold)
        std = fit_token_standardizer((mats[i][prep[i][2]] for i in train), cap=T.TOKEN_CAP)
        W = {"equal": None}
        W["cov_lsml"], meta = fit_l_sml_weights((mats[i][prep[i][2]] for i in train), std)
        fl = [{"rule": "cov_lsml", "weights": W["cov_lsml"].tolist(), "K": int(meta["K"]), "groups": np.asarray(meta["c"]).tolist()}]
        for m in MARKS:
            istd = fit_token_standardizer((ind[m][i][prep[i][2]] for i in train), cap=T.TOKEN_CAP)
            w, mt = fit_l_sml_weights((ind[m][i][prep[i][2]] for i in train), istd)
            W[f"tail_{m}_lsml"] = w
            fl.append({"rule": f"tail_{m}_lsml", "weights": w.tolist(), "K": int(mt["K"]), "groups": np.asarray(mt["c"]).tolist()})
        for i in test:
            a, b = d.off[i], d.off[i + 1]
            for r, w in W.items():
                fused_l, fused_e = fuse_token_matrix(mats[i], std, weights=w)
                fused = fused_e if w is None else fused_l
                for ro, k in READOUTS.items():
                    step[(r, ro)][a:b] = masked_step_top10(fused, prep[i][2], td.sp[i], k=k)
        fits[fold] = fl
        print(f"fold {fold} done ({time.perf_counter() - t0:.0f}s): " + ", ".join(f"{x['rule']} K={x['K']} w={np.round(np.abs(x['weights']) / np.abs(x['weights']).sum(), 2).tolist()}" for x in fl), flush=True)
    for (r, ro), s in step.items():
        methods[f"{r}_{ro}"] = L.method_from_scores(d, L.answer_z(s, d.off))
    return methods, fits


def planned(names):
    pairs = []
    def add(a, b, why):
        if a in names and b in names:
            pairs.append((a, b, why))
    for ro in READOUTS:
        for m in MARKS:
            add(f"tail_{m}_lsml_{ro}", f"cov_lsml_{ro}", "tail_minus_cov_lsml")
            add(f"tail_{m}_lsml_{ro}", f"equal_{ro}", "tail_minus_equal")
            add(f"tail_{m}_lsml_{ro}", "ct7", "tail_minus_ct7")
        add(f"cov_lsml_{ro}", f"equal_{ro}", "cov_lsml_minus_equal")
    add("equal_r30", "equal_r10", "readout30_minus_10_equal"); add("ct7_top10_equal7", "ct7", "after_readout_equal_minus_ct7")
    return pairs


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--config", required=True); p.add_argument("--draws", type=int)
    args = p.parse_args(); started = time.perf_counter()
    d = L.light_dataset(args.config); paths = d.c["paths"]; out = d.out
    L.run_freeze(out, [Path(__file__), Path(T.__file__), L.ROOT / "scripts/experiments/ct7_levers_common.py",
                       L.ROOT / "spectral_utils/ct7_token_streams.py", L.ROOT / "spectral_utils/claude_feature_bank_v1.py",
                       Path(args.config).resolve()],
                 [Path(paths[k]) for k in ("roster", "joined", "folds", "ct7", "ct7_tokens") if k in paths],
                 {"schema": SCHEMA, "candidate_id": d.c["candidate_id"], "development_only": True})
    td = T.TokenData(paths["ct7_tokens"], d)
    L.replay_ct7(d)
    methods, fits = build_methods(td)
    contrasts = planned(list(methods))
    result = L.evaluate_methods(d, methods, contrasts_extra=contrasts, strata_contrasts=contrasts, prmscore=True, draws=args.draws)
    rows = L.summary_rows(d, methods, result); L.write_summary_csv(out / "SUMMARY.csv", rows)
    L.dump(out / "RESULTS.json", {"schema": SCHEMA, "development_only": True, "note": L.DEVELOPMENT_NOTE, "streams": list(STREAMS),
                                  "marks": MARKS, "readouts": READOUTS, "fits": fits, "pb": result["pb"], "prm": result["prm"],
                                  "strata": result["strata"], "timing": result["timing"], "total_seconds": time.perf_counter() - started})
    print(f"{'method':28s} {'SLA':>7s} {'F1':>7s} {'within':>7s}")
    for r in rows:
        print(f"{r['method']:28s} {100*r['sla_macro8']:7.2f} {100*r['f1_common_gate']:7.2f} {r['within_auc']:7.4f}")
    for c in result["uncertainty"]["contrasts"]:
        print(f"{c['endpoint']:>12s} {c['a']:>26s} - {c['b']:<22s} {c['delta']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]")
    print("written:", out, f"({time.perf_counter() - started:.1f}s)")


if __name__ == "__main__":
    main()
