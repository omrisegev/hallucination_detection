#!/usr/bin/env python
"""The readout grid in the C1 architecture, where L-SML actually earns its keep.

The companion script runs this grid in CT7's architecture, which has **no L-SML at all**
-- CT7 is an equal-weight mean of seven views with no fitted parameter. Stage B showed
the L-SML advantage exists only when fusion happens at token level, BEFORE the step
readout (+3.32 pp there; +0.22 with an interval covering zero after it). So the readout
question has to be asked in C1 too, and that is what this does.

The L-SML fit is over token rows and never sees a step, so it does not depend on K at
all: the donor-fold standardizer and weights are fitted ONCE, the fused token series is
cached, and every readout variant is a sweep over that same series. One fit, twelve-plus
readouts, no repeated fitting and no extra route for a label to leak.

`equal` is carried beside `l_sml` throughout, because the whole point of C1 is that the
two differ there.

The grid deliberately runs past the plausible optimum -- K=80, K=160 and the full step
mean -- to test a specific worry about criterion 1. Split-half reproducibility rises
monotonically with K in the CT7-architecture run (43% at K=1 to 71% at K=40). A
criterion that only ever increases is not selecting anything; it would just name the
largest K in whatever grid it is offered. If reproducibility keeps climbing while
accuracy turns over, the criterion is monotone rather than diagnostic, and it must not
be reported as having found the optimum.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ROSTER = ROOT / "results" / "localization_full_benchmark_v3" / "evaluation"
RES = ROOT / "results" / "token_probability_fusion_v1"
OUT = RES / "READOUT_CALIBRATION_C1.json"
SUBS = ("gsm8k", "math", "olympiadbench", "omnimath")

_b = importlib.util.spec_from_file_location("stage_b", Path(__file__).with_name("stage_b_2x2_v1.py"))
SB = importlib.util.module_from_spec(_b)
_b.loader.exec_module(SB)
_r = importlib.util.spec_from_file_location("rc", Path(__file__).with_name("readout_calibration_v1.py"))
RC = importlib.util.module_from_spec(_r)
_r.loader.exec_module(RC)

from spectral_utils.digitfree_broad50 import masked_answer_standardize  # noqa: E402


def fused_token_series(matrices, folds, n_ans):
    """C1's token-level fusion: donor-fold standardizer + L-SML weights, fitted once."""
    l_sml = [None] * n_ans
    equal = [None] * n_ans
    for fold in np.unique(folds):
        train = np.flatnonzero(folds != fold)
        test = np.flatnonzero(folds == fold)
        std = SB.fit_token_standardizer((matrices[i] for i in train), cap=SB.TOKEN_CAP)
        w, _ = SB.fit_l_sml_weights((matrices[i] for i in train), std)
        for i in test:
            l_sml[i], equal[i] = SB.fuse_token_matrix(matrices[i], std, weights=w)
        print(f"  [fit] fold {fold}: {len(train)} donors", flush=True)
    return l_sml, equal


def step_readout_1d(series: np.ndarray, spans: np.ndarray, spec: dict):
    """(full, even-half, odd-half) step values for a single fused token series."""
    out = [np.zeros(len(spans)) for _ in range(3)]
    for s, (a, b) in enumerate(spans):
        block = series[a:b, None]
        n = len(block)
        if n == 0:
            continue
        out[0][s] = RC.topk_mean(block, RC.k_for(spec, n))[0]
        for j, half in enumerate((block[0::2], block[1::2]), start=1):
            h = half if len(half) else block
            out[j][s] = RC.topk_mean(h, RC.k_for(spec, len(h), scale=len(h) / n))[0]
    return out


def main() -> None:
    records = json.load(open(ROSTER / "JOINED.json", encoding="utf8"))["records"]
    with np.load(ROSTER / "JOINED.npz", allow_pickle=False) as z:
        offsets = np.asarray(z["offsets"], int)
        target = np.asarray(z["target"], int)
    cells = np.asarray([r["cell"] for r in records], str)
    sub = np.array([c[3:-3] if c.startswith("pb_") else "prm" for c in cells])
    outer = json.load(open(ROSTER / "FOLDS_V2.json", encoding="utf8"))["outer"]
    folds = np.asarray([outer[r["group_id"]] for r in records], int)
    with np.load(RES / "TOKEN_MATRICES.npz", allow_pickle=False) as z:
        tokens = z["tokens"].astype(float)
        tok_off = np.asarray(z["token_offsets"], int)
        step_spans = np.asarray(z["step_spans"], int)
    n_ans = len(records)
    log_tok = np.log(np.maximum((step_spans[:, 1] - step_spans[:, 0]).astype(float), 1.0))
    spans = [step_spans[offsets[i]:offsets[i + 1]] for i in range(n_ans)]
    mats = [tokens[tok_off[i]:tok_off[i + 1]] for i in range(n_ans)]

    print("[fit] token-level L-SML, once (independent of the readout)", flush=True)
    l_sml, equal = fused_token_series(mats, folds, n_ans)

    report = {"schema": "token-probability-fusion-v1-readout-calibration-c1",
              "architecture": "C1: token-level L-SML fusion, THEN the step readout",
              "variants": {}}

    for name, spec in RC.variants().items():
        print(f"[readout] {name}", flush=True)
        block: dict = {}
        for rule, series in (("l_sml", l_sml), ("equal", equal)):
            full = np.zeros(int(offsets[-1]))
            h1 = np.zeros_like(full)
            h2 = np.zeros_like(full)
            for i in range(n_ans):
                a, b = offsets[i], offsets[i + 1]
                f, x, y = step_readout_1d(series[i], spans[i], spec)
                full[a:b], h1[a:b], h2[a:b] = f, x, y
            std = lambda v: masked_answer_standardize(v[:, None], np.isfinite(v)[:, None], offsets)[:, 0]  # noqa: E731
            F, A, B = std(full), std(h1), std(h2)

            per = {}
            for s in SUBS:
                idx = np.flatnonzero(sub == s)
                hit, repro, marg, corr = [], [], [], []
                for i in idx:
                    a, b = offsets[i], offsets[i + 1]
                    if b - a < 2:
                        continue
                    v = F[a:b]
                    repro.append(int(np.argmax(A[a:b]) == np.argmax(B[a:b])))
                    o = np.sort(v)[::-1]
                    marg.append(float(o[0] - o[1]))
                    y = log_tok[a:b]
                    if v.std() > 0 and y.std() > 0:
                        corr.append(float(np.corrcoef(v, y)[0, 1]))
                for i in idx[target[idx] >= 0]:
                    a, b = offsets[i], offsets[i + 1]
                    hit.append(int(np.argmax(F[a:b]) == target[i]))
                per[s] = {"sla": float(np.mean(hit)),
                          "reproducibility": float(np.mean(repro)),
                          "margin": float(np.mean(marg)),
                          "abs_corr_log_tokens": float(abs(np.nanmean(corr)))}
            block[rule] = {"per_subset": per} | {
                k: float(np.mean([per[s][k] for s in SUBS]))
                for k in ("sla", "reproducibility", "margin", "abs_corr_log_tokens")}
        report["variants"][name] = block

    names = list(report["variants"])
    def pick(metric, rule="l_sml", lo=False):
        f = lambda n: report["variants"][n][rule][metric]  # noqa: E731
        return min(names, key=f) if lo else max(names, key=f)

    ceiling = pick("sla")
    picks = {"criterion1_reproducibility": pick("reproducibility"),
             "criterion2_margin": pick("margin"),
             "criterion3_length_decoupled": pick("abs_corr_log_tokens", lo=True)}
    report["selection"] = {
        "label_using_best": ceiling,
        "label_free_picks": picks,
        "cost_pp": {k: 100 * (report["variants"][ceiling]["l_sml"]["sla"]
                              - report["variants"][v]["l_sml"]["sla"]) for k, v in picks.items()},
        "reproducibility_is_monotone_in_K": None,
    }
    ks = [n for n in names if n.startswith("K=") and "max" not in n]
    rep = [report["variants"][n]["l_sml"]["reproducibility"] for n in ks]
    report["selection"]["reproducibility_is_monotone_in_K"] = bool(
        all(b >= a - 1e-12 for a, b in zip(rep, rep[1:])))

    OUT.write_text(json.dumps(report, indent=1), encoding="utf8")

    print()
    print("=" * 112)
    print("C1 ARCHITECTURE (token-level L-SML, then the readout) -- ceiling is LABEL-USING")
    print("=" * 112)
    print(f"{'readout':20s} {'L-SML':>7s} {'equal':>7s} {'L-SML-eq':>9s} | "
          f"{'gsm8k':>7s} {'olymp':>7s} {'omni':>7s} | {'repro':>7s} {'margin':>7s} {'|corr|':>7s}")
    for n in names:
        a, e = report["variants"][n]["l_sml"], report["variants"][n]["equal"]
        ps = a["per_subset"]
        star = " <<" if n == ceiling else ""
        print(f"{n:20s} {100*a['sla']:7.2f} {100*e['sla']:7.2f} {100*(a['sla']-e['sla']):9.2f} | "
              f"{100*ps['gsm8k']['sla']:7.2f} {100*ps['olympiadbench']['sla']:7.2f} "
              f"{100*ps['omnimath']['sla']:7.2f} | {100*a['reproducibility']:7.2f} "
              f"{a['margin']:7.3f} {a['abs_corr_log_tokens']:7.3f}{star}")
    print()
    print(f"  label-using best : {ceiling} ({100*report['variants'][ceiling]['l_sml']['sla']:.2f})")
    for k, v in picks.items():
        print(f"  {k:32s} picks {v:20s} -> {100*report['variants'][v]['l_sml']['sla']:6.2f} "
              f"(costs {report['selection']['cost_pp'][k]:.2f} pp)")
    print(f"\n  reproducibility monotone in K across the whole grid: "
          f"{report['selection']['reproducibility_is_monotone_in_K']}")
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
