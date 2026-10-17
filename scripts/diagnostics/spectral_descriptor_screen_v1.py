#!/usr/bin/env python
"""Is there ANY information in the per-answer spectra about which readout width to use?

Omri's question: can the right K be read off the L-SML's own spectral quantities -- its
residual, the K its clustering picks, the eigenvalues of the covariance?

A trap first. Our L-SML is fitted on donor folds, so its residual, its group count and
its covariance spectrum are properties of a FOLD, not of an answer, and cannot vary per
answer by construction. Anything per-answer has to be recomputed from that answer's own
tokens -- which is possible: a 500-token answer easily supports an 11x11 covariance, and
an answer-local L-SML fit takes about 0.3 s and succeeds on every answer tried.

This is a **screen, not a selector.** Before building anything, ask whether the
descriptors carry information about the right K at all. The test is deliberately the
narrowest decision-relevant one:

    among answers where K=10 and K=40 DISAGREE about being right,
    can any label-free descriptor tell which of the two is correct?

Chance is 0.5. If no descriptor beats it, this whole family is closed and no selector
should be built. The size of the disagreeing set also bounds what any such selector
could ever be worth.

Descriptors, all computed from the answer's own tokens with no labels:
  spectral   participation ratio, top-1 eigenvalue share, entropy effective rank and
             log condition number of the answer's 11x11 channel correlation
  L-SML      group count K and residual from an answer-local L-SML fit
  criterion  the adaptive K* and peak SNR from the variance decomposition
  structural step count, tokens per step, their spread, total tokens
  shape      the top1-top2 margin of the fused step scores at K=10 and at K=40
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ROSTER = ROOT / "results" / "localization_full_benchmark_v3" / "evaluation"
RES = ROOT / "results" / "token_probability_fusion_v1"
OUT = RES / "SPECTRAL_DESCRIPTOR_SCREEN.json"
SEED = 20260919

_b = importlib.util.spec_from_file_location("stage_b", Path(__file__).with_name("stage_b_2x2_v1.py"))
SB = importlib.util.module_from_spec(_b)
_b.loader.exec_module(SB)

from spectral_utils.adaptive_step_readout_v1 import (  # noqa: E402
    choose_k, ladder_for, step_scores_over_ladder,
)


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, bool)
    s = np.asarray(scores, float)
    ok = np.isfinite(s)
    labels, s = labels[ok], s[ok]
    n1, n0 = int(labels.sum()), int((~labels).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = np.argsort(np.argsort(s)) + 1
    return float((r[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def spectral_descriptors(matrix: np.ndarray) -> dict[str, float]:
    x = np.asarray(matrix, float)
    keep = x.std(axis=0) > 1e-12
    if keep.sum() < 2:
        return {k: float("nan") for k in
                ("participation_ratio", "top1_share", "eff_rank_entropy", "log_condition")}
    c = np.corrcoef(x[:, keep].T)
    lam = np.maximum(np.linalg.eigvalsh(c), 0.0)
    total = lam.sum()
    p = lam / max(total, 1e-12)
    nz = p[p > 1e-12]
    return {
        "participation_ratio": float(total ** 2 / max((lam ** 2).sum(), 1e-12)),
        "top1_share": float(lam.max() / max(total, 1e-12)),
        "eff_rank_entropy": float(np.exp(-(nz * np.log(nz)).sum())),
        "log_condition": float(np.log(max(lam.max(), 1e-12) / max(lam.min(), 1e-12))),
    }


def main() -> None:
    records = json.load(open(ROSTER / "JOINED.json", encoding="utf8"))["records"]
    with np.load(ROSTER / "JOINED.npz", allow_pickle=False) as z:
        offsets = np.asarray(z["offsets"], int)
        target = np.asarray(z["target"], int)
    cells = np.asarray([r["cell"] for r in records], str)
    outer = json.load(open(ROSTER / "FOLDS_V2.json", encoding="utf8"))["outer"]
    folds = np.asarray([outer[r["group_id"]] for r in records], int)
    pb = np.char.startswith(cells, "pb_")
    with np.load(RES / "TOKEN_MATRICES.npz", allow_pickle=False) as z:
        tokens = z["tokens"].astype(float)
        tok_off = np.asarray(z["token_offsets"], int)
        step_spans = np.asarray(z["step_spans"], int)
    n = len(records)
    spans = [step_spans[offsets[i]:offsets[i + 1]] for i in range(n)]
    mats = [tokens[tok_off[i]:tok_off[i + 1]] for i in range(n)]

    print("[fit] token-level L-SML on donor folds, once", flush=True)
    fused = [None] * n
    for fold in np.unique(folds):
        train = np.flatnonzero(folds != fold)
        std = SB.fit_token_standardizer((mats[i] for i in train), cap=SB.TOKEN_CAP)
        w, _ = SB.fit_l_sml_weights((mats[i] for i in train), std)
        for i in np.flatnonzero(folds == fold):
            fused[i], _ = SB.fuse_token_matrix(mats[i], std, weights=w)

    # ---- who does K=10 and K=40 disagree about? -------------------------------
    eligible = np.flatnonzero(pb & (target >= 0))
    hit10, hit40, margins = {}, {}, {}
    for i in eligible:
        sp = spans[i]
        if len(sp) < 2:
            continue
        lad = ladder_for(int((sp[:, 1] - sp[:, 0]).max()))
        grid = step_scores_over_ladder(fused[i], sp, lad)
        for k, store in ((10, hit10), (40, hit40)):
            col = grid[:, int(np.argmin(np.abs(lad - k)))]
            store[i] = int(np.argmax(col) == target[i])
            o = np.sort(col)[::-1]
            margins.setdefault(i, {})[f"margin_K{k}"] = float(o[0] - o[1])
    ids = np.array([i for i in eligible if i in hit10])
    # dtype=bool is load-bearing: hit10[i] is an int, so `x and not y` yields a mix of
    # bool and int, np.array infers an INTEGER array, and ids[a | b] then becomes fancy
    # indexing by position instead of a boolean mask -- which silently "finds" a 100%
    # disagreement rate.
    a = np.array([bool(hit10[i]) and not bool(hit40[i]) for i in ids], dtype=bool)
    b = np.array([bool(hit40[i]) and not bool(hit10[i]) for i in ids], dtype=bool)
    switch = ids[a | b]
    if switch.size and (a & b).any():
        raise AssertionError("an answer cannot be in both disagreement classes")
    y = np.array([bool(hit40[i]) for i in switch])            # True => K=40 is the right one

    print(f"eligible erroneous answers : {len(ids)}")
    print(f"K=10 and K=40 disagree on  : {len(switch)}  "
          f"({100*len(switch)/len(ids):.1f}%)  -- K=40 right in {100*y.mean():.1f}% of them")

    # ---- descriptors, only where the decision actually switches ----------------
    rng = np.random.default_rng(SEED)
    rows: dict[str, list[float]] = {}
    started = time.time()
    for j, i in enumerate(switch):
        m = mats[i]
        sp = spans[i]
        d = spectral_descriptors(m)
        try:
            std = SB.fit_token_standardizer([m], cap=SB.TOKEN_CAP)
            _, meta = SB.fit_l_sml_weights([m], std)
            d["lsml_group_count"] = float(meta["K"])
            d["lsml_residual"] = float(meta["residual"])
        except Exception:
            d["lsml_group_count"] = float("nan")
            d["lsml_residual"] = float("nan")
        lad = ladder_for(int((sp[:, 1] - sp[:, 0]).max()))
        idx, snr = choose_k(fused[i], sp, lad, rng)
        d["adaptive_K_star"] = float(lad[idx])
        d["peak_snr"] = float(snr[idx])
        lengths = (sp[:, 1] - sp[:, 0]).astype(float)
        d["n_steps"] = float(len(sp))
        d["mean_tokens_per_step"] = float(lengths.mean())
        d["sd_tokens_per_step"] = float(lengths.std())
        d["log_total_tokens"] = float(np.log(len(m)))
        d |= margins[i]
        for k, v in d.items():
            rows.setdefault(k, []).append(v)
        if (j + 1) % 200 == 0:
            r = (j + 1) / (time.time() - started)
            print(f"  {j+1}/{len(switch)}  eta {(len(switch)-j-1)/r/60:.1f} min", flush=True)

    report = {"schema": "token-probability-fusion-v1-spectral-descriptor-screen",
              "label_using_screen": True,
              "eligible": int(len(ids)), "switchable": int(len(switch)),
              "switchable_fraction": float(len(switch) / len(ids)),
              "k40_right_share": float(y.mean()),
              "auc": {}}
    for name, vals in rows.items():
        report["auc"][name] = auc(y, np.asarray(vals))

    OUT.write_text(json.dumps(report, indent=1), encoding="utf8")

    print()
    print("=" * 78)
    print("CAN A LABEL-FREE DESCRIPTOR TELL WHICH READOUT IS RIGHT? (chance = 0.500)")
    print("=" * 78)
    for name, v in sorted(report["auc"].items(), key=lambda kv: -abs(kv[1] - 0.5)):
        flag = "  <-- informative" if abs(v - 0.5) > 0.05 else ""
        print(f"  {name:26s} AUC {v:.3f}   |AUC-.5| {abs(v-0.5):.3f}{flag}")
    best = max(report["auc"].values(), key=lambda v: abs(v - 0.5))
    print()
    print(f"  strongest |AUC-0.5| = {abs(best-0.5):.3f}")
    print(f"  the switchable set is {100*report['switchable_fraction']:.1f}% of erroneous answers,")
    print(f"  so a PERFECT selector between these two widths would be worth about "
          f"{100*report['switchable_fraction']*abs(report['k40_right_share']-0.5):.2f} pp.")
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
