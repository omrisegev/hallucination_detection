#!/usr/bin/env python
"""Fitting scope, and the sw_var channel, in the architecture where L-SML matters.

Two questions from Omri, both run in the **C1 architecture** -- fuse the channels at
token level with L-SML, THEN take the Top-10 mean inside each step. Stage B showed that
is the only order in which L-SML beats equal weighting at all (+3.32 pp; after the
readout the advantage is +0.22 and its interval covers zero), so it is the right place
to ask whether a better-scoped fit can buy anything.

**Part 1 - who do we learn the weights from?** The standardizer and the L-SML weights
are fitted inside a scope instead of over everything:

  pooled_all         every answer (this is Stage B's C1; it must replay 35.92)
  per_model          only answers from the SAME model (PRMBench counts as Qwen3-8B)
  per_model_pb_only  same, but PRMBench never donates to a ProcessBench fit
  per_cell           only answers from the same cell (model x dataset)

Source folds are kept inside every scope, so a donor never shares a source question with
the answer it scores. The target is OlympiadBench and Omni-MATH, where we currently lose
6-8 pp to CT7 and up to 12.5 pp to the published comparator, so everything is reported
per subset and the headline contrast is the LONG group.

Note what the per-cell covariance screen already predicts: model identity moves the
marginal correlation by 0.040-0.045, which is at or barely above the within-cell
split-half baseline. If that screen is right, per_model should do very little. It is
run anyway, because the screen measures the covariance a fit would see, not the
accuracy a fit would deliver.

**Part 2 - sw_var as a twelfth channel.** `sw_var_peak` (Phase 3/4) was the project's
most robust single signal: a 16-token sliding window over the entropy series, the
variance inside each window, and the MAX over windows. That max is an answer-level
readout, and here we need a step-level one, so the adaptation keeps the rolling variance
as a token series and lets the existing Top-10 step readout aggregate it -- the step
readout then recovers a local version of the same peak. Two deliberate changes from the
original, both stated rather than hidden: the window is **causal** (trailing, not
forward) to match the bank's contract that every windowed channel is causal, and the max
is replaced by the step readout. The orientation is the declared prior from its original
use -- an unstable stretch is a suspect stretch -- and is not fitted to labels.

It is computed from the cached bank's channel 0, which IS the renormalized top-15
entropy, so no re-extraction is needed and the series is identical to the one the rest
of the bank sees.

Development-only.
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
OUT = RES / "FITTING_SCOPE_AND_SWVAR.json"

SW_WINDOW = 16
SW_SIGN = +1          # unstable stretch = suspect stretch; declared, not fitted
BOOTSTRAP_DRAWS = 10_000
SEED = 20260918
SHORT, LONG = ("gsm8k", "math"), ("olympiadbench", "omnimath")

_b = importlib.util.spec_from_file_location("stage_b", Path(__file__).with_name("stage_b_2x2_v1.py"))
SB = importlib.util.module_from_spec(_b)
_b.loader.exec_module(SB)


def causal_rolling_var(x: np.ndarray, window: int = SW_WINDOW) -> np.ndarray:
    """Variance of the trailing `window` values, expanding over the first few tokens."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return x
    cs = np.concatenate([[0.0], np.cumsum(x)])
    cs2 = np.concatenate([[0.0], np.cumsum(x * x)])
    i = np.arange(n)
    lo = np.maximum(0, i - window + 1)
    cnt = (i - lo + 1).astype(float)
    s = cs[i + 1] - cs[lo]
    s2 = cs2[i + 1] - cs2[lo]
    return np.maximum(s2 / cnt - (s / cnt) ** 2, 0.0)


def run_scoped(matrices, spans, offsets, folds, scope_of) -> dict[str, np.ndarray]:
    """C1 architecture, but the standardizer and L-SML weights are fitted inside a scope.

    Folds are preserved INSIDE each scope, so a donor is never from the scored answer's
    own source fold; narrowing the scope changes who donates, never whether the split is
    source-disjoint.
    """
    total = int(offsets[-1])
    out = {"l_sml": np.full(total, np.nan), "equal": np.full(total, np.nan)}
    for scope in np.unique(scope_of):
        in_scope = scope_of == scope
        for fold in np.unique(folds):
            train = np.flatnonzero(in_scope & (folds != fold))
            test = np.flatnonzero(in_scope & (folds == fold))
            if not len(test):
                continue
            if len(train) < 20:
                raise ValueError(f"scope {scope!r} fold {fold}: only {len(train)} donors")
            std = SB.fit_token_standardizer((matrices[i] for i in train), cap=SB.TOKEN_CAP)
            w, _ = SB.fit_l_sml_weights((matrices[i] for i in train), std)
            for i in test:
                fl, fe = SB.fuse_token_matrix(matrices[i], std, weights=w)
                a, b = int(offsets[i]), int(offsets[i + 1])
                out["l_sml"][a:b] = SB.step_top10(fl, spans[i])
                out["equal"][a:b] = SB.step_top10(fe, spans[i])
    for v in out.values():
        if not np.isfinite(v).all():
            raise ValueError("scoped run left unscored steps")
    return out


def group_means(per_cell: dict) -> tuple[float, float, float]:
    """Mean over the EIGHT cells, and over the short/long halves.

    Keyed by full cell name on purpose. Keying by subset (`cell[3:-3]`) silently
    collapses the q4 and q8 cells of each subset into one entry and keeps only the
    second, so every aggregate is computed on Qwen3-8B alone. That bug has now been
    written twice in this session; the anchor assertion below is what caught it.
    """
    vals = {k: v["sla"] for k, v in per_cell.items()}
    short = float(np.mean([v for k, v in vals.items() if k[3:-3] in SHORT]))
    lng = float(np.mean([v for k, v in vals.items() if k[3:-3] in LONG]))
    return float(np.mean(list(vals.values()))), short, lng


def main() -> None:
    records = json.load(open(ROSTER / "JOINED.json", encoding="utf8"))["records"]
    with np.load(ROSTER / "JOINED.npz", allow_pickle=False) as z:
        offsets = np.asarray(z["offsets"], int)
        target = np.asarray(z["target"], int)
    cells = np.asarray([r["cell"] for r in records], str)
    groups = np.asarray([r["group_id"] for r in records], str)
    outer = json.load(open(ROSTER / "FOLDS_V2.json", encoding="utf8"))["outer"]
    folds = np.asarray([outer[r["group_id"]] for r in records], int)
    pb = np.char.startswith(cells, "pb_")

    with np.load(RES / "TOKEN_MATRICES.npz", allow_pickle=False) as z:
        tokens = z["tokens"].astype(float)
        tok_off = np.asarray(z["token_offsets"], int)
        step_spans = np.asarray(z["step_spans"], int)

    n = len(records)
    spans = [step_spans[offsets[i]:offsets[i + 1]] for i in range(n)]
    bank11 = [tokens[tok_off[i]:tok_off[i + 1]] for i in range(n)]
    # channel 0 is the renormalized top-15 entropy, so sw_var comes straight off the cache
    bank12 = [np.column_stack([m, SW_SIGN * causal_rolling_var(m[:, 0])]) for m in bank11]

    model = np.array(["q4" if c.endswith("q4") else "q8" for c in cells])
    scopes = {
        "pooled_all": np.full(n, "all"),
        "per_model": model,
        "per_model_pb_only": np.array([f"{m}_pb" if p else "prm" for m, p in zip(model, pb)]),
        "per_cell": cells,
    }

    arms: dict[str, np.ndarray] = {}
    report: dict = {"schema": "token-probability-fusion-v1-fitting-scope-and-swvar",
                    "development_only": True, "sw_window": SW_WINDOW, "sw_sign": SW_SIGN,
                    "architecture": "C1: token-level L-SML fusion, then Top-10 per step",
                    "arms": {}}

    for scope_name, scope_of in scopes.items():
        print(f"[scope] {scope_name} (11 channels)", flush=True)
        res = run_scoped(bank11, spans, offsets, folds, scope_of)
        for rule in ("l_sml", "equal"):
            arms[f"bank11/{scope_name}/{rule}"] = res[rule]

    for scope_name in ("pooled_all", "per_model", "per_cell"):
        print(f"[scope] {scope_name} (12 channels, + sw_var)", flush=True)
        res = run_scoped(bank12, spans, offsets, folds, scopes[scope_name])
        for rule in ("l_sml", "equal"):
            arms[f"bank12/{scope_name}/{rule}"] = res[rule]

    peaks = {k: SB.peaks_of(v, offsets) for k, v in arms.items()}
    for k, p in peaks.items():
        per = SB.sla_gate_free(p, target, cells)
        mean, short, lng = group_means(per)
        report["arms"][k] = {"per_cell": per, "mean_sla": mean, "short": short, "long": lng}

    # Hard anchor: pooled_all in the C1 architecture IS Stage B's C1. If it does not
    # replay, something in the scoping or the aggregation is wrong and no other number
    # in this table can be trusted -- so stop rather than print.
    anchor = report["arms"]["bank11/pooled_all/l_sml"]["mean_sla"]
    expected = 0.3592368825176495  # Stage B's C1, same cache, same code path
    report["c1_anchor_replay"] = {"mean_sla": anchor, "expected": expected,
                                  "abs_diff": abs(anchor - expected)}
    if abs(anchor - expected) > 1e-9:
        raise SystemExit(f"ANCHOR FAILED: pooled_all replays {100*anchor:.4f} against Stage B's "
                         f"{100*expected:.4f}; no scope result written.")

    # ---- paired intervals against the pooled 11-channel arm, on LONG especially ----
    rng = np.random.default_rng(SEED)
    acc = {k: [] for k in arms}
    for i in SB._group_draws(groups, pb & (target >= 0), rng, BOOTSTRAP_DRAWS):
        t, c = target[i], cells[i]
        for k, p in peaks.items():
            _, s, l = group_means(SB.sla_gate_free(p[i], t, c))
            acc[k].append((s, l))
    draws = {k: np.asarray(v) for k, v in acc.items()}

    base = draws["bank11/pooled_all/l_sml"]
    report["vs_pooled_11ch_lsml"] = {
        k: {"short": SB.interval(draws[k][:, 0] - base[:, 0]),
            "long": SB.interval(draws[k][:, 1] - base[:, 1])}
        for k in arms if k != "bank11/pooled_all/l_sml"}

    OUT.write_text(json.dumps(report, indent=1), encoding="utf8")
    np.savez_compressed(RES / "FITTING_SCOPE_SCORES.npz", **arms)

    # ---------------------------------------------------------------- console
    print()
    print("=" * 100)
    print("PART 1 - WHO DONATES THE WEIGHTS?   (11 channels, C1 architecture)")
    print("=" * 100)
    print(f"{'arm':38s} {'mean':>7s} {'SHORT':>7s} {'LONG':>7s} {'vs pooled: LONG':>26s}")
    for scope in scopes:
        for rule in ("l_sml", "equal"):
            k = f"bank11/{scope}/{rule}"
            a = report["arms"][k]
            iv = report["vs_pooled_11ch_lsml"].get(k, {}).get("long")
            tail = (f"{iv['point_pp']:+6.2f} [{iv['ci95_pp'][0]:+6.2f},{iv['ci95_pp'][1]:+6.2f}]"
                    f"{'*' if iv['excludes_zero'] else ' '}") if iv else "   (reference)"
            print(f"{k:38s} {100*a['mean_sla']:7.2f} {100*a['short']:7.2f} {100*a['long']:7.2f} {tail:>26s}")
    print(f"\nC1 anchor replay: {100*anchor:.4f} vs {100*expected:.4f} expected "
          f"(|diff| {100*report['c1_anchor_replay']['abs_diff']:.4f} pp)")

    print()
    print("=" * 100)
    print("PART 2 - DOES sw_var (16-token causal rolling entropy variance) ADD ANYTHING?")
    print("=" * 100)
    print(f"{'arm':38s} {'mean':>7s} {'SHORT':>7s} {'LONG':>7s} {'vs pooled 11ch: LONG':>26s}")
    for scope in ("pooled_all", "per_model", "per_cell"):
        for rule in ("l_sml", "equal"):
            k = f"bank12/{scope}/{rule}"
            a = report["arms"][k]
            iv = report["vs_pooled_11ch_lsml"][k]["long"]
            print(f"{k:38s} {100*a['mean_sla']:7.2f} {100*a['short']:7.2f} {100*a['long']:7.2f} "
                  f"{iv['point_pp']:+6.2f} [{iv['ci95_pp'][0]:+6.2f},{iv['ci95_pp'][1]:+6.2f}]"
                  f"{'*' if iv['excludes_zero'] else ' '}")

    print()
    print("per-subset, the arms that matter (gate-free SLA %):")
    show = ["bank11/pooled_all/l_sml", "bank11/per_model/l_sml", "bank11/per_cell/l_sml",
            "bank12/pooled_all/l_sml", "bank12/per_cell/l_sml"]
    print(f"{'cell':24s}" + "".join(f"{s.split('/')[0][-2:]+'/'+s.split('/')[1][:9]:>16s}" for s in show))
    for cell in sorted(report["arms"][show[0]]["per_cell"]):
        print(f"{cell:24s}" + "".join(
            f"{100*report['arms'][s]['per_cell'][cell]['sla']:16.2f}" for s in show))
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
