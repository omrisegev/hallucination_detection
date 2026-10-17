#!/usr/bin/env python
"""Raw-channel readouts and cumulative-vote fusion (binary AND soft) for first-error localization.

Omri, 2026-09-20 (follow-up to Step 423): do not fuse the locators of existing algorithms; take the
eleven raw token channels of the Claude feature bank (q15_H1, q15_VE1, chosen_surprisal,
logprob_margin, true_tail50, energy_level, energy_innovation, top15_turnover, top50_js,
dominant_freq16, bocpd_p0), find for each the readout that suits it, then fuse the resulting
localizers with the binary cumulative-vote L-SML and with its SOFT version, and compare the two.

Pipeline per answer (teacher-forced telemetry row, the project's rich-save schema):
  1. X[t, j] = build_token_feature_matrix(row)         11 risk-oriented channels, causal.
  2. answer-local robust standardization per channel   (median / IQR within the answer).
  3. READOUTS turn a token series into a per-official-step profile r(s):
       top5, top10, max, mean        within-step reducers (top5 is the frozen incumbent readout)
       log_top5                      -LoG(sigma=2.5) delta detector, then top5 within step
       cusum_top5                    |CUSUM| of the standardized series, then top5
       onset80                       prefix profile: r(s)=top5(s) up to the FIRST step reaching
                                     80% of max top5, -inf after -> argmax is the onset step
     Each (channel, readout) is a localizer with s_hat = argmax r(s).
  4. Readout choice per channel: (a) fixed top5 for every channel (label-free control),
     (b) best SLA per channel on the TRAINING folds only (label-selected development choice).
  5. Fusion of the eleven localizers, out-of-fold:
       binary cumulative votes  v_j(n) = +1[s_hat_j <= n]     -> median / SML / L-SML / Dawid-Skene
       soft cumulative curves   F_j(n) = sum_{s<=n} softmax(z(r_j)/tau)  -> equal / SML / L-SML
     Both use the same (question, n) instances; the soft path degenerates to the binary path as
     tau -> 0 (checked numerically).
  6. Mind-the-Gap SLA protocol: erroneous answers only, no gate.  Labels enter only evaluation,
     the bootstrap, and the declared development readout choice (inside training folds).

Usage:
  python scripts/experiments/raw_channel_readout_fusion_v1.py \
      --cell gsm8k=path/to/processbench_gsm8k.pkl --cell math=... --out results/<dir> [--tau 1.0]

Small pilot caches are feasibility checks only (CLAUDE.md, 2026-09-07); full cells are needed for
any comparative claim.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.util
import json
import pathlib
import pickle
import sys
import types
from collections import defaultdict

import numpy as np
from scipy.ndimage import gaussian_laplace

ROOT = pathlib.Path(__file__).resolve().parents[2]

# spectral_utils/__init__ imports torch; register a bare package so the submodules import alone.
if "spectral_utils" not in sys.modules:
    _pkg = types.ModuleType("spectral_utils")
    _pkg.__path__ = [str(ROOT / "spectral_utils")]
    sys.modules["spectral_utils"] = _pkg
feature_bank = importlib.import_module("spectral_utils.claude_feature_bank_v1")
fusion_utils = importlib.import_module("spectral_utils.fusion_utils")

_spec = importlib.util.spec_from_file_location("cvf", pathlib.Path(__file__).with_name("cumulative_vote_fusion_v1.py"))
cvf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cvf)

CHANNELS = list(feature_bank.FEATURE_NAMES)
READOUTS = ["top5", "top10", "max", "mean", "log_top5", "cusum_top5", "onset80"]
LOG_SIGMA = 2.5
ONSET_FRACTION = 0.8
TIE_EPS = 1e-3          # in z-score units per step; negligible at tau >= 0.25, decisive at tau -> 0
SEED = 20260920
N_BOOT = 2000


# ── data ─────────────────────────────────────────────────────────────────────

def load_cell(path: pathlib.Path) -> list[dict]:
    with open(path, "rb") as fh:
        cache = pickle.load(fh)
    rows = []
    for key in sorted(cache, key=str):
        row = cache[key]
        if row.get("align_diag", {}).get("problems"):
            continue
        r = dict(row)
        r["_unit"] = str(row.get("id", key))
        rows.append(r)
    return rows


def stable_fold(unit: str, n_folds: int) -> int:
    return int(hashlib.sha256(unit.encode("utf8")).hexdigest(), 16) % n_folds


def robust_standardize(X: np.ndarray) -> np.ndarray:
    med = np.median(X, axis=0)
    q25, q75 = np.percentile(X, [25, 75], axis=0)
    scale = (q75 - q25) / 1.349
    std = X.std(axis=0)
    scale = np.where(scale > 1e-8, scale, np.where(std > 1e-8, std, 1.0))
    return (X - med) / scale


# ── readouts: token series -> per-step profile ───────────────────────────────

def _topk_in_step(series, spans, k):
    out = np.full(len(spans), np.nan)
    for i, span in enumerate(spans):
        if span is None:
            continue
        v = series[max(0, int(span[0])):min(len(series), int(span[1]))]
        v = v[np.isfinite(v)]
        if v.size:
            kk = min(k, v.size)
            out[i] = float(np.mean(np.partition(v, -kk)[-kk:]))
    return out


def _reduce_in_step(series, spans, fn):
    out = np.full(len(spans), np.nan)
    for i, span in enumerate(spans):
        if span is None:
            continue
        v = series[max(0, int(span[0])):min(len(series), int(span[1]))]
        v = v[np.isfinite(v)]
        if v.size:
            out[i] = float(fn(v))
    return out


def profiles_for_channel(z: np.ndarray, spans) -> dict[str, np.ndarray]:
    top5 = _topk_in_step(z, spans, 5)
    prof = {
        "top5": top5,
        "top10": _topk_in_step(z, spans, 10),
        "max": _reduce_in_step(z, spans, np.max),
        "mean": _reduce_in_step(z, spans, np.mean),
        "log_top5": _topk_in_step(-gaussian_laplace(z, LOG_SIGMA) if len(z) > 3 else z, spans, 5),
        "cusum_top5": _topk_in_step(np.abs(np.cumsum(z - z.mean())), spans, 5),
    }
    onset = np.full(len(spans), -np.inf)
    if np.isfinite(top5).any():
        thr = ONSET_FRACTION * np.nanmax(top5)
        first = int(np.argmax(np.nan_to_num(top5, nan=-np.inf) >= thr))
        onset[: first + 1] = top5[: first + 1]
    prof["onset80"] = onset
    return prof


def argmax_step(profile) -> int:
    p = np.asarray(profile, dtype=float)
    p = np.where(np.isfinite(p), p, -np.inf)
    return int(np.argmax(p)) if np.isfinite(p).any() else 0


# ── soft cumulative curves ───────────────────────────────────────────────────

def soft_cdf(profile: np.ndarray, tau: float) -> np.ndarray:
    """F(n) = sum_{s<=n} softmax(z(r)/tau); -inf entries get zero mass."""
    p = np.asarray(profile, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return np.linspace(1.0 / len(p), 1.0, len(p))
    z = np.full(len(p), -np.inf)
    v = p[finite]
    sd = v.std()
    z[finite] = (v - v.mean()) / sd if sd > 1e-8 else 0.0
    # Tie rule "earliest step wins", shared with argmax_step (np.argmax takes the first maximum):
    # discrete channels (turnover, JS, dominant frequency) tie often, and without this the
    # tau -> 0 limit would split the mass among tied steps instead of reproducing the binary vote.
    z = z - TIE_EPS * np.arange(len(p))
    w = np.exp((z - np.max(z[finite])) / max(tau, 1e-6))
    w[~finite] = 0.0
    w /= w.sum()
    return np.cumsum(w)


# ── fusion arms ──────────────────────────────────────────────────────────────

def binary_instances(S_hat):
    return cvf.disagreement_instances(S_hat)


def soft_instances(cdfs):
    """cdfs: list over questions of (m, S) arrays. Instances are (question, n) for n < S-1."""
    X = []
    for F in cdfs:
        S = F.shape[1]
        if S < 2:
            continue
        X.append((2.0 * F[:, : S - 1] - 1.0).T)
    return np.vstack(X)


def flatten_lsml(meta, m):
    w = np.zeros(m)
    for g, (idx, wg) in enumerate(meta["group_weights"]):
        w[idx] = np.atleast_1d(meta["cross_weights"])[g] * np.asarray(wg)
    return w


def fit_soft(X):
    """Label-free fits on continuous instances: SML (rank-one on continuous), L-SML continuous."""
    Z = (X - X.mean(0)) / np.where(X.std(0) > 1e-8, X.std(0), 1.0)
    _, w_sml = fusion_utils.sml_fuse_signed(*[Z[:, j] for j in range(Z.shape[1])])
    try:
        _, meta = fusion_utils.lsml_continuous(*[Z[:, j] for j in range(Z.shape[1])])
        w_lsml = flatten_lsml(meta, Z.shape[1])
        groups = [int(c) for c in meta["c"]]
    except Exception as exc:  # degenerate small-m cases
        w_lsml, groups = np.asarray(w_sml, float), [-1] * Z.shape[1]
    return np.asarray(w_sml, float), w_lsml, groups


def predict_soft(F, w):
    """F: (m, S) cdfs for one question; fused CDF = sum_j w_j F_j with w clipped >= 0."""
    w = np.clip(np.asarray(w, float), 0.0, None)
    if w.sum() <= 0:
        w = np.ones_like(w)
    w = w / w.sum()
    cdf = (w[:, None] * F).sum(0)
    grid = np.arange(F.shape[1])
    return cvf.readout_from_cdf(grid, cdf)


# ── evaluation helpers ───────────────────────────────────────────────────────

def sla(pred, label, tol=0):
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(label)) <= tol))


def paired_boot(hit_a, hit_b, seed=SEED, n_boot=N_BOOT):
    rng = np.random.default_rng(seed)
    d = np.asarray(hit_a, float) - np.asarray(hit_b, float)
    n = len(d)
    if n == 0:
        return [float("nan")] * 3
    draws = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return [float(d.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def offsets(pred, label):
    off = np.asarray(pred) - np.asarray(label)
    return {"early": float(np.mean(off < 0)), "exact": float(np.mean(off == 0)), "late": float(np.mean(off > 0)),
            "mean_offset": float(off.mean()) if len(off) else float("nan")}


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", action="append", required=True, help="subset=path/to/processbench_<subset>.pkl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1-3. features, standardization, profiles
    records = []
    skipped = defaultdict(int)
    for spec in args.cell:
        subset, path = spec.split("=", 1)
        for row in load_cell(pathlib.Path(path)):
            spans = row.get("step_token_spans") or []
            label = int(row.get("label", -1))
            if len(spans) < 2 or label >= len(spans):
                skipped[subset] += 1
                continue
            try:
                X = np.asarray(feature_bank.build_token_feature_matrix(row), dtype=float)
            except ValueError as exc:
                skipped[f"{subset}:{type(exc).__name__}"] += 1
                continue
            if X.shape[0] < 2:
                skipped[subset] += 1
                continue
            Z = robust_standardize(X)
            prof = {ch: profiles_for_channel(Z[:, j], spans) for j, ch in enumerate(CHANNELS)}
            records.append({"subset": subset, "unit": row["_unit"], "label": label, "n_steps": len(spans),
                            "n_tokens": int(X.shape[0]), "fold": stable_fold(row["_unit"], args.folds), "prof": prof})
    if not records:
        raise SystemExit(f"no usable rows; skipped={dict(skipped)}")
    subsets = sorted({r["subset"] for r in records})
    err = np.array([r["label"] != -1 for r in records])
    label = np.array([r["label"] for r in records])
    fam = np.array([r["subset"] for r in records])
    fold = np.array([r["fold"] for r in records])
    N = len(records)

    # Localizer positions for every (channel, readout)
    S_all = np.zeros((N, len(CHANNELS), len(READOUTS)), dtype=int)
    for i, r in enumerate(records):
        for j, ch in enumerate(CHANNELS):
            for k, rd in enumerate(READOUTS):
                S_all[i, j, k] = argmax_step(r["prof"][ch][rd])

    # 4a. grid: SLA and offsets per (channel, readout) per subset, erroneous only (descriptive)
    grid = {}
    for s in subsets + ["all"]:
        m = err & ((fam == s) if s != "all" else np.ones(N, bool))
        grid[s] = {"n": int(m.sum()), "cells": {}}
        for j, ch in enumerate(CHANNELS):
            for k, rd in enumerate(READOUTS):
                p = S_all[m, j, k]
                grid[s]["cells"][f"{ch}/{rd}"] = {"sla": sla(p, label[m]), "sla_tol1": sla(p, label[m], 1), **offsets(p, label[m])}

    # 5. fusion, out-of-fold.  Two rosters: fixed top5 (label-free) and per-channel best readout
    #    chosen on the training folds (label-selected development).
    rosters = {"fixed_top5": None, "train_selected": None}
    preds = defaultdict(lambda: np.full(N, -99))
    fits = []
    top5_idx = READOUTS.index("top5")
    for f in sorted(set(fold)):
        tr = (fold != f) & err
        te = fold == f
        if tr.sum() < 10 or te.sum() == 0:
            continue
        for roster in rosters:
            if roster == "fixed_top5":
                choice = np.full(len(CHANNELS), top5_idx)
            else:
                choice = np.array([int(np.argmax([sla(S_all[tr, j, k], label[tr]) for k in range(len(READOUTS))]))
                                   for j in range(len(CHANNELS))])
            S_tr = np.stack([S_all[tr, j, choice[j]] for j in range(len(CHANNELS))], 1)
            S_te = np.stack([S_all[te, j, choice[j]] for j in range(len(CHANNELS))], 1)
            # singles
            for j, ch in enumerate(CHANNELS):
                preds[f"{roster}:single:{ch}"][te] = S_te[:, j]
            # ── binary ──
            Xb, _ = binary_instances(S_tr)
            w_sml = cvf.fit_sml(Xb)
            w_lsml, lsml_meta = cvf.fit_lsml(Xb)
            psi, eta, pi = cvf.fit_dawid_skene(Xb, w_sml)
            preds[f"{roster}:bin:median"][te] = np.median(S_te, axis=1).astype(int)
            _, preds[f"{roster}:bin:sml_median"][te] = cvf.predict_weighted(S_te, w_sml)
            preds[f"{roster}:bin:lsml_mode"][te], _ = cvf.predict_weighted(S_te, w_lsml)
            preds[f"{roster}:bin:ds_mode"][te], _ = cvf.predict_ds(S_te, psi, eta, pi)
            # ── soft ──
            tr_idx, te_idx = np.where(tr)[0], np.where(te)[0]
            cdfs_tr = [np.stack([soft_cdf(records[i]["prof"][ch][READOUTS[choice[j]]], args.tau) for j, ch in enumerate(CHANNELS)])
                       for i in tr_idx]
            cdfs_te = [np.stack([soft_cdf(records[i]["prof"][ch][READOUTS[choice[j]]], args.tau) for j, ch in enumerate(CHANNELS)])
                       for i in te_idx]
            Xs = soft_instances(cdfs_tr)
            ws_sml, ws_lsml, groups_soft = fit_soft(Xs)
            for name, w in (("soft:equal", np.ones(len(CHANNELS))), ("soft:sml", ws_sml), ("soft:lsml", ws_lsml)):
                modes, meds = zip(*[predict_soft(F, w) for F in cdfs_te]) if cdfs_te else ([], [])
                preds[f"{roster}:{name}_mode"][te] = np.array(modes, dtype=int)
                preds[f"{roster}:{name}_median"][te] = np.array(meds, dtype=int)
            # tau -> 0 identity check: soft equal median must equal the binary median
            cdfs_te0 = [np.stack([soft_cdf(records[i]["prof"][ch][READOUTS[choice[j]]], 1e-4) for j, ch in enumerate(CHANNELS)])
                        for i in te_idx]
            med0 = np.array([predict_soft(F, np.ones(len(CHANNELS)))[1] for F in cdfs_te0], dtype=int)
            tau0_match = float(np.mean(med0 == preds[f"{roster}:bin:median"][te])) if len(med0) else float("nan")
            fits.append({"fold": int(f), "roster": roster, "readout_choice": [READOUTS[c] for c in choice],
                         "n_bin_instances": int(len(Xb)), "n_soft_instances": int(len(Xs)),
                         "bin_sml_w": w_sml.tolist(), "bin_lsml": lsml_meta, "ds_psi": psi.tolist(), "ds_eta": eta.tolist(),
                         "soft_sml_w": ws_sml.tolist(), "soft_lsml_w": ws_lsml.tolist(), "soft_lsml_groups": groups_soft,
                         "tau0_identity_match": tau0_match})

    # 6. tables
    report = {"tau": args.tau, "n_records": int(N), "n_erroneous": int(err.sum()), "skipped": dict(skipped),
              "channels": CHANNELS, "readouts": READOUTS, "grid": grid, "fits": fits, "fusion": {}}
    ref_name = "fixed_top5:single:q15_H1"      # max-entropy top-5, the transparent incumbent analogue
    for s in subsets + ["all"]:
        m = err & ((fam == s) if s != "all" else np.ones(N, bool))
        blk = {"n": int(m.sum()), "rules": {}}
        for name, p in preds.items():
            if (p[m] == -99).any():
                continue
            entry = {"sla": sla(p[m], label[m]), "sla_tol1": sla(p[m], label[m], 1), **offsets(p[m], label[m])}
            if name != ref_name and ref_name in preds:
                entry["delta_vs_ref"] = paired_boot(p[m] == label[m], preds[ref_name][m] == label[m])
            blk["rules"][name] = entry
        report["fusion"][s] = blk
    json.dump(report, open(out / "REPORT.json", "w"), indent=1)

    with open(out / "PREDICTIONS.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        names = sorted(preds)
        wr.writerow(["unit", "subset", "fold", "label", "n_steps", "n_tokens"] + names)
        for i, r in enumerate(records):
            wr.writerow([r["unit"], r["subset"], r["fold"], r["label"], r["n_steps"], r["n_tokens"]] + [int(preds[n][i]) for n in names])

    # markdown
    md = [f"# Raw-channel readouts + cumulative-vote fusion (binary vs soft), tau={args.tau}\n",
          f"records={N}, erroneous={int(err.sum())}, skipped={dict(skipped)}\n",
          "## Readout grid (SLA %, erroneous only; late fraction in parentheses) — descriptive\n"]
    for s in subsets + ["all"]:
        md.append(f"\n### {s} (n={grid[s]['n']})\n")
        md.append("| channel | " + " | ".join(READOUTS) + " |")
        md.append("|---|" + "---:|" * len(READOUTS))
        for ch in CHANNELS:
            md.append(f"| {ch} | " + " | ".join(
                f"{100 * grid[s]['cells'][f'{ch}/{rd}']['sla']:.1f} ({grid[s]['cells'][f'{ch}/{rd}']['late']:.2f})" for rd in READOUTS) + " |")
    md.append("\n## Fusion: binary vs soft (SLA %, erroneous only, out-of-fold)\n")
    rules = [f"{ro}:{r}" for ro in rosters for r in
             ("single:q15_H1", "bin:median", "bin:sml_median", "bin:lsml_mode", "bin:ds_mode",
              "soft:equal_median", "soft:equal_mode", "soft:sml_median", "soft:sml_mode", "soft:lsml_median", "soft:lsml_mode")]
    md.append("| rule | " + " | ".join(subsets + ["all"]) + " | late (all) | Δ vs q15_H1/top5 (all) |")
    md.append("|---|" + "---:|" * (len(subsets) + 1) + "---:|---|")
    for r in rules:
        if r not in report["fusion"]["all"]["rules"]:
            continue
        cells = [f"{100 * report['fusion'][s]['rules'][r]['sla']:.1f}" if r in report["fusion"][s]["rules"] else "—" for s in subsets + ["all"]]
        e = report["fusion"]["all"]["rules"][r]
        d = e.get("delta_vs_ref")
        ds = f"{100 * d[0]:+.1f} [{100 * d[1]:+.1f}, {100 * d[2]:+.1f}]" if d else "—"
        md.append(f"| {r} | " + " | ".join(cells) + f" | {e['late']:.2f} | {ds} |")
    md.append("\n## Fits\n")
    for fdict in fits:
        md.append(f"- fold {fdict['fold']} {fdict['roster']}: readouts={fdict['readout_choice']}; "
                  f"bin SML w={np.round(fdict['bin_sml_w'], 2).tolist()}; soft SML w={np.round(fdict['soft_sml_w'], 2).tolist()}; "
                  f"soft L-SML groups={fdict['soft_lsml_groups']}; tau->0 identity={fdict['tau0_identity_match']:.2f}")
    (out / "REPORT.md").write_text("\n".join(md) + "\n", encoding="utf8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
