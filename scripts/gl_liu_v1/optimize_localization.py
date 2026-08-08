#!/usr/bin/env python3
"""Frozen candidate study for native-scale localization features.

Candidate definitions are intentionally finite and mechanism-driven.  Scores are
constructed without labels.  The caller may select on the declared development
cells only; OlympiadBench and OmniMath are held-out dataset confirmation cells.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.stats import gaussian_kde, rankdata


ROOT = Path(__file__).resolve().parents[2]
LOC = Path(__file__).resolve().parent / "localization"
sys.path[:0] = [str(ROOT), str(LOC)]

from evidence_drop import EVIDENCE_FNS
from localization_metrics import evaluate, step_drop_scores
from positional_views import POSITIONAL_VIEWS, trace_series
from spectral_utils.streaming_utils import anchor_orient
from spectral_utils.upcr import upcr_fit


FIT = dict(
    loss="l2", exclusion=True, difficulty_gate=False,
    simple_avg_fallback=True, recompute_after_exclusion=True,
    g2_projection_k=1, scale_ratio=0.25,
)
MAX_FIT_TOKENS = 200_000

RAW_CORE = (
    "entropy_series", "sw_var_series", "cusum_abs_series",
    "sw_var_spilled_series", "cusum_abs_spilled_series",
)
RAW_FULL = tuple(POSITIONAL_VIEWS)


def load_rows(path):
    with open(path, "rb") as f:
        cache = pickle.load(f)
    return [cache[key] for key in sorted(cache) if not cache[key]["align_diag"]["problems"]]


def _positive_diff(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return values
    return np.maximum(np.diff(values, prepend=values[0]), 0.0)


def _mode_rank_transform(sample, values_by_row):
    sample = np.asarray(sample, dtype=float)
    sample = sample[np.isfinite(sample)]
    ordered = np.sort(sample)
    if len(ordered) < 50 or ordered.std() < 1e-12:
        centre = 0.5
    else:
        try:
            kde = gaussian_kde(ordered)
            grid = np.linspace(ordered.min(), ordered.max(), 512)
            mode_x = float(grid[int(np.argmax(kde(grid)))])
            centre = float(np.mean(ordered < mode_x))
        except Exception:
            centre = 0.5
    output = []
    for values in values_by_row:
        values = np.asarray(values, dtype=float)
        u = (np.searchsorted(ordered, values, side="right") - 0.5) / max(len(ordered), 1)
        output.append(np.abs(np.clip(u, 0.0, 1.0) - centre))
    return output, centre


def build_channels(rows, seed=0):
    raw = [trace_series(row["token_entropies"], row.get("token_spilled_energies"))
           for row in rows]
    channels = {name: [item[name] for item in raw] for name in RAW_FULL}
    for name in ("sw_var_series", "cusum_abs_series",
                 "sw_var_spilled_series", "cusum_abs_spilled_series"):
        channels["rise_" + name] = [_positive_diff(values) for values in channels[name]]

    # The mode-centred fold is the repository's transform of record.  Estimate
    # its centre from the same unlabeled fit population used by U-PCR.
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))
    chosen, total = [], 0
    for index in order:
        chosen.append(int(index))
        total += len(rows[int(index)]["token_entropies"])
        if total >= MAX_FIT_TOKENS:
            break
    mode_centres = {}
    for name in ("pe_series", "stft_high_series"):
        sample = np.concatenate([channels[name][index] for index in chosen])
        transformed, centre = _mode_rank_transform(sample, channels[name])
        channels["mode_" + name] = transformed
        mode_centres[name] = centre
    return channels, chosen, mode_centres


def fit_arm(channels, row_indices, names):
    """Fit a frozen two-pass U-PCR arm without accepting labels."""
    pool, mu, sd, cols = [], [], [], []
    for name in names:
        values = np.concatenate([channels[name][i] for i in row_indices]).astype(float)
        finite = np.isfinite(values)
        if not finite.any():
            continue
        median = float(np.median(values[finite]))
        values = np.where(finite, values, median)
        scale = float(values.std())
        if scale < 1e-8:
            continue
        pool.append(name)
        mu.append(float(values.mean()))
        sd.append(scale)
        cols.append((values - values.mean()) / scale)
    if len(pool) < 3:
        raise RuntimeError(f"only {len(pool)} usable views for {names}")
    V = np.column_stack(cols)
    first = upcr_fit(V.T, **FIT)
    derived = np.sign(first.rho_hat_full)
    derived[derived == 0] = 1.0
    F = (V * derived).T
    fitted = upcr_fit(F, **FIT)
    score = fitted.w @ F
    anchor = V[:, pool.index("entropy_series")] if "entropy_series" in pool else V[:, 0]
    _, flipped = anchor_orient(score, anchor)
    return {
        "pool": pool, "mu": np.asarray(mu), "sd": np.asarray(sd),
        "derived": derived, "w": fitted.w, "flipped": bool(flipped),
        "n_kept": int(fitted.keep.sum()),
    }


def apply_arm(arm, channels):
    by_row = []
    for row_index in range(len(next(iter(channels.values())))):
        cols = []
        for j, name in enumerate(arm["pool"]):
            values = np.asarray(channels[name][row_index], dtype=float)
            values = np.where(np.isfinite(values), values, arm["mu"][j])
            cols.append((values - arm["mu"][j]) / arm["sd"][j] * arm["derived"][j])
        score = arm["w"] @ np.vstack(cols)
        if arm["flipped"]:
            score = -score
        # All channel definitions, including entropy, are risk-oriented.  The
        # label-free anchor therefore makes this score risk-oriented already.
        by_row.append(score)
    return by_row


def aggregate_level(token_risk, rows, reducer="max"):
    output = []
    fn = np.max if reducer == "max" else np.mean
    for risk, row in zip(token_risk, rows):
        values = np.full(len(row["step_token_spans"]), np.nan)
        for i, span in enumerate(row["step_token_spans"]):
            if span is None:
                continue
            lo, hi = span
            segment = np.asarray(risk[lo:hi], dtype=float)
            segment = segment[np.isfinite(segment)]
            if len(segment):
                values[i] = float(fn(segment))
        output.append(values)
    return output


def aggregate_rise(token_risk, rows):
    return aggregate_level([_positive_diff(risk) for risk in token_risk], rows, "max")


def pooled_zscore(by_row):
    flat = np.concatenate(by_row)
    finite = np.isfinite(flat)
    mean = float(flat[finite].mean())
    sd = float(flat[finite].std())
    sd = sd if sd > 1e-12 else 1.0
    return [(np.asarray(row, float) - mean) / sd for row in by_row]


def blend(a, b, weight_a):
    za, zb = pooled_zscore(a), pooled_zscore(b)
    return [weight_a * x + (1.0 - weight_a) * y for x, y in zip(za, zb)]


def score_candidates(rows):
    """Return every frozen label-free candidate score and fit diagnostics."""
    channels, fit_rows, mode_centres = build_channels(rows)
    channel_sets = {
        "full": RAW_FULL,
        "core": RAW_CORE,
        "core_mode": RAW_CORE + ("mode_pe_series", "mode_stft_high_series"),
        "onset": (
            "entropy_series", "rise_sw_var_series", "rise_cusum_abs_series",
            "rise_sw_var_spilled_series", "rise_cusum_abs_spilled_series",
        ),
        "mixed": RAW_CORE + (
            "rise_sw_var_series", "rise_cusum_abs_series",
            "rise_sw_var_spilled_series", "rise_cusum_abs_spilled_series",
        ),
    }
    scores, diagnostics = {}, {"mode_centres": mode_centres, "arms": {}}
    token = {}
    for name, channel_names in channel_sets.items():
        arm = fit_arm(channels, fit_rows, channel_names)
        token[name] = apply_arm(arm, channels)
        diagnostics["arms"][name] = {
            "pool": arm["pool"], "n_kept": arm["n_kept"], "fit_rows": fit_rows,
        }

    scores["pos_full_max"] = aggregate_level(token["full"], rows, "max")
    scores["pos_core_max"] = aggregate_level(token["core"], rows, "max")
    scores["pos_core_mean"] = aggregate_level(token["core"], rows, "mean")
    scores["pos_core_rise"] = aggregate_rise(token["core"], rows)
    scores["pos_core_blend25"] = blend(
        aggregate_level(token["core"], rows, "max"), aggregate_rise(token["core"], rows), 0.75
    )
    scores["pos_core_blend50"] = blend(
        aggregate_level(token["core"], rows, "max"), aggregate_rise(token["core"], rows), 0.50
    )
    scores["pos_core_mode_max"] = aggregate_level(token["core_mode"], rows, "max")
    scores["pos_core_mode_blend25"] = blend(
        aggregate_level(token["core_mode"], rows, "max"),
        aggregate_rise(token["core_mode"], rows), 0.75,
    )
    scores["pos_onset_max"] = aggregate_level(token["onset"], rows, "max")
    scores["pos_mixed_max"] = aggregate_level(token["mixed"], rows, "max")
    scores["pos_mixed_blend25"] = blend(
        aggregate_level(token["mixed"], rows, "max"), aggregate_rise(token["mixed"], rows), 0.75
    )

    shannon = [step_drop_scores(
        EVIDENCE_FNS["shannon"](row, 20), row["step_token_spans"], ema_span=5
    ) for row in rows]
    scores["shannon_drop"] = shannon
    for positional in ("pos_core_max", "pos_core_mode_max", "pos_mixed_max"):
        for weight in (0.25, 0.50, 0.75):
            token_name = str(weight).replace(".", "p")
            scores[f"hybrid_{positional}_w{token_name}"] = blend(
                scores[positional], shannon, weight
            )
    scores["hybrid_core_shannon_max"] = [
        np.maximum(x, y) for x, y in zip(pooled_zscore(scores["pos_core_max"]),
                                         pooled_zscore(shannon))
    ]
    diagnostics["labels_seen_during_fit"] = False
    return scores, diagnostics


def run(path, out_dir):
    rows = load_rows(path)
    scores, diagnostics = score_candidates(rows)
    hashes = {name: hashlib.sha256(np.concatenate(value).astype("<f8").tobytes()).hexdigest()
              for name, value in scores.items()}

    # Evaluation starts only after every candidate score is frozen and hashed.
    labels = np.asarray([row["label"] for row in rows], dtype=int)
    subset = os.path.basename(path).removeprefix("processbench_").removesuffix(".pkl")
    records = []
    for method, values in scores.items():
        records.append({"subset": subset, "method": method,
                        **evaluate(values, labels, alpha=0.1, n_splits=100, seed=0)})
    records.sort(key=lambda row: row["f1"], reverse=True)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{subset}__localization_candidates.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(records[0]))
        writer.writeheader(); writer.writerows(records)
    with open(os.path.join(out_dir, f"{subset}__localization_candidates_diag.json"), "w") as f:
        json.dump({"score_hashes_before_evaluation": hashes,
                   "labels_used_only_after_scores_frozen": True,
                   "fit": diagnostics}, f, indent=2, default=str)
    print("\n", subset)
    for row in records[:10]:
        print(f"{row['method']:42s} F1={100*row['f1']:6.2f} "
              f"SLA={100*row['sla']:6.2f} SLA1={100*row['sla_tol1']:6.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--subsets", default=None)
    args = parser.parse_args()
    wanted = set(args.subsets.split(",")) if args.subsets else None
    for name in sorted(os.listdir(args.data_dir)):
        if name.startswith("processbench_") and name.endswith(".pkl"):
            subset = name.removeprefix("processbench_").removesuffix(".pkl")
            if wanted is not None and subset not in wanted:
                continue
            run(os.path.join(args.data_dir, name), args.out_dir)


if __name__ == "__main__":
    main()
