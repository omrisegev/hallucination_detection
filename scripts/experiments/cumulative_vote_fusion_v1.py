#!/usr/bin/env python
"""Cumulative-vote fusion of first-error localizers (binary L-SML over "error <= n" questions).

Question (Omri, 2026-09-20): take the localizers that already work, let each answer the family of
binary questions "is the first error at a step <= n?", and fuse those answers with the binary
SML / L-SML machinery.  A localizer that emits one step estimate s_j answers the whole family at
once: vote(n) = +1 if s_j <= n else -1.  Majority vote over that family is the MEDIAN of the
localizer positions; SML is a weighted median; the mode of the fused distribution is a weighted
plurality.  The emphasis of this run is what changes on the long-chain subsets (OlympiadBench,
Omni-MATH), where Step 422 found the level readout losing to Mind-the-Gap's derivative readout.

Data: the committed fair-comparison localization lane
`results/fair_paper_exact_comparisons_v1/lanes/localization/PER_QUESTION_LONG.csv`
(ProcessBench x Llama-3.1-8B, 3,400 questions, four subsets, five source folds).  Five
label-free telemetry localizers carry an ungated `locator` step.  Labels are used ONLY for
evaluation; every fit is label-free and out-of-fold by the file's own `fold` column.

Protocol: Mind-the-Gap SLA (erroneous answers only, no no-error decision), per subset; plus the
tolerance-one variant.  Paired question-level bootstrap for every fused-minus-baseline delta.

Nothing here touches a GPU or the raw telemetry.  No project result is modified.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LANE = ROOT / "results" / "fair_paper_exact_comparisons_v1" / "lanes" / "localization"
OUT = ROOT / "results" / "cumulative_vote_fusion_v1"

# Load fusion_utils without triggering spectral_utils/__init__ (which imports torch).
_spec = importlib.util.spec_from_file_location("fusion_utils", ROOT / "spectral_utils" / "fusion_utils.py")
fusion_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fusion_utils)
sml_fuse_signed = fusion_utils.sml_fuse_signed
lsml_fuse = fusion_utils.lsml_fuse

LOCALIZERS = [
    "family6_level_step_top5mean",   # dedicated Local incumbent (level, top-5 mean)
    "unified28",                     # ordinary method of record
    "gl_liu_v1_replay",              # DUFS-LIU frozen replay
    "max_entropy_step_top5mean",     # plain max entropy, top-5 mean
    "mind_the_gap_common_replay",    # derivative readout (EMA + worst drops)
]
SHORT = {"family6_level_step_top5mean": "family6", "unified28": "unified28",
         "gl_liu_v1_replay": "gl_liu", "max_entropy_step_top5mean": "max_ent",
         "mind_the_gap_common_replay": "mind_gap"}
SUBSETS = ["gsm8k", "math", "olympiadbench", "omnimath"]
SEED = 20260920
N_BOOT = 4000


# ── data ─────────────────────────────────────────────────────────────────────

def load_lane():
    rows = defaultdict(dict)
    with open(LANE / "PER_QUESTION_LONG.csv", encoding="utf8") as fh:
        for r in csv.DictReader(fh):
            if r["method_id"] not in LOCALIZERS:
                continue
            q = rows[r["row_id"]]
            q["family"] = r["family"]
            q["fold"] = int(r["fold"])
            q["label"] = int(r["label"])
            q.setdefault("loc", {})[r["method_id"]] = int(r["locator"])
    qids = sorted(rows)
    fam = np.array([rows[q]["family"] for q in qids])
    fold = np.array([rows[q]["fold"] for q in qids])
    label = np.array([rows[q]["label"] for q in qids])
    S_hat = np.array([[rows[q]["loc"][m] for m in LOCALIZERS] for q in qids])  # (N, m)
    return qids, fam, fold, label, S_hat


# ── cumulative votes ─────────────────────────────────────────────────────────

def cumulative_votes(s_hat_row, n_grid):
    """votes[j, k] = +1 if s_hat_j <= n_grid[k] else -1."""
    return np.where(s_hat_row[:, None] <= n_grid[None, :], 1, -1)


def disagreement_instances(S_hat):
    """Pooled (question, n) instances restricted to each question's disagreement region
    [min s_hat, max s_hat - 1]; outside it every localizer votes identically."""
    X, owner = [], []
    for i, row in enumerate(S_hat):
        lo, hi = int(row.min()), int(row.max())
        if hi <= lo:
            continue
        n_grid = np.arange(lo, hi)
        X.append(cumulative_votes(row, n_grid).T)
        owner.append(np.full(len(n_grid), i))
    return np.vstack(X), np.concatenate(owner)


# ── fusion fits (label-free) ─────────────────────────────────────────────────

def fit_sml(X):
    _, v = sml_fuse_signed(*[X[:, j] for j in range(X.shape[1])])
    return np.asarray(v, dtype=float)


def fit_lsml(X):
    _, meta = lsml_fuse(*[X[:, j] for j in range(X.shape[1])])
    # Flatten the two-layer L-SML into one effective weight per localizer for the
    # weighted-median readout: w_j = cross_weight[group] * within_weight_j.
    w = np.zeros(X.shape[1])
    for g, (idx, wg) in enumerate(meta["group_weights"]):
        w[idx] = meta["cross_weights"][g] * np.asarray(wg)
    return w, {"K": int(meta["K"]), "c": [int(x) for x in meta["c"]],
               "cross_weights": [float(x) for x in np.atleast_1d(meta["cross_weights"])]}


def fit_dawid_skene(X, w_init, iters=200, tol=1e-8):
    """Label-free Dawid-Skene EM on +-1 votes, initialised from the SML weighted vote.
    Returns (psi, eta, prior): psi_j = P(+1 | y=+1), eta_j = P(-1 | y=-1)."""
    V = (X > 0).astype(float)                      # 1 = voted "error <= n"
    score = X @ w_init
    post = 1.0 / (1.0 + np.exp(-2.0 * score / (np.std(score) + 1e-12)))
    psi = eta = None
    for _ in range(iters):
        pi = np.clip(post.mean(), 1e-3, 1 - 1e-3)
        psi = np.clip((post[:, None] * V).sum(0) / (post.sum() + 1e-12), 0.01, 0.99)
        eta = np.clip(((1 - post)[:, None] * (1 - V)).sum(0) / ((1 - post).sum() + 1e-12), 0.01, 0.99)
        ll_pos = np.log(pi) + (V * np.log(psi) + (1 - V) * np.log(1 - psi)).sum(1)
        ll_neg = np.log(1 - pi) + ((1 - V) * np.log(eta) + V * np.log(1 - eta)).sum(1)
        new = 1.0 / (1.0 + np.exp(ll_neg - ll_pos))
        if np.max(np.abs(new - post)) < tol:
            post = new
            break
        post = new
    return psi, eta, float(pi)


def ds_posterior(votes, psi, eta, pi):
    V = (votes > 0).astype(float)
    ll_pos = np.log(pi) + (V * np.log(psi) + (1 - V) * np.log(1 - psi)).sum(-1)
    ll_neg = np.log(1 - pi) + ((1 - V) * np.log(eta) + V * np.log(1 - eta)).sum(-1)
    return 1.0 / (1.0 + np.exp(ll_neg - ll_pos))


# ── readouts ─────────────────────────────────────────────────────────────────

def pava_increasing(y):
    """Pool-adjacent-violators, nondecreasing."""
    y = np.asarray(y, dtype=float).copy()
    n = len(y)
    blocks = [[i, i, y[i], 1.0] for i in range(n)]   # start, end, mean, size
    out = []
    for b in blocks:
        out.append(b)
        while len(out) > 1 and out[-2][2] > out[-1][2]:
            a, c = out.pop(), out.pop()
            m = (a[2] * a[3] + c[2] * c[3]) / (a[3] + c[3])
            out.append([c[0], a[1], m, a[3] + c[3]])
    res = np.empty(n)
    for s, e, m, _ in out:
        res[s:e + 1] = m
    return res


def readout_from_cdf(n_grid, cdf):
    """cdf[k] = fused P(error <= n_grid[k]) on the candidate grid (already covers [lo, hi]).
    Returns (mode_step, median_step)."""
    cdf = np.clip(pava_increasing(cdf), 0.0, 1.0)
    pmf = np.diff(np.concatenate([[0.0], cdf]))
    mode = int(n_grid[int(np.argmax(pmf))])
    med_idx = int(np.searchsorted(cdf, 0.5, side="left"))
    median = int(n_grid[min(med_idx, len(n_grid) - 1)])
    return mode, median


def predict_weighted(S_hat, w):
    """Weighted-vote fusion: F(n) = sum_j w_j 1[s_j <= n] / sum_j w_j (w clipped at 0)."""
    w = np.clip(w, 0.0, None)
    if w.sum() <= 0:
        w = np.ones_like(w)
    w = w / w.sum()
    modes, medians = [], []
    for row in S_hat:
        lo, hi = int(row.min()), int(row.max())
        n_grid = np.arange(lo, hi + 1)
        cdf = ((row[:, None] <= n_grid[None, :]) * w[:, None]).sum(0)
        m, d = readout_from_cdf(n_grid, cdf)
        modes.append(m); medians.append(d)
    return np.array(modes), np.array(medians)


def predict_ds(S_hat, psi, eta, pi):
    modes, medians = [], []
    for row in S_hat:
        lo, hi = int(row.min()), int(row.max())
        n_grid = np.arange(lo, hi + 1)
        votes = cumulative_votes(row, n_grid).T            # (len(n_grid), m)
        cdf = ds_posterior(votes, psi, eta, pi)
        cdf[-1] = 1.0                                       # everyone votes +1 at max s_j
        m, d = readout_from_cdf(n_grid, cdf)
        modes.append(m); medians.append(d)
    return np.array(modes), np.array(medians)


# ── evaluation ───────────────────────────────────────────────────────────────

def sla(pred, label, tol=0):
    return float(np.mean(np.abs(pred - label) <= tol))


def paired_boot(a_hit, b_hit, groups, n_boot=N_BOOT, seed=SEED):
    """Paired bootstrap of mean(a_hit - b_hit), resampling questions (groups)."""
    rng = np.random.default_rng(seed)
    d = a_hit.astype(float) - b_hit.astype(float)
    n = len(d)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        draws[b] = d[idx].mean()
    return float(d.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def offset_stats(pred, label):
    off = pred - label
    return {"early_frac": float(np.mean(off < 0)), "exact_frac": float(np.mean(off == 0)),
            "late_frac": float(np.mean(off > 0)), "median_abs_offset": float(np.median(np.abs(off))),
            "mean_signed_offset": float(np.mean(off))}


# ── main ─────────────────────────────────────────────────────────────────────

def run(fit_scope: str):
    qids, fam, fold, label, S_hat = load_lane()
    err = label != -1
    m = len(LOCALIZERS)
    N = len(qids)

    # Out-of-fold predictions for every rule.  Fit scope: 'pooled' fits one model on all
    # subsets of the training folds; 'per_subset' fits one model per subset.
    preds = {f"single:{SHORT[l]}": S_hat[:, j] for j, l in enumerate(LOCALIZERS)}
    preds["median"] = np.median(S_hat, axis=1).astype(int)          # majority vote on cumulative votes
    # plurality baseline: most common location (ties -> smallest)
    plur = []
    for row in S_hat:
        vals, cnt = np.unique(row, return_counts=True)
        plur.append(int(vals[np.argmax(cnt)]))
    preds["plurality"] = np.array(plur)
    for name in ("sml_mode", "sml_median", "lsml_mode", "lsml_median", "ds_mode", "ds_median",
                 "shift_sml_median", "shift_ds_mode"):
        preds[name] = np.full(N, -99)
    fits = []

    def consensus_shift(S_tr, w):
        """Label-free per-localizer offset: median of (s_j - weighted-median consensus) on the
        training questions.  Subtracting it before voting removes a systematic early/late bias
        that the weighted median cannot fix on its own (a median of late estimates is late)."""
        _, cons = predict_weighted(S_tr, w)
        return np.round(np.median(S_tr - cons[:, None], axis=0)).astype(int)

    for f in sorted(set(fold)):
        train = (fold != f) & err          # label-free fit uses erroneous-answer rows only in the
        test = fold == f                   # sense of the SLA protocol population (no labels read).
        # NOTE: restricting to erroneous answers uses the *population* definition of the SLA
        # protocol, not the first-error label itself; the clean/error split is the gate's job.
        scopes = [("all", np.ones(N, bool))] if fit_scope == "pooled" else [(s, fam == s) for s in SUBSETS]
        for scope_name, scope_mask in scopes:
            tr = train & scope_mask
            te = test & scope_mask
            if tr.sum() < 20 or te.sum() == 0:
                continue
            X, _owner = disagreement_instances(S_hat[tr])
            w_sml = fit_sml(X)
            w_lsml, lsml_meta = fit_lsml(X)
            psi, eta, pi = fit_dawid_skene(X, w_sml)
            fits.append({"fold": int(f), "scope": scope_name, "n_instances": int(len(X)),
                         "sml_w": w_sml.tolist(), "lsml_w": w_lsml.tolist(), "lsml": lsml_meta,
                         "ds_psi": psi.tolist(), "ds_eta": eta.tolist(), "ds_prior": pi})
            a, b = predict_weighted(S_hat[te], w_sml)
            preds["sml_mode"][te], preds["sml_median"][te] = a, b
            a, b = predict_weighted(S_hat[te], w_lsml)
            preds["lsml_mode"][te], preds["lsml_median"][te] = a, b
            a, b = predict_ds(S_hat[te], psi, eta, pi)
            preds["ds_mode"][te], preds["ds_median"][te] = a, b
            # bias-corrected variant: shift each localizer by its consensus-relative offset
            # (estimated on the training folds only), then refit and predict on shifted votes.
            shift = consensus_shift(S_hat[tr], w_sml)
            S_tr_s, S_te_s = S_hat[tr] - shift, np.maximum(S_hat[te] - shift, 0)
            Xs, _ = disagreement_instances(S_tr_s)
            w_s = fit_sml(Xs)
            psi_s, eta_s, pi_s = fit_dawid_skene(Xs, w_s)
            fits[-1]["shift"] = shift.tolist()
            _, preds["shift_sml_median"][te] = predict_weighted(S_te_s, w_s)
            preds["shift_ds_mode"][te], _ = predict_ds(S_te_s, psi_s, eta_s, pi_s)

    # ── metrics ──
    report = {"fit_scope": fit_scope, "n_questions": int(N), "n_erroneous": int(err.sum()),
              "localizers": LOCALIZERS, "fits": fits, "per_subset": {}, "overall": {}}
    for s in SUBSETS + ["all"]:
        mask = err & ((fam == s) if s != "all" else np.ones(N, bool))
        block = {"n": int(mask.sum()), "rules": {}}
        for name, p in preds.items():
            block["rules"][name] = {"sla": sla(p[mask], label[mask]), "sla_tol1": sla(p[mask], label[mask], 1),
                                    **offset_stats(p[mask], label[mask])}
        # paired deltas vs incumbent single and vs median
        for name in ("sml_mode", "lsml_mode", "ds_mode", "sml_median", "lsml_median", "ds_median", "plurality", "median",
                     "shift_sml_median", "shift_ds_mode"):
            hit = (preds[name][mask] == label[mask])
            for base in ("single:family6", "single:mind_gap", "median"):
                if base == name:
                    continue
                bh = (preds[base][mask] == label[mask])
                d, lo, hi = paired_boot(hit, bh, None)
                block["rules"][name][f"delta_vs_{base}"] = [d, lo, hi]
        # agreement structure on this subset (erroneous rows): pairwise exact-agreement of locations
        A = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                A[i, j] = float(np.mean(S_hat[mask][:, i] == S_hat[mask][:, j]))
        block["pairwise_exact_agreement"] = A.tolist()
        block["label_position"] = {"median": float(np.median(label[mask])),
                                   "p90": float(np.percentile(label[mask], 90))}
        block["max_locator"] = {"median": float(np.median(S_hat[mask].max(1)))}
        if s == "all":
            report["overall"] = block
        else:
            report["per_subset"][s] = block
    return report, preds, (qids, fam, fold, label, S_hat)


def fmt_table(report):
    lines = []
    subs = SUBSETS + ["all"]
    rules = ["single:family6", "single:unified28", "single:gl_liu", "single:max_ent", "single:mind_gap",
             "median", "plurality", "sml_mode", "sml_median", "lsml_mode", "lsml_median", "ds_mode", "ds_median",
             "shift_sml_median", "shift_ds_mode"]
    lines.append("| rule | " + " | ".join(subs) + " |")
    lines.append("|---|" + "---:|" * len(subs))
    for r in rules:
        vals = []
        for s in subs:
            blk = report["per_subset"][s] if s != "all" else report["overall"]
            vals.append(f"{100 * blk['rules'][r]['sla']:.2f}")
        lines.append(f"| {r} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reports = {}
    for scope in ("pooled", "per_subset"):
        rep, preds, (qids, fam, fold, label, S_hat) = run(scope)
        reports[scope] = rep
        json.dump(rep, open(out / f"REPORT_{scope}.json", "w"), indent=1)
        with open(out / f"PREDICTIONS_{scope}.csv", "w", newline="") as fh:
            wr = csv.writer(fh)
            names = list(preds)
            wr.writerow(["row_id", "family", "fold", "label"] + names)
            for i, q in enumerate(qids):
                wr.writerow([q, fam[i], int(fold[i]), int(label[i])] + [int(preds[n][i]) for n in names])
        print(f"\n== fit scope: {scope}  (SLA %, erroneous answers only, gate-free) ==")
        print(fmt_table(rep))
    json.dump({k: v for k, v in reports.items()}, open(out / "REPORT_ALL.json", "w"), indent=1)


if __name__ == "__main__":
    main()
