#!/usr/bin/env python3
"""
scripts/rag_ec_v1/run.py — Evidence-Contrast U-PCR/DUFS-LIU evaluator for RAGTruth.

Six primary arms + two quoted ceilings, per `cluster/manifests/ragtruth_ec_v1.json`'s
`frozen_arm_roster` and `docs/research_notes/ragtruth_ec_preregistration_v1.md`. Every arm
consumes the SAME teacher-forced conditional-rescore cache
(`cluster/run_conditional_rescore.py`'s output) and is fit with NO LABELS — `response_label`
/ `span_labels` are read only in `evaluate_cell`, after every score is hashed.

REUSED LOW-LEVEL PRIMITIVES (unchanged, imported from spectral_utils — the same functions
`scripts/gl_liu_v1/run.py` uses for ProcessBench):
    spectral_utils.upcr.upcr_fit                  — the deployed U-PCR solver
    spectral_utils.laplacian_upcr.build_graph_from_features / laplacian_iu_path
    spectral_utils.adapted_dufs.adapted_dufs_soft_gates
    spectral_utils.streaming_utils.anchor_orient
    spectral_utils.token_feature_views.token_feature_views / CORE_TOKEN_VIEWS

NOT reused: `scripts/gl_liu_v1/run.py`'s own wrapper functions (`fit_answer_mixed`,
`fit_token_laplacian`, `token_scores`, ...) — those are hard-coupled to ProcessBench's
`trace_features`/`CONFIDENCE_FEATURE_SIGNS_V1` contract (step boundaries, math-answer
features) which has no RAGTruth analogue. This module re-derives the same FUSION MECHANISM
(temporal-chain graph over tokens, DUFS soft gates, Laplacian-IU-PCR path) on top of
RAGTruth's own feature set — the "tailor, never transplant" convention
(`feedback_tailor_not_transplant`): same algorithmic core, new feature contract.

ARM 5b — THE NEW GRAPH CONSTRUCTION, MECHANICAL INTERPRETATION (flag for confirmation)
--------------------------------------------------------------------------------------
The preregistration describes arm 5b's graph only at the concept level: "nodes =
intervention views (full, noctx, each loo_j); edges = TF-IDF chunk-text similarity."
`spectral_utils.evidence_contrast.build_evidence_graph` builds exactly that per-response
graph, but `laplacian_iu_path` fuses FEATURES across one graph shared by every SAMPLE
(token or response) — it has no notion of a graph over one response's own conditions. This
module's operationalization: before reducing a response's K per-chunk quantities (mean
spilled-energy delta, mean JSD) to the `_max`/`_mean` aggregate Δ-views, Laplacian-smooth
them through the response's own evidence graph — closed form
`s* = (I + lambda_graph * L)^{-1} v` where `L` is the graph Laplacian and `v` the raw
per-chunk values — so textually-redundant chunks (near-duplicate Data2txt fields, say) get
pooled instead of each independently inflating a max-based feature. The smoothed Δ-views
then enter the SAME temporal-chain DUFS-LIU token fusion as arm 5, so 5 vs 5b differ ONLY in
whether the LOO-derived Δ features were evidence-graph-smoothed first — a clean ablation of
the graph's effect. This is ONE reasonable reading of the preregistration text, not
verified against Omri's own mental model of the mechanism; flag before treating 5b's
numbers as a confirmed test of the "new Laplacian construction" idea.
"""
from __future__ import annotations

import argparse
import hashlib
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from spectral_utils.adapted_dufs import adapted_dufs_soft_gates
from spectral_utils.evidence_contrast import build_evidence_graph, response_delta_views
from spectral_utils.laplacian_upcr import build_graph_from_features, laplacian_iu_path
from spectral_utils.streaming_utils import anchor_orient
from spectral_utils.token_feature_views import CORE_TOKEN_VIEWS, token_feature_views
from spectral_utils.upcr import upcr_fit
from spectral_utils.ragtruth import chunk_source, load_ragtruth

import gasp as gasp_mod

K = 7
LAMBDAS = (0.03, 0.1, 0.3)
DUFS_SEEDS = (11, 23, 37)
DUFS_EPOCHS = 80
GRAPH_SMOOTH_LAMBDA = 0.3  # arm 5b's per-response evidence-graph smoothing strength

RESPONSE_EC_VIEWS = (
    "dnll_noctx", "dent_noctx", "djs_noctx", "dlogsumexp_noctx",
    "dnll_loo_max", "dnll_loo_mean", "dent_loo_max", "dent_loo_mean",
)
TOKEN_EC_VIEWS = (
    "dnll_noctx", "dent_noctx", "djs_noctx",
    "dnll_loo_max", "dnll_loo_mean", "dent_loo_max", "dent_loo_mean",
)


# ── data loading ─────────────────────────────────────────────────────────────────────

def load_cell(path) -> dict:
    """`{response_id: {condition: cache_entry}}`."""
    with open(path, "rb") as handle:
        cache = pickle.load(handle)
    by_response = {}
    for entry in cache.values():
        by_response.setdefault(entry["response_id"], {})[entry["condition"]] = entry
    return by_response


def load_rows_by_id(split="test", wanted_ids=None) -> dict:
    """`{response_id (str): ragtruth_row}` with `_chunk_texts` populated (needed for arm
    5b's evidence graph). `wanted_ids`, if given, restricts the (possibly large) corpus
    read to just the responses actually present in a scored cell."""
    rows, _ = load_ragtruth(split=split)
    out = {}
    wanted = {str(x) for x in wanted_ids} if wanted_ids is not None else None
    for row in rows:
        rid = str(row["response_id"])
        if wanted is not None and rid not in wanted:
            continue
        try:
            spans = chunk_source(row)
        except ValueError:
            continue
        row["_chunk_spans"] = spans
        row["_chunk_texts"] = [row["context"][s:e] for s, e in spans]
        out[rid] = row
    return out


def _hash_array(values) -> str:
    arr = np.asarray(values, dtype="<f8")
    return hashlib.sha256(arr.tobytes()).hexdigest()


# ── evidence-graph smoothing (arm 5b) ───────────────────────────────────────────────

def _smooth_loo_values(raw_per_chunk: dict, graph: np.ndarray, node_names: list,
                        lam=GRAPH_SMOOTH_LAMBDA) -> dict:
    """`raw_per_chunk`: `{loo_j: scalar}`. Smooths via `s* = (I + lam*L)^{-1} v` over the
    FULL node set (full/noctx included, pinned to 0 — they are not "removed chunk"
    quantities), then returns only the `loo_j` entries."""
    n = len(node_names)
    v = np.zeros(n, dtype=float)
    mask = np.zeros(n, dtype=bool)
    for j, name in enumerate(node_names):
        if name in raw_per_chunk:
            v[j] = raw_per_chunk[name]
            mask[j] = True
    if mask.sum() < 2:
        return dict(raw_per_chunk)
    degree = np.diag(graph.sum(axis=1))
    L = degree - graph
    smoothed = np.linalg.solve(np.eye(n) + lam * L, v)
    return {name: float(smoothed[j]) for j, name in enumerate(node_names) if mask[j]}


def _response_evidence_graph_deltas(row, records_by_condition: dict) -> dict | None:
    """Rebuilds `dnll_loo_max/mean`, `dent_loo_max/mean` using evidence-graph-smoothed
    per-chunk values instead of the raw ones `response_delta_views` uses. Needs the
    original RAGTruth `row` (for chunk texts) — callers without it should skip arm 5b."""
    loo = sorted((c for c in records_by_condition if c.startswith("loo_")),
                 key=lambda c: int(c[len("loo_"):]))
    if not loo or row is None:
        return None
    full = records_by_condition["full"]
    T = len(full["token_entropies"])
    spill_full = np.asarray(full["token_spilled_energies"], dtype=float)[:T]
    ent_full = np.asarray(full["token_entropies"], dtype=float)[:T]

    chunk_texts = row.get("_chunk_texts")
    if chunk_texts is None:
        return None
    node_names = ["full", "noctx"] + [f"loo_{j}" for j in range(len(chunk_texts))]
    graph = build_evidence_graph(chunk_texts, node_names)

    raw_dnll = {c: float(np.mean(np.asarray(records_by_condition[c]["token_spilled_energies"],
                                             dtype=float)[:T] - spill_full)) for c in loo}
    raw_dent = {c: float(np.mean(np.asarray(records_by_condition[c]["token_entropies"],
                                             dtype=float)[:T] - ent_full)) for c in loo}
    sm_dnll = _smooth_loo_values(raw_dnll, graph, node_names)
    sm_dent = _smooth_loo_values(raw_dent, graph, node_names)
    if not sm_dnll:
        return None
    return {
        "dnll_loo_max": max(sm_dnll.values()), "dnll_loo_mean": float(np.mean(list(sm_dnll.values()))),
        "dent_loo_max": max(sm_dent.values()), "dent_loo_mean": float(np.mean(list(sm_dent.values()))),
    }


# ── response-level feature matrices ─────────────────────────────────────────────────

def _response_feature_matrix(rows_ec: list, names) -> tuple:
    """`rows_ec`: list of `{name: value_or_None}`. Z-scores each column (median-impute
    missing), drops zero-variance/all-missing columns. Returns `(F [features, samples],
    kept_names, mu, sd)`."""
    kept, mu, sd, columns = [], [], [], []
    for name in names:
        raw = np.asarray([row.get(name) for row in rows_ec], dtype=float)
        finite = np.isfinite(raw)
        if finite.sum() < max(3, 0.3 * len(raw)):
            continue
        median = float(np.median(raw[finite]))
        raw = np.where(finite, raw, median)
        scale = float(raw.std())
        if scale < 1e-10:
            continue
        kept.append(name); mu.append(float(raw.mean())); sd.append(scale)
        columns.append((raw - raw.mean()) / scale)
    if not columns:
        return None, [], np.array([]), np.array([])
    return np.column_stack(columns).T, kept, np.asarray(mu), np.asarray(sd)


# ── token-level channels + temporal-chain DUFS-LIU (shared by arms 1, 5, 5b) ──────

def _prepare_token_matrix(channel_lists: dict, names, max_tokens=200_000):
    """`channel_lists`: `{name: [array_per_response, ...]}`, one entry per response in a
    FIXED order shared across all names. Flattens into one `[features, tokens]` matrix
    plus the per-response `lengths` needed for `_temporal_graph`."""
    n_responses = len(next(iter(channel_lists.values())))
    lengths, total = [], 0
    for i in range(n_responses):
        length = len(channel_lists[names[0]][i])
        take = min(length, max_tokens - total) if max_tokens - total > 0 else 0
        lengths.append(take)
        total += take
    kept, mu, sd, columns = [], [], [], []
    for name in names:
        values = np.concatenate([
            np.asarray(channel_lists[name][i], dtype=float)[:lengths[i]]
            for i in range(n_responses) if lengths[i] > 0
        ]) if total > 0 else np.array([])
        if values.size < 10:
            continue
        finite = np.isfinite(values)
        if not finite.any():
            continue
        median = float(np.median(values[finite]))
        values = np.where(finite, values, median)
        scale = float(values.std())
        if scale < 1e-10:
            continue
        kept.append(name); mu.append(float(values.mean())); sd.append(scale)
        columns.append((values - values.mean()) / scale)
    if not columns:
        return None, [], np.array([]), np.array([]), lengths
    return np.column_stack(columns).T, kept, np.asarray(mu), np.asarray(sd), lengths


def _temporal_graph(lengths):
    rows, cols, vals, start = [], [], [], 0
    for length in lengths:
        if length > 1:
            left = np.arange(start, start + length - 1)
            right = left + 1
            rows.extend(left.tolist()); cols.extend(right.tolist()); vals.extend([1.0] * len(left))
            rows.extend(right.tolist()); cols.extend(left.tolist()); vals.extend([1.0] * len(left))
        start += length
    return coo_matrix((vals, (rows, cols)), shape=(start, start)).tocsr()


def _fit_temporal_dufs_liu(channel_lists: dict, names, anchor_name=None):
    """One reusable fit: builds the token matrix, a temporal-chain graph, DUFS soft gates,
    and the Laplacian-IU-PCR path at the carried lambdas. Returns the arm dict needed by
    `_apply_token_arm` plus diagnostics. `anchor_name` orients the global sign (falls back
    to the first kept channel if the requested anchor isn't available)."""
    F, names_kept, mu, sd, lengths = _prepare_token_matrix(channel_lists, names)
    if F is None or F.shape[0] < 3 or F.shape[1] < 3:
        return None, {"skipped": "too few features/tokens"}
    graphs = {"temporal": _temporal_graph(lengths)}
    gates, gate_diag = adapted_dufs_soft_gates(F, seeds=DUFS_SEEDS, epochs=DUFS_EPOCHS)
    graphs["dufs_temporal_gated"] = build_graph_from_features(F, gates=gates, k=K) \
        if F.shape[1] >= K + 1 else graphs["temporal"]

    anchor_idx = names_kept.index(anchor_name) if anchor_name in names_kept else 0
    anchor = F[anchor_idx]

    best_key, best_path = "temporal", laplacian_iu_path(F, LAMBDAS, graph=graphs["temporal"])
    weights = best_path[LAMBDAS[len(LAMBDAS) // 2]].w
    _, flipped = anchor_orient(weights @ F, anchor)
    return ({"names": names_kept, "mu": mu, "sd": sd, "flipped": bool(flipped),
             "weights": np.asarray(weights)},
            {"n_tokens": F.shape[1], "n_features": F.shape[0], "lengths": lengths,
             "gate_effective_count": gate_diag.get("effective_feature_count")})


def _apply_token_arm(arm, channel_lists, response_index):
    columns = []
    for j, name in enumerate(arm["names"]):
        values = np.asarray(channel_lists[name][response_index], dtype=float)
        values = np.where(np.isfinite(values), values, arm["mu"][j])
        columns.append((values - arm["mu"][j]) / arm["sd"][j])
    risk = arm["weights"] @ np.vstack(columns)
    return -risk if arm["flipped"] else risk


# ── arm assembly ─────────────────────────────────────────────────────────────────────

def build_arms(by_response: dict, rows_by_id=None):
    """`rows_by_id`: optional `{response_id: ragtruth_row}` with `_chunk_texts`/
    `_chunk_spans` populated — needed only for arm 5b. Returns
    `(response_ids, response_scores: {arm_name: array}, token_curves: {arm_name: [array]},
    diagnostics)`. Responses without a `full` pass are dropped."""
    response_ids = [rid for rid, conds in by_response.items() if "full" in conds]
    diagnostics = {"n_responses": len(response_ids)}

    delta_views = {}
    gasp_feats, gasp_curves = [], []
    full_channels = {name: [] for name in CORE_TOKEN_VIEWS}
    for rid in response_ids:
        conds = by_response[rid]
        delta_views[rid] = response_delta_views(conds)
        gasp_feats.append(gasp_mod.response_gasp_features(conds))
        gasp_curves.append(gasp_mod.token_gasp_curve(conds))
        views = token_feature_views(conds["full"])
        for name in CORE_TOKEN_VIEWS:
            full_channels[name].append(views.get(name, np.zeros(len(conds["full"]["token_entropies"]))))

    response_scores, token_curves = {}, {}

    # Arm 2 — likelihood-drop baseline: mean(NLL_noctx - NLL_full), no fusion.
    response_scores["likelihood_drop"] = np.array(
        [delta_views[rid]["response"].get("dnll_noctx", np.nan) for rid in response_ids])

    # Arm 3 — GASP-threshold reproduction.
    response_scores["gasp_reproduction"] = gasp_mod.gasp_threshold_scores(gasp_feats)
    curves, pop_stats = gasp_mod.gasp_threshold_curve(gasp_curves)
    token_curves["gasp_reproduction"] = curves
    diagnostics["gasp_population_stats"] = pop_stats

    # Arm 6 — fusion-isolation ablation: EC views, z-scored, naively averaged (no U-PCR).
    ec_rows = [delta_views[rid]["response"] for rid in response_ids]
    Fec, kept_ec, _, _ = _response_feature_matrix(ec_rows, RESPONSE_EC_VIEWS)
    if Fec is not None:
        response_scores["fusion_isolation_naive_avg"] = -Fec.mean(axis=0)
        diagnostics["fusion_isolation_features"] = kept_ec
    else:
        response_scores["fusion_isolation_naive_avg"] = np.full(len(response_ids), np.nan)

    # Arm 4 — EC-U-PCR: same feature matrix, deployed U-PCR solver.
    if Fec is not None and Fec.shape[0] >= 2 and Fec.shape[1] >= 5:
        result = upcr_fit(Fec, loss="l2", scale_ratio=0.25)
        derived = np.sign(result.rho_hat_full)
        derived[derived == 0] = 1.0
        response_scores["ec_upcr"] = -(result.w @ (Fec * derived[:, None]))
        diagnostics["ec_upcr_features"] = kept_ec
    else:
        response_scores["ec_upcr"] = np.full(len(response_ids), np.nan)

    # Arm 1 — full-context-only DUFS-LIU: CORE_TOKEN_VIEWS on the `full` pass alone.
    arm1, diag1 = _fit_temporal_dufs_liu(full_channels, list(CORE_TOKEN_VIEWS),
                                          anchor_name="entropy_series")
    diagnostics["arm1"] = diag1
    if arm1 is not None:
        curves1 = [_apply_token_arm(arm1, full_channels, i) for i in range(len(response_ids))]
        token_curves["full_context_only_dufs_liu"] = curves1
        response_scores["full_context_only_dufs_liu"] = np.array(
            [float(np.mean(c)) for c in curves1])
    else:
        response_scores["full_context_only_dufs_liu"] = np.full(len(response_ids), np.nan)

    # Arm 5 — EC-DUFS-LIU, temporal-chain graph, on the token-level Δ-views.
    token_channels = {name: [] for name in TOKEN_EC_VIEWS}
    for rid in response_ids:
        tv = delta_views[rid]["token"]
        T = len(by_response[rid]["full"]["token_entropies"])
        for name in TOKEN_EC_VIEWS:
            token_channels[name].append(tv.get(name, np.full(T, np.nan)))
    arm5, diag5 = _fit_temporal_dufs_liu(token_channels, list(TOKEN_EC_VIEWS),
                                          anchor_name="dnll_noctx")
    diagnostics["arm5"] = diag5
    if arm5 is not None:
        curves5 = [_apply_token_arm(arm5, token_channels, i) for i in range(len(response_ids))]
        token_curves["ec_dufs_liu_temporal"] = curves5
        response_scores["ec_dufs_liu_temporal"] = np.array([float(np.mean(c)) for c in curves5])
    else:
        response_scores["ec_dufs_liu_temporal"] = np.full(len(response_ids), np.nan)

    # Arm 5b — same as arm 5, but LOO Δ-views are evidence-graph-smoothed first.
    if rows_by_id is not None:
        token_channels_5b = {name: list(vals) for name, vals in token_channels.items()}
        any_smoothed = False
        for i, rid in enumerate(response_ids):
            smoothed = _response_evidence_graph_deltas(rows_by_id.get(rid), by_response[rid])
            if smoothed is None:
                continue
            any_smoothed = True
            T = len(token_channels_5b["dnll_noctx"][i])
            for name, scalar in smoothed.items():
                if name in token_channels_5b:
                    token_channels_5b[name][i] = np.full(T, scalar)
        if any_smoothed:
            arm5b, diag5b = _fit_temporal_dufs_liu(token_channels_5b, list(TOKEN_EC_VIEWS),
                                                     anchor_name="dnll_noctx")
            diagnostics["arm5b"] = diag5b
            if arm5b is not None:
                curves5b = [_apply_token_arm(arm5b, token_channels_5b, i)
                           for i in range(len(response_ids))]
                token_curves["ec_dufs_liu_evidence_graph"] = curves5b
                response_scores["ec_dufs_liu_evidence_graph"] = np.array(
                    [float(np.mean(c)) for c in curves5b])
        else:
            diagnostics["arm5b"] = {"skipped": "no rows_by_id chunk text available"}

    return response_ids, response_scores, token_curves, diagnostics


# ── metrics (labels read here only) ─────────────────────────────────────────────────

def evaluate(by_response, response_ids, response_scores, token_curves):
    labels = np.array([bool(by_response[rid]["full"]["response_label"]) for rid in response_ids])
    detector_rows = []
    for name, scores in response_scores.items():
        finite = np.isfinite(scores)
        if finite.sum() < len(scores) or len(np.unique(labels[finite])) < 2:
            detector_rows.append({"arm": name, "auroc": None, "auprc": None,
                                  "n_valid": int(finite.sum())})
            continue
        detector_rows.append({
            "arm": name,
            "auroc": float(roc_auc_score(labels[finite], scores[finite])),
            "auprc": float(average_precision_score(labels[finite], scores[finite])),
            "n_valid": int(finite.sum()),
        })

    locator_rows = []
    for name, curves in token_curves.items():
        hits, total = 0, 0
        for rid, curve in zip(response_ids, curves):
            spans = by_response[rid]["full"]["span_token_spans"]
            if not spans or curve is None or not np.isfinite(curve).any():
                continue
            total += 1
            idx = int(np.nanargmax(curve))
            if any(start <= idx < end for start, end in spans):
                hits += 1
        locator_rows.append({"arm": name, "argmax_hit_rate": (hits / total) if total else None,
                             "n_with_spans": total})

    return {"response_ids": response_ids, "labels": labels.tolist(),
            "detector_rows": detector_rows, "locator_rows": locator_rows}


def freeze_hashes(response_ids, response_scores, token_curves) -> dict:
    hashes = {"response": {}, "token": {}}
    for name, scores in response_scores.items():
        hashes["response"][name] = _hash_array(np.nan_to_num(scores, nan=0.0))
    for name, curves in token_curves.items():
        flat = np.concatenate([np.nan_to_num(np.asarray(c, dtype=float), nan=0.0)
                               for c in curves]) if curves else np.array([])
        hashes["token"][name] = _hash_array(flat)
    return hashes


# ── CLI ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--smoke", action="store_true", help="run the synthetic-data smoke test and exit")
    ap.add_argument("pkl_path", nargs="?")
    ap.add_argument("--open-labels", action="store_true",
                    help="compute AUROC/AUPRC/locator metrics (otherwise scores+hashes only)")
    ap.add_argument("--split", default="test", help="RAGTruth split to join for arm 5b's chunk texts")
    ap.add_argument("--skip-evidence-graph", action="store_true",
                    help="skip arm 5b (no RAGTruth row join, e.g. offline smoke use)")
    args = ap.parse_args()

    if args.smoke:
        smoke()
        return
    if not args.pkl_path:
        ap.error("pkl_path is required unless --smoke is given")

    by_response = load_cell(args.pkl_path)
    rows_by_id = None
    if not args.skip_evidence_graph:
        rows_by_id = load_rows_by_id(split=args.split, wanted_ids=by_response.keys())
        print(f"joined {len(rows_by_id)}/{len(by_response)} responses to RAGTruth rows "
              f"(split={args.split}) for arm 5b")
    response_ids, response_scores, token_curves, diagnostics = build_arms(by_response, rows_by_id)
    hashes = freeze_hashes(response_ids, response_scores, token_curves)
    print(f"n_responses={len(response_ids)} arms={sorted(response_scores)}")
    print(f"diagnostics: {diagnostics}")
    print(f"score hashes (frozen BEFORE labels): {hashes['response']}")

    if args.open_labels:
        result = evaluate(by_response, response_ids, response_scores, token_curves)
        for row in result["detector_rows"]:
            print(row)
        for row in result["locator_rows"]:
            print(row)


# ── known-answer test ────────────────────────────────────────────────────────────────

def _synthetic_response(rng, T, n_loo, grounded, hallucinated_span=None):
    """A synthetic response's `{condition: cache_entry}`, `grounded` controlling whether
    noctx/loo passes hurt (grounded) or barely move (hallucinated) the telemetry."""
    def _lp(shift):
        ids = np.tile(np.arange(6), (T, 1))
        logits = -np.abs(rng.normal(0, 1, size=(T, 6))) - shift
        return {"ids": ids, "logprobs": np.sort(logits, axis=1)[:, ::-1]}

    base_ent = rng.uniform(0.3, 1.2, T)
    base_spill = rng.uniform(0.5, 1.5, T)
    base_z = rng.uniform(4.0, 6.0, T)
    full = {"token_entropies": base_ent.tolist(), "token_spilled_energies": base_spill.tolist(),
            "token_logsumexp": base_z.tolist(), "top_k_logprobs": _lp(0.0)}
    bump = 0.6 if grounded else 0.02
    conds = {"full": full, "noctx": {
        "token_entropies": (base_ent + bump).tolist(),
        "token_spilled_energies": (base_spill + bump).tolist(),
        "token_logsumexp": (base_z - bump).tolist(), "top_k_logprobs": _lp(bump),
    }}
    for j in range(n_loo):
        k_bump = bump * rng.uniform(0.5, 1.5)
        conds[f"loo_{j}"] = {
            "token_entropies": (base_ent + k_bump).tolist(),
            "token_spilled_energies": (base_spill + k_bump).tolist(),
            "top_k_logprobs": _lp(k_bump),
        }
    spans = [hallucinated_span] if hallucinated_span else []
    for entry in conds.values():
        entry["response_label"] = bool(hallucinated_span)
        entry["span_token_spans"] = spans
    return conds


def smoke() -> None:
    rng = np.random.default_rng(0)
    by_response, rows_by_id = {}, {}
    for i in range(12):
        rid = f"r{i}"
        grounded = i % 2 == 0
        span = (5, 8) if not grounded else None
        by_response[rid] = _synthetic_response(rng, T=40, n_loo=3, grounded=grounded,
                                                hallucinated_span=span)
        rows_by_id[rid] = {"_chunk_texts": [
            "alpha beta gamma delta", "epsilon zeta eta theta", "iota kappa lambda mu",
        ]}

    # 1. Runs end-to-end with all 7 arms present, no crash, no NaN-only scores.
    response_ids, response_scores, token_curves, diagnostics = build_arms(by_response, rows_by_id)
    expected_arms = {"likelihood_drop", "gasp_reproduction", "fusion_isolation_naive_avg",
                     "ec_upcr", "full_context_only_dufs_liu", "ec_dufs_liu_temporal",
                     "ec_dufs_liu_evidence_graph"}
    assert expected_arms.issubset(response_scores.keys()), response_scores.keys()
    for name, scores in response_scores.items():
        assert np.isfinite(scores).any(), f"{name}: all-NaN scores"

    # 2. Score hashing is deterministic and label-free (same input -> same hash twice).
    h1 = freeze_hashes(response_ids, response_scores, token_curves)
    h2 = freeze_hashes(response_ids, response_scores, token_curves)
    assert h1 == h2

    # 3. evaluate() runs and produces a finite AUROC for at least one arm (labels opened
    #    only here, exactly mirroring the real CLI's --open-labels boundary).
    result = evaluate(by_response, response_ids, response_scores, token_curves)
    assert any(row["auroc"] is not None for row in result["detector_rows"])
    assert any(row["argmax_hit_rate"] is not None for row in result["locator_rows"])

    # 4. A cell with zero loo/evidence-graph coverage (Summary-only) degrades gracefully:
    #    arm 5b is simply absent, not a crash.
    solo_by_response = {rid: {"full": conds["full"], "noctx": conds["noctx"]}
                        for rid, conds in list(by_response.items())[:6]}
    rid2, rs2, tc2, diag2 = build_arms(solo_by_response, rows_by_id=None)
    assert "ec_dufs_liu_evidence_graph" not in rs2
    assert "likelihood_drop" in rs2

    print("run.smoke: PASS (4 checks)")


if __name__ == "__main__":
    main()
