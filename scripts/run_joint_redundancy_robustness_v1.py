#!/usr/bin/env python3
"""Joint L-SML redundancy-robustness test on the frozen 13,769-answer population.

One question: when redundant streams are added to the eight-stream L08 roster,
does the Joint global+local factor model lose less localization quality than
Continuous L-SML, IU-PCR and the equal reference?

Design (development only, no untouched confirmation):
  * rosters   L08 (Step-397 atlas diverse eight), L11 = L08 + one duplicate per
              family, L14x = L11 + three more duplicates, and Codex's exact L14
              as a replay anchor.
  * arms      continuous (auto groups), joint_global_v, joint_inverse,
              joint_hier (Joint's own LOAO consensus groups, minimum size two
              via the checked pair-product fit), joint_lsmlgroups_hier (same
              Joint fit on the groups Continuous L-SML discovered), iu (frozen
              IU_FIT_DEFAULTS), equal (labelled reference row, not a method).
  * harness   five outer source folds; every weight, group and Joint fit uses
              training answers only; the frozen incumbent gate scores all OOF
              locators; paired source-group bootstrap for the declared contrasts.
No label enters any fit. Groups are never supplied by hand.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_lsml_gate_locator_research_v1 import load_inputs, merge_step_bank, score_locator  # noqa: E402
from spectral_utils.historical_fusion_evaluation import pb_metrics  # noqa: E402
from spectral_utils.joint_lsml import (  # noqa: E402
    covariance_matrix, discover_loao_consensus_groups, hierarchical_joint_weights,
    regularized_joint_map_weights,
)
from spectral_utils.joint_group_readouts import hierarchical_group_readout  # noqa: E402
from spectral_utils.joint_pair_jacobian import fit_joint_pairs_checked  # noqa: E402
from spectral_utils.laplacian_upcr import IU_FIT_DEFAULTS  # noqa: E402
from spectral_utils.lsml_gate_locator_research import (  # noqa: E402
    FusionRecipe, _orient, answer_standardize, fit_fusion_weights,
)
from spectral_utils.upcr import upcr_fit  # noqa: E402

SCHEMA = "joint-redundancy-robustness-v1"
OUTPUT = ROOT / "results/joint_redundancy_robustness_v1"
SEED = 398000

L08 = (
    "step::digit.disagreement::top2", "step::digit.token_clock_innovation::top1",
    "step::q15.VE0.75.prefix_mean_innovation::top10", "step::renyi_escort.a0.25::top10",
    "step::direct_probability.rank_1_risk::top10", "step::direct_probability.rank_9_risk::top8",
    "step::step395.logtail15::top10", "step::step395.mass_above::top10",
)
L11 = L08 + (
    "step::digit.disagreement::top1", "step::renyi_escort.a8::top10", "step::step395.logtail50::top10",
)
L14X = L11 + (
    "step::q15.H0lim.prefix_mean_innovation::top10", "step::direct_probability.rank_3_risk::top10",
    "step::direct_probability.rank_10_risk::top8",
)
L14_CODEX = (
    "step::digit.disagreement::top1", "step::digit.disagreement::top2",
    "step::digit.token_clock_innovation::top1",
    "step::q15.VE0.75.prefix_mean_innovation::top10", "step::q15.H0lim.prefix_mean_innovation::top10",
    "step::renyi_escort.a0.25::top10", "step::renyi_escort.a8::top10",
    "step::direct_probability.rank_1_risk::top10", "step::direct_probability.rank_3_risk::top10",
    "step::direct_probability.rank_9_risk::top10", "step::direct_probability.rank_10_risk::top8",
    "step::step395.logtail15::top10", "step::step395.logtail50::top10", "step::step395.mass_above::top10",
)
L24 = (  # every eligible step_post_readout stream of the atlas (LALL24 in Step 397)
    "step::digit.disagreement::top1", "step::digit.disagreement::top2",
    "step::digit.token_clock_innovation::top1",
    "step::direct_probability.rank_10_risk::top8", "step::direct_probability.rank_1_risk::top10",
    "step::direct_probability.rank_1_risk::top8", "step::direct_probability.rank_3_risk::top10",
    "step::direct_probability.rank_9_risk::top10", "step::direct_probability.rank_9_risk::top8",
    "step::q15.H0lim.prefix_mean_innovation::top10", "step::q15.VE0.75.prefix_mean_innovation::top10",
    "step::q15.VE0.75.prefix_mean_innovation::top8",
    "step::renyi_escort.a0.05::top10", "step::renyi_escort.a0.05::top8", "step::renyi_escort.a0.15::top8",
    "step::renyi_escort.a0.25::top10", "step::renyi_escort.a0.3::top10", "step::renyi_escort.a4::top8",
    "step::renyi_escort.a8::top10",
    "step::step395.logtail15::top10", "step::step395.logtail50::top10", "step::step395.logtail50::top8",
    "step::step395.mass_above::top10", "step::step395.mass_above::top8",
)
ROSTERS = {"L08": L08, "L11": L11, "L14x": L14X, "L14_codex": L14_CODEX, "L24": L24}
EXPANDED = ("L11", "L14x", "L24")  # rosters contrasted against L08 in the bootstrap
ARMS = ("continuous", "joint_global_v", "joint_inverse", "joint_hier", "joint_hier_u", "joint_hier_sml",
        "joint_lsmlgroups_hier", "iu", "equal")
REPLAY_ANCHORS = {  # Step 397 / soft_joint_auto_v1 point estimates that must replay
    ("L08", "continuous"): (0.437402, 0.778143),
    ("L14_codex", "continuous"): (0.393519, 0.755577),
    ("L24", "continuous"): (0.389074, 0.750769),
}


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def rows_for(offsets, answers):
    return np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in answers])


def digit_share(weight, names):
    w = np.abs(np.asarray(weight, float))
    total = float(w.sum())
    if total <= 0:
        return float("nan")
    return float(sum(w[i] for i, n in enumerate(names) if n.startswith("step::digit")) / total)


def jacobian_summary(audit):
    return {
        "full_global_rank": bool(audit.get("full_global_rank")),
        "condition_number": float(audit.get("condition_number", float("nan"))),
        "rank": int(audit.get("rank", -1)),
    }


def joint_readouts(train, labels, seed, names, anchor=0):
    """Fit Joint (pair-checked) on one partition; return the three readouts and audit."""
    cov = covariance_matrix(train)
    fit = fit_joint_pairs_checked(cov, labels, anchor_index=anchor, seed=seed, starts=5)
    joint = fit.joint
    out = {}
    w_v, o_v = _orient(train, joint.global_loading, anchor)
    out["joint_global_v"] = (w_v, {"anchor": o_v})
    w_inv, inv_detail = regularized_joint_map_weights(
        None, joint.model_covariance, joint.global_loading, mode="liu", lam=0.0, target_condition=1000.0)
    w_inv, o_inv = _orient(train, w_inv, anchor)
    out["joint_inverse"] = (w_inv, {"anchor": o_inv, "inverse": {k: v for k, v in inv_detail.items()
                                                                if isinstance(v, (int, float, str, bool))}})
    _, w_h, h_meta = hierarchical_joint_weights(train, labels, joint.global_loading, anchor_index=anchor,
                                                small_m_guard=True)
    w_h, o_h = _orient(train, w_h, anchor)
    out["joint_hier"] = (w_h, {"anchor": o_h, "cross_group_weights": h_meta["cross_group_weights"],
                               "cross_small_m_guarded": h_meta["cross_small_m_guarded"]})
    # Within-group readouts independent of v (Step 399 follow-up): group factor u_g, within-group SML.
    for readout, arm in (("group_factor", "joint_hier_u"), ("within_sml", "joint_hier_sml")):
        w_r, r_meta = hierarchical_group_readout(train, labels, joint.global_loading, joint.group_loading,
                                                 readout=readout, small_m_guard=True)
        w_r, o_r = _orient(train, w_r, anchor)
        out[arm] = (w_r, {"anchor": o_r, "cross_group_weights": r_meta["cross_group_weights"],
                          "group_notes": r_meta["group_notes"]})
    audit = {
        "converged": bool(joint.converged), "converged_starts": int(joint.converged_starts),
        "multistart": joint.multistart_audit["status"], "native_map": fit.native_map_audit["status"],
        "pair_status": fit.pair_audit.get("status"), "pair_count": int(fit.pair_audit.get("pair_count", 0)),
        "relative_offdiag_misfit": float(joint.relative_offdiag_misfit),
        "jacobian": jacobian_summary(joint.jacobian_audit),
        "group_sizes": [int(np.sum(np.asarray(labels) == g)) for g in np.unique(labels)],
        "labels": np.asarray(labels, int).tolist(),
        "valid": bool(joint.converged and joint.multistart_audit["status"] == "PASS"
                      and joint.jacobian_audit.get("full_global_rank", False)
                      and np.isfinite(joint.jacobian_audit.get("condition_number", np.inf))
                      and joint.jacobian_audit.get("condition_number", np.inf) <= 1e8),
    }
    return out, audit


def run_fold(x, names, data, step_owner, outer, seed):
    offsets = np.asarray(data["offsets"]); folds = np.asarray(data["folds"])
    train_answers = np.flatnonzero(folds != outer); test_answers = np.flatnonzero(folds == outer)
    tr = rows_for(offsets, train_answers); te = rows_for(offsets, test_answers)
    train = x[tr]; weights = {}; meta = {}
    t0 = time.perf_counter()

    # 1. Continuous L-SML with its own group discovery (replay anchor path).
    cw, cmeta = fit_fusion_weights(train, FusionRecipe("continuous", tuple(names), "continuous", anchor=0),
                                   seed=seed)
    weights["continuous"] = cw; meta["continuous"] = cmeta

    # 2. Joint's own label-free grouping, minimum group size two (pairs allowed).
    discovery = discover_loao_consensus_groups(
        train, step_owner[tr], k_range=(3, 4), seed=seed + 100, minimum_group_size=2,
        minimum_held_admissible_fraction=0.95, pairwise_diagnostic_cap=4096, use_minimum_ari_tiebreak=True)
    meta["joint_discovery"] = {k: discovery.get(k) for k in ("status", "K", "group_sizes", "median_ari", "minimum_ari")}
    if discovery["status"] == "SELECTED":
        readouts, audit = joint_readouts(train, np.asarray(discovery["labels"], int), seed + 200, names)
        for arm, (w, m) in readouts.items():
            weights[arm] = w; meta[arm] = {**m, "audit": audit}
    else:
        meta["joint_auto"] = {"status": discovery["status"]}

    # 3. Same Joint fit on the partition Continuous L-SML discovered, if admissible.
    lsml_groups = np.asarray(cmeta["groups"], int)
    sizes = [int(np.sum(lsml_groups == g)) for g in np.unique(lsml_groups)]
    if len(sizes) >= 3 and min(sizes) >= 2:
        try:
            readouts, audit = joint_readouts(train, lsml_groups, seed + 300, names)
            w, m = readouts["joint_hier"]
            weights["joint_lsmlgroups_hier"] = w; meta["joint_lsmlgroups_hier"] = {**m, "audit": audit}
        except Exception as exc:  # explicit failure, never a silent fallback
            meta["joint_lsmlgroups_hier"] = {"status": "FAILED", "reason": str(exc), "group_sizes": sizes}
    else:
        meta["joint_lsmlgroups_hier"] = {"status": "INADMISSIBLE_PARTITION", "group_sizes": sizes}

    # 4. IU-PCR control with the frozen localization defaults.
    try:
        iu = upcr_fit(train.T, **dict(IU_FIT_DEFAULTS))
        w, o = _orient(train, iu.w, 0)
        weights["iu"] = w; meta["iu"] = {"anchor": o, "abstained": bool(iu.abstained), "g2_hat": float(iu.g2_hat),
                                         "n_components_used": int(iu.n_components_used)}
    except Exception as exc:
        meta["iu"] = {"status": "FAILED", "reason": str(exc)}

    # 5. Equal reference row.
    weights["equal"] = np.ones(len(names)) / len(names); meta["equal"] = {"mode": "equal"}

    scores = {arm: x[te] @ w for arm, w in weights.items()}
    for arm, w in weights.items():
        meta[arm]["weights"] = np.asarray(w, float).tolist()
        meta[arm]["digit_share"] = digit_share(w, names)
    meta["seconds"] = time.perf_counter() - t0
    return te, scores, meta


def paired_bootstrap(data, oof, draws, seed):
    """Paired source-group bootstrap of PB macro-F1 and PRMB within-AUC for every arm."""
    gate = np.asarray(data["current_gate"], bool); target = np.asarray(data["target"])
    cells = data["cells"].astype(str); groups = data["groups"].astype(str)
    unique, group_id = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    preds, withins, keys = [], [], []
    for roster, arms in oof.items():
        for arm, score in arms.items():
            res = score_locator(score, gate, data)
            preds.append(res["prediction"].astype(np.int32)); withins.append(res["within_values"]); keys.append((roster, arm))
    valid = np.ones(len(target), bool)
    pb = np.empty((draws, len(keys))); within = np.empty((draws, len(keys)))
    for d in range(draws):
        count = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
        wts = count[group_id].astype(float)
        for k in range(len(keys)):
            pb[d, k] = pb_metrics(target, preds[k], valid, cells, weights=wts)["macros"]["all"]
            m = np.isfinite(withins[k]) & (wts > 0)
            within[d, k] = np.average(withins[k][m], weights=wts[m])
        if (d + 1) % 500 == 0:
            print(f"bootstrap {d + 1}/{draws}", flush=True)
    return keys, pb, within


def interval(draws, point):
    return {"point": float(point), "low": float(np.quantile(draws, 0.025)), "high": float(np.quantile(draws, 0.975)),
            "probability_positive": float(np.mean(draws > 0))}


def build_contrasts(metrics, oof, data, draws, seed):
    """Declared contrasts with paired source-group bootstrap intervals."""
    keys, pb, within = paired_bootstrap(data, oof, draws, seed)
    col = {k: i for i, k in enumerate(keys)}
    pt = lambda r, a, f: metrics[r][a][f]  # noqa: E731
    contrasts = {}
    def add(name, dp, dw, pp, pw):
        contrasts[name] = {"pb": interval(dp, pp), "within": interval(dw, pw)}
    for big in EXPANDED:
        for arm in ARMS:
            if ("L08", arm) in col and (big, arm) in col:
                add(f"{arm}: {big} - L08", pb[:, col[(big, arm)]] - pb[:, col[("L08", arm)]],
                    within[:, col[(big, arm)]] - within[:, col[("L08", arm)]],
                    pt(big, arm, "pb") - pt("L08", arm, "pb"), pt(big, arm, "within") - pt("L08", arm, "within"))
    for roster in ("L08",) + EXPANDED:
        for arm in ARMS:
            if arm != "continuous" and (roster, arm) in col and (roster, "continuous") in col:
                add(f"{arm} - continuous on {roster}", pb[:, col[(roster, arm)]] - pb[:, col[(roster, "continuous")]],
                    within[:, col[(roster, arm)]] - within[:, col[(roster, "continuous")]],
                    pt(roster, arm, "pb") - pt(roster, "continuous", "pb"),
                    pt(roster, arm, "within") - pt(roster, "continuous", "within"))
    # Difference in differences: does the arm lose less than Continuous when redundancy is added?
    for big in EXPANDED:
        for arm in ("joint_hier", "joint_inverse", "joint_global_v", "joint_lsmlgroups_hier", "iu", "equal"):
            need = (("L08", arm), (big, arm), ("L08", "continuous"), (big, "continuous"))
            if all(k in col for k in need):
                dp = (pb[:, col[(big, arm)]] - pb[:, col[("L08", arm)]]) - (
                    pb[:, col[(big, "continuous")]] - pb[:, col[("L08", "continuous")]])
                dw = (within[:, col[(big, arm)]] - within[:, col[("L08", arm)]]) - (
                    within[:, col[(big, "continuous")]] - within[:, col[("L08", "continuous")]])
                pp = (pt(big, arm, "pb") - pt("L08", arm, "pb")) - (pt(big, "continuous", "pb") - pt("L08", "continuous", "pb"))
                pw = (pt(big, arm, "within") - pt("L08", arm, "within")) - (
                    pt(big, "continuous", "within") - pt("L08", "continuous", "within"))
                add(f"DiD {arm} vs continuous ({big}-L08)", dp, dw, pp, pw)
    return contrasts


def bootstrap_only(args, data):
    """Combine saved OOF arrays of every available roster into one contrast table."""
    out = Path(args.out); gate = np.asarray(data["current_gate"], bool)
    oof, metrics = {}, {}
    for roster in ROSTERS:
        path = out / f"OOF_{roster}.npz"
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as z:
            oof[roster] = {arm: z[arm].copy() for arm in z.files}
        metrics[roster] = {}
        for arm, v in oof[roster].items():
            res = score_locator(v, gate, data)
            metrics[roster][arm] = {"pb": res["pb"], "within": res["within"], "within_n": res["within_n"],
                                    "pb_cells": {c: r["f1"] for c, r in res["pb_cells"].items()}}
    contrasts = build_contrasts(metrics, oof, data, args.draws, SEED + 99)
    result = {"schema": SCHEMA + "/combined", "rosters": sorted(oof), "metrics": metrics, "contrasts": contrasts,
              "bootstrap": {"draws": args.draws, "seed": SEED + 99, "unit": "source group"}}
    (out / "RUN_COMBINED.json").write_text(json.dumps(clean(result), indent=1, sort_keys=True) + "\n")
    print(json.dumps(clean({"metrics": metrics, "contrasts": contrasts}), indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rosters", nargs="*", default=list(ROSTERS))
    ap.add_argument("--folds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--out", default=str(OUTPUT))
    ap.add_argument("--tag", default="RUN")
    ap.add_argument("--bootstrap-only", action="store_true")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    data = load_inputs(); raw, names = merge_step_bank(data); index = {n: i for i, n in enumerate(names)}
    offsets = np.asarray(data["offsets"]); folds = np.asarray(data["folds"])
    source = data["groups"].astype(str); _, source_id = np.unique(source, return_inverse=True)
    step_owner = np.repeat(source_id, np.diff(offsets))
    gate = np.asarray(data["current_gate"], bool)
    incumbent = score_locator(np.asarray(data["digit025_scores"], float), gate, data)
    print(f"incumbent replay: PB {incumbent['pb']:.6f} within {incumbent['within']:.6f}", flush=True)
    assert abs(incumbent["pb"] - 0.432546) < 5e-7 and abs(incumbent["within"] - 0.776036) < 5e-7, "incumbent replay failed"
    if args.bootstrap_only:
        return bootstrap_only(args, data)

    full = set(args.folds) == {0, 1, 2, 3, 4}
    oof, fold_meta, metrics = {}, {}, {}
    for r_i, roster in enumerate(args.rosters):
        members = ROSTERS[roster]
        missing = [n for n in members if n not in index]
        if missing:
            raise RuntimeError(f"{roster}: missing streams {missing}")
        x = answer_standardize(raw[:, [index[n] for n in members]], offsets)
        scores = {}; fold_meta[roster] = {}
        for outer in args.folds:
            te, s, m = run_fold(x, list(members), data, step_owner, outer, SEED + 1000 * r_i + 10 * outer)
            for arm, v in s.items():
                scores.setdefault(arm, np.full(len(x), np.nan))[te] = v
            fold_meta[roster][outer] = m
            print(f"{roster} fold {outer}: {m['seconds']:.1f}s discovery={m['joint_discovery']} "
                  f"arms={sorted(s)}", flush=True)
        kept_answers = np.flatnonzero(np.isin(folds, args.folds))
        kept_rows = rows_for(offsets, kept_answers)
        oof[roster] = {arm: v for arm, v in scores.items() if np.isfinite(v[kept_rows]).all()}
        metrics[roster] = {}
        for arm, v in oof[roster].items():
            res = score_locator(v, gate, data) if full else score_locator(v, gate, data, indexes=kept_answers)
            metrics[roster][arm] = {"pb": res["pb"], "within": res["within"], "within_n": res["within_n"],
                                    "pb_cells": {c: r["f1"] for c, r in res["pb_cells"].items()},
                                    "digit_share_mean": float(np.mean([fold_meta[roster][o][arm]["digit_share"]
                                                                       for o in args.folds if arm in fold_meta[roster][o]]))}
            print(f"  {roster:9s} {arm:22s} PB {100 * res['pb']:.4f}%  within {res['within']:.6f}", flush=True)
        np.savez_compressed(out / f"OOF_{roster}.npz", **oof[roster])

    replay = {}
    for (roster, arm), (pb_ref, within_ref) in REPLAY_ANCHORS.items():
        if roster in metrics and arm in metrics[roster] and full:
            got = metrics[roster][arm]
            replay[f"{roster}/{arm}"] = {"expected": [pb_ref, within_ref], "got": [got["pb"], got["within"]],
                                         "pass": bool(abs(got["pb"] - pb_ref) < 5e-7 and abs(got["within"] - within_ref) < 5e-7)}

    contrasts = build_contrasts(metrics, oof, data, args.draws, SEED + 99) if (full and args.draws > 0) else {}

    result = {
        "schema": SCHEMA, "development_only": True, "labels_used_for_fit": False, "groups_supplied_by_hand": False,
        "rosters": {k: list(v) for k, v in ROSTERS.items() if k in args.rosters}, "folds": args.folds,
        "incumbent": {"pb": incumbent["pb"], "within": incumbent["within"]},
        "metrics": metrics, "replay_anchors": replay, "contrasts": contrasts,
        "bootstrap": {"draws": args.draws if full else 0, "seed": SEED + 99, "unit": "source group"},
        "fold_meta": fold_meta, "seconds_total": time.time() - started,
    }
    (out / f"{args.tag}.json").write_text(json.dumps(clean(result), indent=1, sort_keys=True) + "\n")
    print(json.dumps(clean({"metrics": metrics, "replay": replay, "contrasts": contrasts,
                            "seconds": result["seconds_total"]}), indent=1))


if __name__ == "__main__":
    main()
