#!/usr/bin/env python
"""Item 2 (2026-09-23): CT7 family-equal — partition the seven frozen CT7 views into declared
families, equal within, equal across; compared with CT7 (1/7 each) on the frozen population.

Protocol: docs/experiments/CT7_FAMILY_EQUAL_V1.md (written before any number).
Inputs: the frozen `profiles.npy` of `results/cumulative_vote_fusion_v2/ct7_profiles_v1/`
(sha checked against PROFILE_VALIDATION.json), CT7_DEV_SCORES.npz (frozen gate and scores),
the roster and folds. No fit reads a label. No file inside cvf_v2/ is edited or written.

    python -B scripts/experiments/ct7_family_equal_v1.py --config configs/ct7_family_equal_v1.json
    python -B scripts/experiments/ct7_family_equal_v1.py --dry-run     # synthetic, no data
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ct7_levers_common as L  # noqa: E402

L.ensure_spectral_package()
from spectral_utils.family_equal_readout import (  # noqa: E402
    FAMILIES_421, FAMILIES_511, _groups_from_families, discover_partition, family_means, partition_scores,
)

SCHEMA = "ct7-family-equal-v1"


def fit_free_arm(profiles, groups, off, rule):
    return partition_scores(profiles, groups, off, rule=rule)


def fold_arm(d, profiles, groups, rule):
    """Per outer source fold: scale/eigen fitted on the training answers' steps, applied to test."""
    total = int(d.off[-1]); score = np.full(total, np.nan); info = {}
    step_fold = np.repeat(d.fold, np.diff(d.off))
    for f in range(5):
        train = step_fold != f; test = ~train
        s, meta = partition_scores(profiles, groups, d.off, rule=rule, train_rows=train)
        score[test] = s[test]; info[f"fold{f}"] = meta
    assert np.isfinite(score).all()
    return score, info


def auto_arm(d, profiles, k_range, min_size, seed):
    total = int(d.off[-1]); s_fold = np.full(total, np.nan); s_ans = np.full(total, np.nan); info = {}
    step_fold = np.repeat(d.fold, np.diff(d.off))
    for f in range(5):
        train = step_fold != f; test = ~train
        labels, diag = discover_partition(profiles[train], step_fold[train], k_range=k_range, min_size=min_size, seed=seed)
        sf, mf = partition_scores(profiles, labels, d.off, rule="fold", train_rows=train)
        sa, ma = partition_scores(profiles, labels, d.off, rule="answer")
        s_fold[test] = sf[test]; s_ans[test] = sa[test]
        info[f"fold{f}"] = {"labels": labels.tolist(), "selected_k": diag["selected_k"],
                            "candidates": {str(k): {kk: vv for kk, vv in v.items() if kk != "labels"} | {"labels": v["labels"]}
                                           for k, v in diag["candidates"].items()}, "family_sd_train": mf["family_sd_train"]}
    assert np.isfinite(s_fold).all() and np.isfinite(s_ans).all()
    return s_fold, s_ans, info


def build_methods(d, profiles, cfg, step_lengths=None):
    off = d.off; m = profiles.shape[1]
    partitions = {"421": _groups_from_families(FAMILIES_421, m), "511": _groups_from_families(FAMILIES_511, m)}
    scores, fits = {}, {}
    ct7 = profiles.mean(1)
    scores["ct7"] = ct7
    scores["ct7_z"] = L.answer_z(ct7, off)
    for tag, g in partitions.items():
        for rule in ("raw", "answer"):
            s, info = fit_free_arm(profiles, g, off, rule); scores[f"fam{tag}_{rule}"] = s; fits[f"fam{tag}_{rule}"] = info
        for rule in ("fold", "eigen"):
            s, info = fold_arm(d, profiles, g, rule); scores[f"fam{tag}_{rule}"] = s; fits[f"fam{tag}_{rule}"] = info
    sf, sa, info = auto_arm(d, profiles, tuple(cfg.get("auto_k_range", [2, 3, 4])), int(cfg.get("auto_min_size", 1)), int(cfg.get("seed", 20260923)))
    scores["fam_auto_fold"] = sf; scores["fam_auto_answer"] = sa; fits["fam_auto"] = info
    g421 = partitions["421"]
    scores["six_equal"] = profiles[:, :6].mean(1)
    scores["level_only"] = profiles[:, g421 == 0].mean(1)
    scores["temporal_only"] = profiles[:, g421 == 1].mean(1)
    for j, name in enumerate(L.CT7_VIEWS[:m]):
        scores["ct7_single__" + name] = profiles[:, j]
    if step_lengths is not None:
        scores["control__longest_step"] = np.asarray(step_lengths, float)
    methods = {}
    for name, s in scores.items():
        # every arm's scores are answer-standardized for the pooled endpoints (PRMScore); argmax
        # and within-answer AUROC are invariant, checked here for the reference row
        z = L.answer_z(s, off) if name != "ct7" else s
        methods[name] = L.method_from_scores(d, z)
        if name != "ct7":
            assert np.array_equal(methods[name]["pred"], d.peaks(s)), name
    return methods, fits


def planned_contrasts(names):
    pairs = []
    def add(a, b, why):
        if a in names and b in names and (a, b, why) not in pairs:
            pairs.append((a, b, why))
    for tag in ("421", "511"):
        for rule in ("raw", "answer", "fold", "eigen"):
            add(f"fam{tag}_{rule}", "ct7", "family_minus_ct7")
    for rule in ("answer", "fold"):
        add(f"fam421_{rule}", "fam421_raw", "scaling_minus_partition_only")
        add(f"fam421_{rule}", f"fam511_{rule}", "partition_421_minus_511")
    add("fam_auto_fold", "fam421_fold", "discovered_minus_declared")
    add("fam_auto_answer", "fam421_answer", "discovered_minus_declared")
    add("fam421_eigen", "fam421_answer", "eigen_minus_equal_across")
    add("fam421_answer", "six_equal", "family_minus_six_equal")
    add("level_only", "ct7", "level_family_minus_ct7"); add("temporal_only", "ct7", "temporal_family_minus_ct7")
    return pairs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config"); p.add_argument("--dry-run", action="store_true"); p.add_argument("--draws", type=int)
    args = p.parse_args()
    started = time.perf_counter()
    if args.dry_run:
        tmp = Path(tempfile.mkdtemp(prefix="ct7_family_equal_dry_"))
        d, profiles = L.synthetic_dataset(tmp, n_answers=220)
        cfg = dict(d.c); step_lengths = None; inputs = []
        out = d.out
    else:
        d = L.light_dataset(args.config)
        cfg = d.c; paths = cfg["paths"]
        profiles = L.load_ct7_profiles(d, paths["profiles"], paths.get("profile_validation"))
        step_lengths = np.load(paths["step_lengths"]) if paths.get("step_lengths") and Path(paths["step_lengths"]).exists() else None
        inputs = [Path(paths[k]) for k in ("roster", "joined", "folds", "ct7", "profiles") if k in paths]
        out = d.out
        L.run_freeze(out, [Path(__file__), L.ROOT / "scripts/experiments/ct7_levers_common.py",
                           L.ROOT / "spectral_utils/family_equal_readout.py", Path(args.config).resolve()],
                     inputs, {"schema": SCHEMA, "candidate_id": cfg["candidate_id"], "development_only": True})
    if step_lengths is not None:
        assert step_lengths.shape == (int(d.off[-1]),)
    methods, fits = build_methods(d, profiles, cfg, step_lengths)
    if not args.dry_run:
        L.replay_ct7(d)
    contrasts = planned_contrasts(list(methods))
    primary = [("fam421_answer", "ct7", "PRIMARY_11plus"), ("fam421_fold", "ct7", "family_minus_ct7"),
               ("fam421_raw", "ct7", "family_minus_ct7"), ("fam511_answer", "ct7", "family_minus_ct7"),
               ("fam_auto_answer", "ct7", "discovered_minus_ct7"), ("fam421_answer", "fam421_raw", "scaling_minus_partition_only")]
    result = L.evaluate_methods(d, methods, contrasts_extra=contrasts, strata_contrasts=primary + contrasts,
                                prmscore=not args.dry_run, draws=args.draws)
    rows = L.summary_rows(d, methods, result)
    L.write_summary_csv(out / "SUMMARY.csv", rows)
    coverage = {"answers": d.n, "steps": int(d.off[-1]),
                "two_step_answers": int(np.sum(np.diff(d.off) == 2)),
                "zeroed_family_answers": {k: v.get("zeroed_family_answers") for k, v in fits.items() if isinstance(v, dict) and "zeroed_family_answers" in v},
                "gate_open_pb": int(d.gate[d.pb].sum()), "gate_open_prm": int(d.gate[d.prm].sum())}
    record = {"schema": SCHEMA, "development_only": True, "note": L.DEVELOPMENT_NOTE,
              "families": {"421": FAMILIES_421, "511": FAMILIES_511}, "views": list(L.CT7_VIEWS),
              "access": {"fit_free": ["fam*_raw", "fam*_answer", "six_equal", "level_only", "temporal_only", "ct7", "ct7_z"],
                         "pooled_donor_fold_fits_label_free": ["fam*_fold", "fam*_eigen", "fam_auto_*"]},
              "fits": fits, "pb": result["pb"], "prm": {k: v for k, v in result["prm"].items()},
              "strata": result["strata"], "coverage": coverage, "timing": result["timing"],
              "total_seconds": time.perf_counter() - started, "dry_run": bool(args.dry_run)}
    L.dump(out / "RESULTS.json", record)
    print(f"{'method':28s} {'SLA':>7s} {'F1':>7s} {'within':>7s} {'early':>6s} {'late':>6s} {'11+':>7s}")
    for r in rows:
        s11 = r.get("sla_steps_11_plus"); s11 = f"{100*s11:7.2f}" if s11 is not None else "      -"
        print(f"{r['method']:28s} {100*r['sla_macro8']:7.2f} {100*r['f1_common_gate']:7.2f} {r['within_auc']:7.4f} "
              f"{r['early']:6.3f} {r['late']:6.3f} {s11}")
    for c in result["uncertainty"]["contrasts"]:
        if c["a"] in ("fam421_answer", "fam421_fold", "fam421_raw") and c["b"] == "ct7":
            print(f"{c['endpoint']:18s} {c['a']:>16s} - ct7: {c['delta']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}] holm {c['p_holm']:.3f}")
    for stratum, tab in result["strata"].items():
        for c in tab["contrasts"]:
            if c["contrast"] == "PRIMARY_11plus":
                print(f"{stratum:16s} fam421_answer - ct7: {c['delta_pp']:+.2f} pp [{c['ci95_pp'][0]:+.2f}, {c['ci95_pp'][1]:+.2f}]")
    print("written:", out, f"({time.perf_counter() - started:.1f}s)")


if __name__ == "__main__":
    main()
