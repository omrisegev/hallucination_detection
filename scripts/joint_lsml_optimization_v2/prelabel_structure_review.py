"""Pre-label structural review of the frozen Joint L-SML v2 structure stage.

This is the label-free scientific assessment the protocol (Section 7.4 / 7.5 / 3.4) and the
2026-09-06 late integrity amendment require BEFORE the evaluator is allowed to open labels:

  1. admission census   — Amendment R2 coverage completeness per row (outer + inner lanes),
                          per-lane failures, `labels_accessed` flags;
  2. fallback rates     — INTERNAL / gated-affinity blocked lanes vs the Section-3.4 caps
                          (>10/40 PB, >1/5 PRMB → STRUCTURALLY_FRAGILE);
  3. stability          — gate seed std cap (0.15 → GATE_UNSTABLE), cross-fold map cosine
                          (>= 0.5 else UNSTABLE_MAP), gate inertness futility (cosine
                          gated-vs-ungated >= 0.995 on every lane → MECHANISM_INERT), small-m
                          census, INTERNAL K distribution;
  4. map agreement      — per-arm Spearman of held-out step scores vs the fixed-family control
                          (`prov5_cont`, Section 4.3 item 5), floor 0.50, per-arm flags only
                          (>=2 PB lanes or any PRMB fold → freeze barred);
  5. source/config      — constant seed / n_arms, deployed IU grid match, roster identity.

Reads ONLY: structure/*/outer*/{meta_outer.json, scores_outer.npz, inner*/...},
cells/*.npz (target-free bundles), folds/folds.json. Never touches labels/.
Writes:  prelabel_structure_review.json + prelabel_structure_review.md under --results-root.
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from spectral_utils.joint_lsml_v2_localization import (  # noqa: E402
    GATE_SEED_STD_CAP, IU_ROSTER, LSML_ROSTER, DEPLOYED_IU_ROW,
)

N_OUTER = 5
N_INNER = 5
PB_SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")
PB_MODELS = ("q4", "q8")
PRM_CELL = "prmbench_qwen3_8b"
PB_CELLS = [f"pb_{s}_{t}" for s in PB_SUBSETS for t in PB_MODELS]
FIXED_CONTROL = "prov5_cont"
MAP_AGREEMENT_FLOOR = 0.50
CROSS_FOLD_COSINE_FLOOR = 0.50
INERTNESS_COSINE = 0.995
FRAGILITY_CAP = {"processbench": 10, "prmbench": 1}
BASE_SEED = 20260905          # run_v2.SEED
N_ARMS_REGISTERED = 36        # 16 + 16 + 2 equal controls + 2 permutation controls
# gated row -> its ungated (lambda=0) frozen reference. Hook 3a/3b rows have no frozen
# lambda=0 model-inverse row (the identity was verified in the audit suite); they are
# compared against the hierarchical joint head they share the fit with, and disclosed.
INERTNESS_PAIRS = {
    "prov5_cont_gate050": "prov5_cont",
    "prov5_cont_gate100": "prov5_cont",
    "internal_cont_gate100": "internal_cont",
    "internal_joint_gate050": "internal_joint",
    "internal_joint_gate100": "internal_joint",
    "internal_gaff_cont": "internal_cont",
    "internal_gaff_joint": "internal_joint",
}
HOOK3_REF = "internal_joint_modelinv_lam0"   # Amendment R3 lambda=0 reference (scores_amend_r3.npz)
HOOK3_ROWS = ("internal_joint_liu010", "internal_joint_liu050",
              "internal_joint_diag010", "internal_joint_diag050")
INTERNAL_ROWS = ("internal_cont", "internal_joint", "internal_cont_gate100",
                 "internal_joint_gate050", "internal_joint_gate100", *HOOK3_ROWS)
GAFF_ROWS = ("internal_gaff_cont", "internal_gaff_joint")


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / den) if den > 0 else float("nan")


def _panel(cell_id: str) -> str:
    return "prmbench" if cell_id == PRM_CELL else "processbench"


def _step_rows(cell) -> np.ndarray:
    return np.repeat(np.arange(len(cell["row_ids"])), np.diff(cell["step_row_offsets"]))


def _held_step_mask(cell, fold_map: dict, held: int) -> np.ndarray:
    groups = [str(g) for g in cell["group_ids"]]
    row_test = np.asarray([fold_map.get(g, -1) == held for g in groups], bool)
    return row_test[_step_rows(cell)]


def review(root: Path) -> dict:
    folds = json.loads((root / "folds" / "folds.json").read_text(encoding="utf-8"))
    cells = PB_CELLS + [PRM_CELL]
    lsml_ids = [r for r, *_ in LSML_ROSTER] if isinstance(LSML_ROSTER[0], (tuple, list)) else list(LSML_ROSTER)
    iu_ids = [r for r, _ in IU_ROSTER]

    out: dict = {"cells": {}, "panels": {}, "rows": {}, "aborts": [], "flags": {}}
    per_row_flags: dict[str, set] = {}

    def flag(row, f):
        per_row_flags.setdefault(row, set()).add(f)

    # ── per-cell walk ────────────────────────────────────────────────────────
    lane_records = []          # (cell, k, meta)
    coverage = {p: {"lanes": 0, "present": {}} for p in ("processbench", "prmbench")}
    weights: dict[tuple[str, str], dict[int, np.ndarray]] = {}   # (cell,row) -> {k: w}
    map_agreement: dict[str, list] = {}                          # row -> [(cell,k,rho)]
    incomplete_folds = []
    for cell_id in cells:
        panel = _panel(cell_id)
        cov = coverage[panel]
        suffix = "spanmax" if panel == "prmbench" else "detector"
        cell = None
        for k in range(N_OUTER):
            base = root / "structure" / cell_id / f"outer{k}"
            if not (base / "COMPLETE.json").exists():
                incomplete_folds.append(f"{cell_id}/outer{k}")
                continue
            meta = json.loads((base / "meta_outer.json").read_text(encoding="utf-8"))
            lane_records.append((cell_id, k, meta))
            # coverage over outer + inner lanes (R2 rule, mirrors evaluate_v2)
            paths = [base / "scores_outer.npz"] + [
                base / f"inner{j}" / "scores_inner.npz" for j in range(N_INNER)]
            for path in paths:
                if not path.exists():
                    continue
                cov["lanes"] += 1
                keys = set(np.load(path, allow_pickle=False).files)
                for row in dict.fromkeys(lsml_ids + iu_ids + [FIXED_CONTROL]):
                    if f"{row}__{suffix}" in keys:
                        cov["present"][row] = cov["present"].get(row, 0) + 1
            # weights + map agreement on held-out steps
            z = np.load(base / "scores_outer.npz", allow_pickle=False)
            arms = sorted({key.rsplit("__", 1)[0] for key in z.files})
            for arm in arms:
                if f"{arm}__w" in z.files:
                    weights.setdefault((cell_id, arm), {})[k] = np.asarray(z[f"{arm}__w"], float)
            r3 = base / "scores_amend_r3.npz"
            if r3.exists():
                zr = np.load(r3, allow_pickle=False)
                if f"{HOOK3_REF}__w" in zr.files:
                    weights.setdefault((cell_id, HOOK3_REF), {})[k] = np.asarray(zr[f"{HOOK3_REF}__w"], float)
            score_key = "spanmax" if panel == "prmbench" else "top10"
            if f"{FIXED_CONTROL}__{score_key}" in z.files:
                if cell is None:
                    bundle = np.load(root / "cells" / f"{cell_id}.npz", allow_pickle=False)
                    cell = {key: bundle[key] for key in bundle.files}
                held = _held_step_mask(cell, folds[panel]["outer"], k)
                ref = np.asarray(z[f"{FIXED_CONTROL}__{score_key}"], float)[held]
                ref_ok = np.isfinite(ref).all() and np.std(ref) > 0
                for arm in arms:
                    if arm == FIXED_CONTROL or f"{arm}__{score_key}" not in z.files:
                        continue
                    s = np.asarray(z[f"{arm}__{score_key}"], float)[held]
                    rho = float(spearmanr(ref, s).correlation) if ref_ok and np.std(s) > 0 else float("nan")
                    map_agreement.setdefault(arm, []).append((cell_id, k, rho, bool(ref_ok)))

    out["incomplete_folds"] = incomplete_folds
    if incomplete_folds:
        out["aborts"].append(f"{len(incomplete_folds)} outer folds not frozen - review is PARTIAL")

    # ── 1. admission census ──────────────────────────────────────────────────
    admission = {}
    for panel, cov in coverage.items():
        lanes = cov["lanes"]
        eligible = [r for r in lsml_ids + iu_ids if cov["present"].get(r, 0) == lanes]
        incomplete = {r: cov["present"].get(r, 0) for r in lsml_ids + iu_ids
                      if cov["present"].get(r, 0) < lanes}
        admission[panel] = {
            "lanes": lanes,
            "eligible_lsml": [r for r in eligible if r in lsml_ids],
            "eligible_iu": [r for r in eligible if r in iu_ids],
            "incomplete": incomplete,
            "fixed_control_complete": cov["present"].get(FIXED_CONTROL, 0) == lanes,
        }
        for r in incomplete:
            flag(r, f"COVERAGE_INCOMPLETE[{panel}]")
    out["admission"] = admission

    labels_accessed = [f"{c}/outer{k}" for c, k, m in lane_records if m.get("labels_accessed")]
    if labels_accessed:
        out["aborts"].append(f"labels_accessed=True on {labels_accessed}")
    failures = {}
    for c, k, m in lane_records:
        for f in m.get("failures", []):
            row = f.get("row", f) if isinstance(f, dict) else str(f)
            failures.setdefault(str(row), []).append(f"{c}/outer{k}")
    out["outer_failures"] = failures

    # ── 2. fallback rates ────────────────────────────────────────────────────
    fallback = {}
    for panel in ("processbench", "prmbench"):
        recs = [(c, k, m) for c, k, m in lane_records if _panel(c) == panel]
        n = len(recs)
        internal_blocked = [f"{c}/outer{k}" for c, k, m in recs
                            if m.get("internal_grouping_status") != "SELECTED"]
        gaff_blocked = [f"{c}/outer{k}" for c, k, m in recs
                        if m.get("gated_affinity_grouping_status") != "SELECTED"]
        ks = [m.get("internal_K") for c, k, m in recs if m.get("internal_grouping_status") == "SELECTED"]
        kdist = {str(v): ks.count(v) for v in sorted(set(ks), key=lambda x: (x is None, x))}
        cap = FRAGILITY_CAP[panel]
        fb_rows = {}
        for c, k, m in recs:
            for ev in m.get("fallback_events", []):
                fb_rows.setdefault(ev["row"], []).append(f"{c}/outer{k}")
        fallback[panel] = {
            "lanes": n,
            "internal_blocked": internal_blocked,
            "internal_blocked_rate": f"{len(internal_blocked)}/{n}",
            "internal_fragile": len(internal_blocked) > cap,
            "gated_affinity_blocked_rate": f"{len(gaff_blocked)}/{n}",
            "gated_affinity_fragile": len(gaff_blocked) > cap,
            "cap": cap,
            "internal_K_distribution": kdist,
            "fallback_events_by_row": {r: len(v) for r, v in fb_rows.items()},
        }
        if len(internal_blocked) > cap:
            for r in INTERNAL_ROWS:
                flag(r, f"STRUCTURALLY_FRAGILE[{panel}]")
            out["aborts"].append(
                f"INTERNAL blocked-lane rate {len(internal_blocked)}/{n} above the Section-3.4 "
                f"cap ({cap}) on {panel} - Section 7.5 requires a registered amendment before labels")
        if len(gaff_blocked) > cap:
            for r in GAFF_ROWS:
                flag(r, f"STRUCTURALLY_FRAGILE[{panel}]")
    out["fallback"] = fallback

    # ── 3. stability ─────────────────────────────────────────────────────────
    seed_std = {f"{c}/outer{k}": m.get("gate_seed_std") for c, k, m in lane_records}
    bad_seed = {lane: v for lane, v in seed_std.items() if v is None or not np.isfinite(v) or v > GATE_SEED_STD_CAP}
    out["gate_seed_std"] = {
        "cap": GATE_SEED_STD_CAP,
        "max": max(v for v in seed_std.values() if v is not None),
        "violations": bad_seed,
    }
    if bad_seed:
        for r in list(INERTNESS_PAIRS) + list(HOOK3_ROWS) + ["dufs_pf_lsml"]:
            flag(r, "GATE_UNSTABLE")

    cross_fold = {}
    for (cell_id, arm), ws in weights.items():
        if len(ws) < 2:
            continue
        cs = [_cos(ws[a], ws[b]) for a, b in combinations(sorted(ws), 2)]
        cross_fold.setdefault(arm, {})[cell_id] = {"mean_pairwise_cosine": float(np.mean(cs)),
                                                   "min": float(np.min(cs)), "n_folds": len(ws)}
    unstable = {}
    for arm, per_cell in cross_fold.items():
        low = {c: v["mean_pairwise_cosine"] for c, v in per_cell.items()
               if v["mean_pairwise_cosine"] < CROSS_FOLD_COSINE_FLOOR}
        if low:
            unstable[arm] = low
            flag(arm, "UNSTABLE_MAP")
    out["cross_fold_map_cosine"] = {"floor": CROSS_FOLD_COSINE_FLOOR, "per_arm": cross_fold,
                                    "unstable": unstable}

    inert = {}
    for gated, ref in INERTNESS_PAIRS.items():
        vals = []
        for (cell_id, arm), ws in weights.items():
            if arm != gated:
                continue
            for k, w in ws.items():
                wr = weights.get((cell_id, ref), {}).get(k)
                if wr is not None:
                    vals.append((f"{cell_id}/outer{k}", _cos(w, wr)))
        if vals:
            cs = np.array([v for _, v in vals])
            inert[gated] = {"reference": ref, "n_lanes": len(vals), "min_cosine": float(cs.min()),
                            "median_cosine": float(np.median(cs)),
                            "inert_on_every_lane": bool((cs >= INERTNESS_COSINE).all()),
                            "lanes_at_or_above_0995": int((cs >= INERTNESS_COSINE).sum())}
            if inert[gated]["inert_on_every_lane"]:
                flag(gated, "MECHANISM_INERT")
    hook3 = {}
    have_ref = any(arm == HOOK3_REF for (_, arm) in weights)
    for row in HOOK3_ROWS:
        vals, vals_head, n_lanes_row = [], [], 0
        for (cell_id, arm), ws in weights.items():
            if arm != row:
                continue
            for k, w in ws.items():
                n_lanes_row += 1
                wr = weights.get((cell_id, HOOK3_REF), {}).get(k)
                if wr is not None:
                    vals.append(_cos(w, wr))
                wh = weights.get((cell_id, "internal_joint"), {}).get(k)
                if wh is not None:
                    vals_head.append(_cos(w, wh))
        entry = {"vs_hierarchical_head_median_cosine": float(np.median(vals_head)) if vals_head else None}
        if have_ref and vals:
            cs = np.array(vals)
            entry.update({"reference": HOOK3_REF, "n_lanes": len(vals), "n_lanes_row": n_lanes_row,
                          "min_cosine": float(cs.min()), "median_cosine": float(np.median(cs)),
                          "lanes_at_or_above_0995": int((cs >= INERTNESS_COSINE).sum()),
                          "inert_on_every_lane": bool((cs >= INERTNESS_COSINE).all()) and len(vals) == n_lanes_row})
            if entry["inert_on_every_lane"]:
                flag(row, "MECHANISM_INERT")
            elif len(vals) < n_lanes_row:
                flag(row, "HOOK3_REFERENCE_INCOMPLETE")
        else:
            entry.update({"reference": "MISSING (run third_pass_amendment_r3.py) - guard NOT applied",
                          "n_lanes": 0})
            flag(row, "HOOK3_INERTNESS_UNASSESSED")
        hook3[row] = entry
    dose = {}
    for a, b in (("internal_joint_liu010", "internal_joint_liu050"),
                 ("internal_joint_diag010", "internal_joint_diag050"),
                 ("prov5_cont_gate050", "prov5_cont_gate100"),
                 ("internal_joint_gate050", "internal_joint_gate100")):
        vals = [_cos(weights[(c, a)][k], weights[(c, b)][k])
                for (c, arm) in weights if arm == a
                for k in weights[(c, a)] if k in weights.get((c, b), {})]
        if vals:
            dose[f"{a}~{b}"] = {"median_cosine": float(np.median(vals)), "min": float(np.min(vals))}
    out["inertness"] = {"threshold": INERTNESS_COSINE, "hook1_hook2": inert,
                        "hook3_vs_lambda0_reference": hook3, "dose_pairs": dose}

    small_m = {}
    for c, k, m in lane_records:
        for row, rm in m.get("row_meta", {}).items():
            if rm.get("small_m_guarded") or rm.get("cross_small_m_guarded") or rm.get("small_m_flags"):
                small_m.setdefault(row, {"guarded_lanes": 0, "flagged_lanes": 0})
                if rm.get("small_m_guarded") or rm.get("cross_small_m_guarded"):
                    small_m[row]["guarded_lanes"] += 1
                if rm.get("small_m_flags"):
                    small_m[row]["flagged_lanes"] += 1
    out["small_m_census"] = small_m

    # ── 4. map agreement ─────────────────────────────────────────────────────
    agree = {}
    for arm, recs in map_agreement.items():
        pb_viol = [f"{c}/outer{k}" for c, k, rho, ok in recs if _panel(c) == "processbench" and ok and (not np.isfinite(rho) or rho < MAP_AGREEMENT_FLOOR)]
        prm_viol = [f"{c}/outer{k}" for c, k, rho, ok in recs if _panel(c) == "prmbench" and ok and (not np.isfinite(rho) or rho < MAP_AGREEMENT_FLOOR)]
        degenerate_ref = [f"{c}/outer{k}" for c, k, rho, ok in recs if not ok]
        rhos = np.array([rho for _, _, rho, ok in recs if ok])
        agree[arm] = {"n_lanes": len(recs), "min_spearman": float(np.nanmin(rhos)) if len(rhos) else None,
                      "median_spearman": float(np.nanmedian(rhos)) if len(rhos) else None,
                      "pb_violations": pb_viol, "prm_violations": prm_viol,
                      "control_degenerate_lanes": degenerate_ref,
                      "freeze_barred": len(pb_viol) >= 2 or len(prm_viol) >= 1}
        if agree[arm]["freeze_barred"]:
            flag(arm, "MAP_AGREEMENT_FREEZE_BARRED")
    out["map_agreement"] = {"floor": MAP_AGREEMENT_FLOOR, "reference": FIXED_CONTROL,
                            "score": "top10 (PB) / spanmax (PRMB) on held-out steps", "per_arm": agree}

    # ── 5. source / config fidelity ──────────────────────────────────────────
    # run_v2.SEED + 100*k per outer lane; n_arms = 36 registered arms minus fail-closed rows
    seed_bad = [f"{c}/outer{k}" for c, k, m in lane_records if m.get("seed") != BASE_SEED + 100 * k]
    arms_bad = [f"{c}/outer{k}: n_arms={m.get('n_arms')} failures={len(m.get('failures', []))}"
                for c, k, m in lane_records if m.get("n_arms") != N_ARMS_REGISTERED - len(m.get("failures", []))]
    iu_grid = [f"{c}/outer{k}" for c, k, m in lane_records if not m.get("deployed_iu_matches_grid", False)]
    roster_seen = sorted({r for (_, r) in weights})
    missing_roster = [r for r in lsml_ids + iu_ids if r not in roster_seen]
    out["fidelity"] = {"seed_rule": f"{BASE_SEED} + 100*outer", "seed_violations": seed_bad,
                       "n_arms_rule": f"{N_ARMS_REGISTERED} - fail_closed", "n_arms_violations": arms_bad,
                       "deployed_iu_row": DEPLOYED_IU_ROW,
                       "deployed_iu_grid_mismatch_lanes": iu_grid,
                       "roster_rows_missing_from_all_lanes": missing_roster,
                       "frozen_arm_ids": roster_seen}
    if seed_bad or arms_bad or iu_grid or missing_roster:
        out["aborts"].append("source/config fidelity issue - see fidelity block")

    out["flags"] = {r: sorted(f) for r, f in sorted(per_row_flags.items())}
    out["late_freeze_limitation"] = (
        "EXECUTION_REGISTRY.json was not written at launch; source/config hashes are bound by "
        "freeze_integrity.py AFTER the structure stage (dated 2026-09-06). Lineage rests on the "
        "committed producer revision plus the per-fold manifests, not on a launch-time freeze. "
        "This limitation is retained verbatim in the results report."
    )
    return out


def _md(rep: dict) -> str:
    L = ["# Pre-label structural review — Joint L-SML v2", ""]
    L.append(f"Incomplete folds: {len(rep['incomplete_folds'])}  |  aborts: {len(rep['aborts'])}")
    for a in rep["aborts"]:
        L.append(f"- ABORT/AMEND: {a}")
    L += ["", "## Admission (R2 coverage completeness)"]
    for p, a in rep["admission"].items():
        L.append(f"- {p}: {a['lanes']} lanes; eligible LSML {len(a['eligible_lsml'])}/16, IU {len(a['eligible_iu'])}/16; "
                 f"incomplete: {a['incomplete'] or 'none'}; fixed control complete: {a['fixed_control_complete']}")
    L += ["", "## Fallback rates (Section 3.4 caps)"]
    for p, f in rep["fallback"].items():
        L.append(f"- {p}: INTERNAL blocked {f['internal_blocked_rate']} (cap {f['cap']}) fragile={f['internal_fragile']}; "
                 f"gated-affinity blocked {f['gated_affinity_blocked_rate']} fragile={f['gated_affinity_fragile']}; "
                 f"K dist {f['internal_K_distribution']}")
    g = rep["gate_seed_std"]
    L += ["", "## Stability", f"- gate seed std max {g['max']:.4f} (cap {g['cap']}); violations: {g['violations'] or 'none'}"]
    L.append(f"- cross-fold map cosine (floor {rep['cross_fold_map_cosine']['floor']}): unstable arms: "
             f"{ {a: {c: round(v,3) for c, v in d.items()} for a, d in rep['cross_fold_map_cosine']['unstable'].items()} or 'none'}")
    L.append("- inertness (Hook 1/2 rows vs lambda=0 reference):")
    for r, v in rep["inertness"]["hook1_hook2"].items():
        L.append(f"  - {r} vs {v['reference']}: min cos {v['min_cosine']:.4f}, median {v['median_cosine']:.4f}, "
                 f"lanes >=0.995: {v['lanes_at_or_above_0995']}/{v['n_lanes']} -> inert={v['inert_on_every_lane']}")
    for r, v in rep["inertness"]["hook3_vs_lambda0_reference"].items():
        if v.get("n_lanes"):
            L.append(f"  - {r} vs {v['reference']}: min cos {v['min_cosine']:.4f}, median {v['median_cosine']:.4f}, "
                     f"lanes >=0.995: {v['lanes_at_or_above_0995']}/{v['n_lanes_row']} -> inert={v['inert_on_every_lane']}"
                     f" (vs hierarchical head median {v['vs_hierarchical_head_median_cosine']:.3f})")
        else:
            L.append(f"  - {r}: {v['reference']} (vs hierarchical head median {v['vs_hierarchical_head_median_cosine']:.3f})")
    L.append(f"- dose pairs: { {k: round(v['median_cosine'],4) for k, v in rep['inertness']['dose_pairs'].items()} }")
    L.append(f"- small-m census (rows with any guard/flag): { {r: v for r, v in rep['small_m_census'].items()} }")
    L += ["", f"## Map agreement vs {rep['map_agreement']['reference']} (floor {rep['map_agreement']['floor']})"]
    for arm, v in sorted(rep["map_agreement"]["per_arm"].items()):
        mark = " **FREEZE BARRED**" if v["freeze_barred"] else ""
        L.append(f"- {arm}: median {v['median_spearman']:.3f}, min {v['min_spearman']:.3f}, "
                 f"PB viol {len(v['pb_violations'])}, PRM viol {len(v['prm_violations'])}{mark}")
    fd = rep["fidelity"]
    L += ["", "## Source/config fidelity",
          f"- seed rule {fd['seed_rule']} violations: {fd['seed_violations'] or 'none'}; n_arms rule {fd['n_arms_rule']} "
          f"violations: {fd['n_arms_violations'] or 'none'}; deployed IU grid mismatches: {fd['deployed_iu_grid_mismatch_lanes'] or 'none'}, "
          f"roster rows missing: {fd['roster_rows_missing_from_all_lanes'] or 'none'}"]
    L += ["", "## Per-row flags"]
    for r, fl in rep["flags"].items():
        L.append(f"- {r}: {', '.join(fl)}")
    if not rep["flags"]:
        L.append("- none")
    L += ["", "## Late-freeze limitation", rep["late_freeze_limitation"], ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default=str(REPO / "results" / "joint_lsml_optimization_v2"))
    args = ap.parse_args()
    root = Path(args.results_root)
    rep = review(root)
    (root / "prelabel_structure_review.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    md = _md(rep)
    (root / "prelabel_structure_review.md").write_text(md, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(md)


if __name__ == "__main__":
    main()
