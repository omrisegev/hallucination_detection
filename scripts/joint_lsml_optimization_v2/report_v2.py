"""Post-evaluation report for the Joint L-SML optimization v2 study.

Runs AFTER `evaluate_v2.py` (the gated evaluator) has written `evaluation/headline.json`,
`inner_selection.json`, `moduleb.json`. Reuses the evaluator's panel classes verbatim so every
number here is the same estimator as the headline; adds only what the protocol requires for the
freeze / attribution decisions and the descriptive tables:

  tables     per-arm outer-refit metrics for every frozen row on both panels (PB cross-fitted
             macro-F1 + error-side / clean-side components + activation guard; PRMB AUROC),
             selection-frequency tables, UNSTABLE_SELECTION qualifiers
  contrasts  extra paired grouped bootstraps (same seed / draws as the evaluator):
             PRMB  permutation-control attribution (Section 7.3), lambda=0 reference (R3),
                   continuity row vs deployed IU, guard cost (prov5_cont vs unguarded)
             PB    tuned-Joint vs deployed IU and most-selected Joint config vs deployed IU
                   (Section 7.2 non-inferiority floor), guard cost
  render     REPORT.md from evaluation/*.json + the pre-label review + the integrity record

    python scripts/joint_lsml_optimization_v2/report_v2.py --stage tables|contrasts|render|all
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import evaluate_v2 as ev  # noqa: E402
from spectral_utils.joint_lsml_integrity import verify_prelabel_record  # noqa: E402
from spectral_utils.joint_lsml_v2_localization import (  # noqa: E402
    DEPLOYED_IU_ROW, DEPLOYED_UPCR_PORT_ROW, HOOK3_LAMBDA0_REFERENCE_ROW, IU_ROSTER,
    LSML_ROSTER, SUCCESSOR_S1, SUCCESSOR_S2,
)

OUT = ev.OUT
EVAL = OUT / "evaluation"
TABLES = EVAL / "report_tables.json"
CONTRASTS = EVAL / "report_contrasts.json"
CONTINUITY = "fixed_family_cont_unguarded"
PERMCTL_GRAPH = "permctl_graph_internal_joint_liu010"
PERMCTL_GATE = "permctl_gate_prov5_cont"
ROW_LABEL = {
    "prov5_cont": "R1 provenance CONT L-SML (lambda=0 anchor = fixed-family control)",
    "prov5_joint": "R2 provenance-merged hierarchical Joint",
    "internal_cont": "R3 INTERNAL CONT L-SML (= S2)",
    "internal_joint": "R4 INTERNAL hierarchical Joint (= S1)",
    "prov5_cont_gate050": "R5 Hook 2 congruence, provenance CONT, lambda=0.5",
    "prov5_cont_gate100": "R6 Hook 2 congruence, provenance CONT, lambda=1",
    "internal_cont_gate100": "R7 Hook 2 congruence, INTERNAL CONT, lambda=1",
    "internal_joint_gate050": "R8 Hook 2 on joint fit, lambda=0.5",
    "internal_joint_gate100": "R9 Hook 2 on joint fit, lambda=1",
    "internal_joint_liu010": "R10 Hook 3a LIU-transplant model-inverse, lambda=0.1",
    "internal_joint_liu050": "R11 Hook 3a LIU-transplant model-inverse, lambda=0.5",
    "internal_joint_diag010": "R12 Hook 3b diagonal gate prior, lambda=0.1",
    "internal_joint_diag050": "R13 Hook 3b diagonal gate prior, lambda=0.5",
    "internal_gaff_cont": "R14 Hook 1 gated-affinity grouping, CONT",
    "internal_gaff_joint": "R15 Hook 1 gated-affinity grouping, Joint",
    "dufs_pf_lsml": "R16 historical hard DUFS-PF selector + CONT",
    "equal_all23": "control: equal weights over all 23",
    "equal_family_active23": "control: equal-family",
    CONTINUITY: "continuity: historical unguarded fixed-family CONT (Amendment R1)",
    HOOK3_LAMBDA0_REFERENCE_ROW: "R3 reference: ungated model-inverse map (lambda=0)",
    PERMCTL_GATE: "negative control: feature-permuted gates on R6",
    PERMCTL_GRAPH: "negative control: node-relabeled graph on R10",
}


def _scores_with_r3(cell_id: str, outer: int, inner: int | None = None):
    scores = _orig_scores(cell_id, outer, inner)
    if inner is None:
        path = OUT / "structure" / cell_id / f"outer{outer}" / "scores_amend_r3.npz"
        if path.exists():
            extra = np.load(path, allow_pickle=False)
            scores.update({key: extra[key] for key in extra.files})
    return scores


_orig_scores = ev._scores
ev._scores = _scores_with_r3


def _open() -> tuple[dict, dict, dict]:
    cell_ids = [f"pb_{s}_{t}" for s in ev.PB_SUBSETS for t in ev.PB_MODELS] + [ev.PRM_CELL]
    verify_prelabel_record(OUT, cell_ids, n_outer=ev.N_OUTER, n_inner=ev.N_INNER)
    ev._PRELABEL_VERIFIED = True
    folds = ev._folds()
    cells = {c: ev._cell(c) for c in cell_ids}
    labels = {c: ev._labels(c) for c in cell_ids}
    return folds, cells, labels


def _all_arms() -> list[str]:
    keys = set(ev._scores(ev.PRM_CELL, 0).keys()) & set(ev._scores("pb_gsm8k_q4", 0).keys())
    return sorted({k.rsplit("__", 1)[0] for k in keys})


def _pb_components(panel: ev.PBPanel) -> dict:
    """Cross-fitted error-side hit rate and clean-side rate (macro over cells) + guard."""
    hit_all = np.zeros(len(panel.detector), dtype=bool)
    clean_pred = np.zeros(len(panel.detector), dtype=bool)
    for k in range(ev.N_OUTER):
        train = panel.fold != k
        test = ~train
        tau = ev._pb_threshold(panel.detector[train], panel.hit[train], panel.clean[train], panel.cell[train])
        flagged = panel.detector[test] >= tau
        hit_all[test] = panel.hit[test] & flagged
        clean_pred[test] = ~flagged
    wa, wc = [], []
    for cell_index in range(int(panel.cell.max()) + 1):
        rows = panel.cell == cell_index
        err, cln = rows & ~panel.clean, rows & panel.clean
        wa.append(float(hit_all[err].mean()))
        wc.append(float(clean_pred[cln].mean()))
    return {"error_side_hit_rate": float(np.mean(wa)), "clean_side_rate": float(np.mean(wc))}


def _activation_guard(activation: dict, control: dict) -> dict:
    viol = [lane for lane, x in activation.items() if x < max(0.10, 0.5 * control[lane])]
    cells = {lane.split("_")[0] for lane in viol}
    return {"violating_lanes": len(viol), "violating_cells": len(cells),
            "verdict": "CATASTROPHE" if len(cells) >= 2 else "ok",
            "mean_activation": float(np.mean(list(activation.values())))}


def stage_tables() -> None:
    folds, cells, labels = _open()
    arms = _all_arms()
    selection = json.loads((EVAL / "inner_selection.json").read_text(encoding="utf-8"))
    out = {"prmbench": {}, "processbench": {}, "selection": {}, "arms": arms}
    control_act = None
    for arm in arms:
        prm = ev.PRMPanel(folds, cells[ev.PRM_CELL], labels[ev.PRM_CELL], arm=arm)
        out["prmbench"][arm] = {"auroc": prm.auroc()}
    pb_cache = {}
    for arm in arms:
        panel = ev.PBPanel(arm, folds, cells, labels)
        f1, extras = panel.crossfit_macro_f1()
        pb_cache[arm] = (f1, extras["activation"], _pb_components(panel))
    control_act = pb_cache["prov5_cont"][1]
    for arm, (f1, act, comp) in pb_cache.items():
        out["processbench"][arm] = {"macro_f1": f1, **comp, **_activation_guard(act, control_act)}
    for panel in ("processbench", "prmbench"):
        sel = selection[panel]
        for family in ("lsml", "iu"):
            counts = Counter(sel[k][family] for k in sel)
            top, n = counts.most_common(1)[0]
            out["selection"][f"{panel}_{family}"] = {
                "counts": dict(counts), "most_selected": top, "most_selected_count": n,
                "unstable_selection": n < 3,
            }
    TABLES.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("tables written:", len(arms), "arms")


def stage_contrasts() -> None:
    folds, cells, labels = _open()
    selection = json.loads((EVAL / "inner_selection.json").read_text(encoding="utf-8"))
    tables = json.loads(TABLES.read_text(encoding="utf-8"))
    out = {"prmbench": {}, "processbench": {}, "n_boot": {"pb": ev.PB_BOOT, "prm": ev.PRM_BOOT},
           "seed": ev.BOOT_SEED}
    prm = lambda arm=None, by_fold=None: ev.PRMPanel(  # noqa: E731
        folds, cells[ev.PRM_CELL], labels[ev.PRM_CELL], arm=arm, selected_by_fold=by_fold)
    tuned_iu_prm = prm(by_fold={int(k): selection["prmbench"][k]["iu"] for k in selection["prmbench"]})
    for name, (a, b) in {
        "permctl_graph_vs_tuned_iu": (prm(PERMCTL_GRAPH), tuned_iu_prm),
        "lambda0_reference_vs_tuned_iu": (prm(HOOK3_LAMBDA0_REFERENCE_ROW), tuned_iu_prm),
        "liu010_vs_lambda0_reference": (prm("internal_joint_liu010"), prm(HOOK3_LAMBDA0_REFERENCE_ROW)),
        "continuity_vs_deployed_iu": (prm(CONTINUITY), prm(DEPLOYED_IU_ROW)),
        "guarded_vs_unguarded_fixed_family": (prm("prov5_cont"), prm(CONTINUITY)),
        "deployed_upcr_port_vs_deployed_iu": (prm(DEPLOYED_UPCR_PORT_ROW), prm(DEPLOYED_IU_ROW)),
    }.items():
        mean, ci = ev._prm_paired_bootstrap(a, b, n_boot=ev.PRM_BOOT, seed=ev.BOOT_SEED)
        out["prmbench"][name] = {"delta": mean, "ci95": ci}
        print("PRM", name, round(mean, 4), [round(c, 4) for c in ci], flush=True)
    CONTRASTS.write_text(json.dumps(out, indent=1), encoding="utf-8")
    pb = lambda arm="", by_fold=None: ev.PBPanel(arm, folds, cells, labels, selected_by_fold=by_fold)  # noqa: E731
    tuned_lsml_pb = pb(by_fold={int(k): selection["processbench"][k]["lsml"] for k in selection["processbench"]})
    most = tables["selection"]["processbench_lsml"]["most_selected"]
    deployed = pb(DEPLOYED_IU_ROW)
    for name, (a, b) in {
        "tuned_lsml_vs_deployed_iu": (tuned_lsml_pb, deployed),
        f"{most}_vs_deployed_iu": (pb(most), deployed),
        "guarded_vs_unguarded_fixed_family": (pb("prov5_cont"), pb(CONTINUITY)),
    }.items():
        mean, ci = ev._pb_paired_bootstrap(a, b, n_boot=ev.PB_BOOT, seed=ev.BOOT_SEED)
        out["processbench"][name] = {"delta": mean, "ci95": ci}
        print("PB", name, round(mean, 4), [round(c, 4) for c in ci], flush=True)
        CONTRASTS.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("contrasts written")


def _fmt_ci(entry: dict, digits: int = 4) -> str:
    lo, hi = entry["ci95"]
    return f"{entry['delta']:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]"


def stage_render() -> None:
    headline = json.loads((EVAL / "headline.json").read_text(encoding="utf-8"))
    selection = json.loads((EVAL / "inner_selection.json").read_text(encoding="utf-8"))
    moduleb = json.loads((EVAL / "moduleb.json").read_text(encoding="utf-8"))
    tables = json.loads(TABLES.read_text(encoding="utf-8"))
    contrasts = json.loads(CONTRASTS.read_text(encoding="utf-8")) if CONTRASTS.exists() else {"prmbench": {}, "processbench": {}}
    review = json.loads((OUT / "prelabel_structure_review.json").read_text(encoding="utf-8"))
    record = json.loads((OUT / "INTEGRITY_RECORD_V2.json").read_text(encoding="utf-8"))
    flags = review.get("flags", {})
    sel = tables["selection"]
    prm_c, pb_c = contrasts["prmbench"], contrasts["processbench"]
    prm_h, pb_h = headline["prmbench"], headline["processbench"]
    prm_t, pb_t = tables["prmbench"], tables["processbench"]
    lsml_ids = list(LSML_ROSTER)
    iu_ids = [r for r, _ in IU_ROSTER]
    most_pb = sel["processbench_lsml"]["most_selected"]
    most_prm = sel["prmbench_lsml"]["most_selected"]

    # Section 7.3 attribution on the PRMB SUPPORT
    attribution = "n/a"
    if headline["gates"]["development_prm"] == "SUPPORT" and most_prm in ("internal_joint_liu010", "internal_joint_liu050"):
        pc = prm_c.get("permctl_graph_vs_tuned_iu")
        if pc is not None:
            control_passes = pc["ci95"][0] > 0 and pc["delta"] >= ev.PRM_FLOOR
            attribution = "MECHANISM_UNATTRIBUTED (graph-permutation control passes the same gate)" if control_passes \
                else "MECHANISM_ATTRIBUTED (graph-permutation control fails the gate)"
    # Section 7.2 freeze
    freeze = {}
    prm_ok = prm_h["tuned_lsml_vs_tuned_iu"]["ci95"][0] > ev.PRM_NONINF  # tuned IU == deployed IU on PRMB 5/5
    freeze["prmbench"] = {"config": most_prm, "selected": f"{sel['prmbench_lsml']['most_selected_count']}/5",
                          "noninferior_vs_deployed_iu": bool(prm_ok and sel["prmbench_iu"]["most_selected"] == DEPLOYED_IU_ROW),
                          "flags": flags.get(most_prm, [])}
    pb_key = f"{most_pb}_vs_deployed_iu"
    pb_ok = pb_c.get(pb_key, {}).get("ci95", [float("nan")])[0] > ev.PB_NONINF if pb_key in pb_c else None
    freeze["processbench"] = {"config": most_pb, "selected": f"{sel['processbench_lsml']['most_selected_count']}/5",
                              "noninferior_vs_deployed_iu": pb_ok, "flags": flags.get(most_pb, [])}

    L = []
    L.append("# Joint L-SML optimization v2 — results report")
    L.append("")
    L.append("Protocol: `docs/experiments/JOINT_LSML_OPTIMIZATION_PLAN_V2.md` with pre-label amendments R1, R2, R3 and the")
    L.append("2026-09-06 late integrity amendment (`docs/experiments/JOINT_LSML_V2_LATE_INTEGRITY_AMENDMENT_20260906.md`).")
    L.append("Populations: 8 ProcessBench cells (Qwen3-4B / Qwen3-8B x gsm8k / math / olympiadbench / omnimath) and PRMBench")
    L.append("Qwen3-8B; nested 5-outer / 5-inner label-free grouped folds; every learned arm fitted on outer-train only.")
    L.append("")
    L.append("## Late-freeze limitation (retained verbatim)")
    L.append("")
    L.append(review["late_freeze_limitation"])
    L.append(f"Integrity record: `INTEGRITY_RECORD_V2.json` (`{record['provenance_status']}`, created {record['created_utc']}),")
    L.append("Amendment R3 artifacts bound by `AMENDMENT_R3_FREEZE.json`. Labels were decoded only by the patched evaluator")
    L.append("after `verify_prelabel_record` passed; the pre-label structural review reported zero aborts.")
    L.append("")
    L.append("## 1. Headline (tuned-vs-tuned, 16 vs 16) and label-free successors")
    L.append("")
    L.append("| Contrast | PRMBench step AUROC | ProcessBench macro-F1 |")
    L.append("|---|---|---|")
    L.append(f"| tuned Joint/L-SML family | {prm_h['tuned_lsml']['auroc']:.4f} | {pb_h['tuned_lsml']['macro_f1']:.4f} |")
    L.append(f"| tuned IU family | {prm_h['tuned_iu']['auroc']:.4f} | {pb_h['tuned_iu']['macro_f1']:.4f} |")
    L.append(f"| **tuned Joint − tuned IU** (paired grouped bootstrap) | {_fmt_ci(prm_h['tuned_lsml_vs_tuned_iu'])} → **{headline['gates']['development_prm']}** | {_fmt_ci(pb_h['tuned_lsml_vs_tuned_iu'])} → **{headline['gates']['development_pb']}** |")
    L.append(f"| deployed IU-PCR (`{DEPLOYED_IU_ROW}`) | {prm_h[DEPLOYED_IU_ROW]['auroc']:.4f} | {pb_h[DEPLOYED_IU_ROW]['macro_f1']:.4f} |")
    L.append(f"| deployed U-PCR port (`{DEPLOYED_UPCR_PORT_ROW}`) | {prm_h[DEPLOYED_UPCR_PORT_ROW]['auroc']:.4f} | {pb_h[DEPLOYED_UPCR_PORT_ROW]['macro_f1']:.4f} |")
    L.append(f"| S1 `internal_joint` (label-free, the repaired contribution) | {prm_h[SUCCESSOR_S1]['auroc']:.4f}; vs deployed IU {_fmt_ci(prm_h['s1_vs_deployed_iu'])} | {pb_h[SUCCESSOR_S1]['macro_f1']:.4f}; vs deployed IU {_fmt_ci(pb_h['s1_vs_deployed_iu'])}; activation guard **{pb_t[SUCCESSOR_S1]['verdict']}** |")
    L.append(f"| S2 `internal_cont` (label-free) | {prm_h[SUCCESSOR_S2]['auroc']:.4f}; vs deployed IU {_fmt_ci(prm_h['s2_vs_deployed_iu'])} | {pb_h[SUCCESSOR_S2]['macro_f1']:.4f}; vs deployed IU {_fmt_ci(pb_h['s2_vs_deployed_iu'])}; activation guard **{pb_t[SUCCESSOR_S2]['verdict']}** |")
    L.append(f"| fixed-family CONT control `prov5_cont` | {prm_h['prov5_cont']['auroc']:.4f} | {pb_h['prov5_cont']['macro_f1']:.4f} |")
    L.append(f"| continuity row `{CONTINUITY}` (historical estimator) | {prm_h[CONTINUITY]['auroc']:.4f} | {pb_h[CONTINUITY]['macro_f1']:.4f} |")
    L.append("")
    L.append(f"Gates: development PRMB **{headline['gates']['development_prm']}**, development PB **{headline['gates']['development_pb']}**, "
             f"S1 {headline['gates']['s1_promotion']}, S2 {headline['gates']['s2_promotion']}.")
    L.append("")
    L.append("### Selection frequency (inner 5-fold, tuned-vs-tuned)")
    L.append("")
    L.append("| Panel | Family | Selected per outer fold | Most selected | Qualifier |")
    L.append("|---|---|---|---|---|")
    for panel in ("prmbench", "processbench"):
        for fam in ("lsml", "iu"):
            s = sel[f"{panel}_{fam}"]
            L.append(f"| {panel} | {fam} | {s['counts']} | `{s['most_selected']}` ({s['most_selected_count']}/5) | "
                     f"{'UNSTABLE_SELECTION' if s['unstable_selection'] else 'stable (>=3/5)'} |")
    L.append("")
    L.append("### Mechanism attribution (Section 7.3) and the R3 lambda=0 reference")
    L.append("")
    L.append(f"- PRMB winner `{most_prm}` vs tuned IU: {_fmt_ci(prm_h['tuned_lsml_vs_tuned_iu'])}.")
    if "permctl_graph_vs_tuned_iu" in prm_c:
        L.append(f"- graph-permutation control `{PERMCTL_GRAPH}` (AUROC {prm_t[PERMCTL_GRAPH]['auroc']:.4f}) vs tuned IU: {_fmt_ci(prm_c['permctl_graph_vs_tuned_iu'])}.")
    if "lambda0_reference_vs_tuned_iu" in prm_c:
        L.append(f"- ungated lambda=0 model-inverse reference `{HOOK3_LAMBDA0_REFERENCE_ROW}` (AUROC {prm_t[HOOK3_LAMBDA0_REFERENCE_ROW]['auroc']:.4f}) vs tuned IU: {_fmt_ci(prm_c['lambda0_reference_vs_tuned_iu'])}; "
                 f"`internal_joint_liu010` vs that reference: {_fmt_ci(prm_c['liu010_vs_lambda0_reference'])}.")
    h3 = review["inertness"]["hook3_vs_lambda0_reference"].get("internal_joint_liu010", {})
    if h3.get("n_lanes"):
        L.append(f"- weight-map cosine liu010 vs lambda=0 reference: median {h3['median_cosine']:.4f}, min {h3['min_cosine']:.4f}, "
                 f"{h3['lanes_at_or_above_0995']}/{h3['n_lanes_row']} lanes at >= 0.995 (not MECHANISM_INERT under the every-lane rule; near-inert).")
    L.append(f"- **Verdict: {attribution}.**")
    L.append("")
    L.append("### Fresh-data freeze (Section 7.2)")
    L.append("")
    L.append(f"- PRMBench: `{freeze['prmbench']['config']}` ({freeze['prmbench']['selected']}); non-inferior vs deployed IU: {freeze['prmbench']['noninferior_vs_deployed_iu']}; flags {freeze['prmbench']['flags'] or 'none'}.")
    L.append(f"- ProcessBench: `{freeze['processbench']['config']}` ({freeze['processbench']['selected']}); non-inferior vs deployed IU: {freeze['processbench']['noninferior_vs_deployed_iu']}"
             + (f" ({_fmt_ci(pb_c[pb_key])})" if pb_key in pb_c else "") + f"; flags {freeze['processbench']['flags'] or 'none'}.")
    if "tuned_lsml_vs_deployed_iu" in pb_c:
        L.append(f"- PB per-fold tuned Joint vs deployed IU: {_fmt_ci(pb_c['tuned_lsml_vs_deployed_iu'])}.")
    L.append("- S1 / S2 are carried as registered but fail promotion on both panels (HARM, and CATASTROPHE on ProcessBench).")
    L.append("")
    L.append("### Guard cost (Amendment R1 continuity row) and named-control contrasts")
    L.append("")
    for panel, c in (("prmbench", prm_c), ("processbench", pb_c)):
        for name, entry in c.items():
            if name in ("guarded_vs_unguarded_fixed_family", "continuity_vs_deployed_iu", "deployed_upcr_port_vs_deployed_iu"):
                L.append(f"- {panel} {name}: {_fmt_ci(entry)}")
    L.append("")
    L.append("## 2. Per-arm descriptive table (outer-refit, no selection; both panels)")
    L.append("")
    L.append("| Row | PRMB AUROC | PB macro-F1 | PB error-side hit | PB clean-side | PB activation guard | pre-label flags |")
    L.append("|---|---|---|---|---|---|---|")
    order = lsml_ids + [HOOK3_LAMBDA0_REFERENCE_ROW, CONTINUITY, "equal_all23", "equal_family_active23",
                        PERMCTL_GATE, PERMCTL_GRAPH] + iu_ids
    for arm in order:
        if arm not in prm_t and arm not in pb_t:
            L.append(f"| `{arm}` — {ROW_LABEL.get(arm, '')} | excluded (coverage) | excluded (coverage) | | | | {', '.join(flags.get(arm, [])) or ''} |")
            continue
        p = prm_t.get(arm, {}); b = pb_t.get(arm, {})
        L.append(f"| `{arm}` — {ROW_LABEL.get(arm, 'IU grid')} | {p.get('auroc', float('nan')):.4f} | {b.get('macro_f1', float('nan')):.4f} | "
                 f"{b.get('error_side_hit_rate', float('nan')):.3f} | {b.get('clean_side_rate', float('nan')):.3f} | {b.get('verdict', '')} (act {b.get('mean_activation', float('nan')):.2f}) | "
                 f"{', '.join(flags.get(arm, [])) or ''} |")
    L.append("")
    L.append("## 3. Module B — learned trajectory-axis reducer (PRMBench primary)")
    L.append("")
    g = moduleb["grid_primary"]
    L.append(f"Primary contrast (inner-selected best of the 3x3 grid vs frozen top-1/span-max control on the same substrate): selected `{g['selected_by_fold']['0']}` on "
             f"{Counter(g['selected_by_fold'].values()).most_common(1)[0][1]}/5 folds; winner AUROC {g['winner_auroc']:.4f} vs B0 {g['b0_same_substrate_auroc']:.4f}; "
             f"delta {_fmt_ci(g)} → **{g['gate']}**.")
    L.append("")
    L.append("| Row | PRMB step AUROC | vs B0 (descriptive) |")
    L.append("|---|---|---|")
    names = {"b0": "B0 frozen span-max control", "b1": "B1 label-free SML weights over 10 order statistics",
             "b2a": f"B2a max-vs-mean blend (alpha by fold {moduleb['b2a_alpha']})", "b2b": "B2b positional bins (label-free)",
             "b3": "B3 supervised LR over order statistics (competitor)"}
    for key, label in names.items():
        d = moduleb["vs_b0_descriptive"].get(key)
        L.append(f"| {label} | {moduleb['auroc'][key]:.4f} | {_fmt_ci(d) if d else '—'} |")
    L.append("")
    L.append("3x3 grid (substrate x trajectory fuser), descriptive AUROC: " + ", ".join(
        f"`{k}` {v:.4f}" for k, v in moduleb["grid_descriptive_auroc"].items()))
    blocked = sorted({k for k, st in moduleb["grid_status"].items() if all(v != "OK" for v in st.values())})
    L.append(f"Grid cells BLOCKED on every fold (no admissible LOAO partition over 10 order-statistic units): {blocked}.")
    L.append("")
    L.append("Pre-registered mechanism prediction (trajectory-IU beats trajectory-SML because the order statistics are near-collinear): "
             + ("NOT confirmed on the deployed-IU substrate (sml > iu)" if moduleb["grid_descriptive_auroc"].get("iu_c2_s25_l2_exoff__sml", 0) > moduleb["grid_descriptive_auroc"].get("iu_c2_s25_l2_exoff__iu", 0) else "confirmed on the deployed-IU substrate (iu > sml)")
             + ".")
    L.append("")
    L.append("## 4. Pre-label structural review (label-free, registered before evaluation)")
    L.append("")
    fb = review["fallback"]
    L.append(f"- INTERNAL grouping blocked: PB {fb['processbench']['internal_blocked_rate']} (cap {fb['processbench']['cap']}), PRMB {fb['prmbench']['internal_blocked_rate']} (cap {fb['prmbench']['cap']}); K distribution PB {fb['processbench']['internal_K_distribution']}, PRMB {fb['prmbench']['internal_K_distribution']}.")
    L.append(f"- Hook 1 gated-affinity grouping blocked: PB {fb['processbench']['gated_affinity_blocked_rate']} → both Hook 1 rows STRUCTURALLY_FRAGILE on ProcessBench; PRMB {fb['prmbench']['gated_affinity_blocked_rate']}.")
    L.append(f"- Gate seed std max {review['gate_seed_std']['max']:.4f} (cap {review['gate_seed_std']['cap']}); no candidate arm below the 0.5 cross-fold map-cosine floor; every arm passes map agreement vs `prov5_cont` (floor 0.50).")
    L.append("- No row MECHANISM_INERT under the every-lane rule; Hook 3 rows at lambda=0.1 are near-inert relative to the lambda=0 reference (see Section 1).")
    L.append(f"- Amendment R2 coverage: `dufs_pf_lsml` fails closed on {review['admission']['processbench']['incomplete'].get('dufs_pf_lsml')}/240 PB lanes and {review['admission']['prmbench']['incomplete'].get('dufs_pf_lsml')}/30 PRMB lanes → excluded from selection, descriptive only.")
    L.append("")
    L.append("Full review: `prelabel_structure_review.md`. Evaluator outputs: `evaluation/headline.json`, `inner_selection.json`, `moduleb.json`, `report_tables.json`, `report_contrasts.json`.")
    L.append("")
    L.append("## 5. Interpretation (written after evaluation; not part of the registered gates)")
    L.append("")
    L.append("1. **The Step-349 objective/head mismatch was real, and fixing the head is what wins on PRMBench.** Every map that solves "
             "`(C_model + gamma I) w = v` with the fitted joint factors `u_g` (the four Hook 3 rows, their permutation control, and the "
             "ungated lambda=0 reference) beats deployed IU-PCR by 0.6-0.7pp AUROC with CIs clear of zero; the hierarchical head on the "
             "same fit and the same groups scores 0.611. The lambda=0 reference is the best of the family (0.6734).")
    L.append("2. **DUFS integration into the coefficients adds nothing here.** Dose-response is monotone against it on both hooks "
             "(lambda 0 > 0.1 > 0.5), the node-relabeled graph reproduces the win, and Hook 3a at lambda=0.1 is a small but significant "
             "harm relative to its own lambda=0 map (-0.0010 [-0.0015, -0.0005]). Hook 2 congruence and Hook 1 gated grouping are also "
             "at or below their ungated references. The PRMB SUPPORT is therefore MECHANISM_UNATTRIBUTED to DUFS and attributed to the "
             "regularized model-inverse head.")
    L.append("3. **INTERNAL grouping, not the map, is the ProcessBench failure.** Every INTERNAL-grouping row (CONT, hierarchical, model-inverse) "
             "sits at 0.13-0.28 macro-F1 while every provenance-grouping row sits at 0.34-0.35; the model-inverse rows keep normal activation "
             "(0.79) but a weaker detector ranking, so this is not the Step-349 threshold collapse. INTERNAL selected K=3 on 35/35 admissible "
             "PB lanes.")
    L.append("4. **K=3 is forced by the minimum-group-size-3 rule, not by instability** (label-free re-fit of the LOAO consensus on two frozen "
             "folds): K=4/5/6 partitions have median ARI 1.000 to their consensus but always contain a size-2 group and are rejected; K=8 can "
             "never be admissible with 23 features. At m=2 the within-group system is exactly determined (Amendment R1 Section 4) and the "
             "model-inverse head uses `u_g` directly, so pairs are safe; only singletons must stay forbidden.")
    L.append("5. **The Step-205 small-m guard costs the fixed-family row 1.5pp AUROC on PRMBench** (-0.0154 [-0.0165, -0.0144]) and is NULL on "
             "ProcessBench; the historical unguarded estimator is statistically indistinguishable from deployed IU on PRMBench "
             "(-0.0005 [-0.0018, +0.0008]).")
    L.append("6. **Module B:** the label-free order-statistic reducer does not beat the frozen max (HARM, -0.0034); the max-vs-mean blend "
             "(alpha=0.5 on every fold) and the supervised LR both add ~+0.006 AUROC descriptively, i.e. the informative signal is the "
             "tail plus a bulk correction, not a learned reweighting of the bulk. Trajectory-IU did not beat trajectory-SML on the deployed "
             "substrate (prediction not confirmed). The Joint fuser is BLOCKED on all folds over 10 order-statistic units (min group size 3).")
    L.append("")
    L.append("**Registered next step (Step 354, to be written before any run):** (a) `target_condition` dose {30, 100, 300, 1000, 3000, 1e4} "
             "for the ungated model-inverse map on INTERNAL and on provenance groups; (b) grouping with minimum group size 2, K in {3,...,8}, "
             "same stability rule, for both the model-inverse map and the hierarchical head. Same nested folds and controls; labelled "
             "development, because both axes were chosen after seeing these results.")
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("REPORT.md written")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("tables", "contrasts", "render", "all"), default="all")
    args = ap.parse_args()
    if args.stage in ("tables", "all"):
        stage_tables()
    if args.stage in ("contrasts", "all"):
        stage_contrasts()
    if args.stage in ("render", "all"):
        stage_render()


if __name__ == "__main__":
    main()
