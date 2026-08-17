#!/usr/bin/env python
"""
L0 — the shared ProcessBench table, rebuilt from existing artifacts. CPU only, no inference.

Handoff §L0: one 3,400-row-by-method table with stable IDs and the official
earliest-erroneous-step-or-(-1) evaluator. Every method is scored on the *identical ordered
population* with the *same* scorer, so the comparison is between methods rather than between
two labs' evaluation scripts.

What belongs in it
------------------
    ours            the frozen family-six / step-top-five locator
    max_entropy     maximum entropy + the same top-five locator (the strongest transparent
                    token statistic — the control that matters most, since Step 273 found
                    us only at parity with it)
    mind_the_gap    the reproduced Evidence-Drop control, shared protocol
    uprm_judge      the uPRM paper's own Eq. 6 control (L1, once it lands)
    prm_*, critic_* released PRM / critic checkpoints — HIGHER ACCESS TIER, see below
    published_*     numbers quoted from papers, never rerun

Access tiers are printed beside every row and never collapsed. A released PRM sees
step-level supervision and a 72B critic makes eight sampled passes; our method makes one
pass over log-probabilities it already had. Ranking them in one column without the tier is
the single most misleading thing this table could do, so `--strict-tiers` (default) refuses
to emit a bare ranking.

Mind-the-Gap's native SLA goes in a SEPARATE panel because SLA is computed on erroneous
traces only — it is not on the same population as ProcessBench F1.

Regression anchors (expected, not gates): ours 0.3662 macro F1; max entropy 0.3614;
shared-protocol Mind the Gap 0.2646; Qwen2.5-Math PRM ceiling 0.7280; critic ceiling 0.5895.

Usage:
    # first: see what per-row predictions actually exist and in what schema
    python scripts/paper_exact_l0_table.py --inventory --roots $SHARED/results

    # then build
    python scripts/paper_exact_l0_table.py --roots $SHARED/results \
        --out $SHARED/results/paper_exact/l0
"""
import argparse
import glob
import json
import os
import pickle
import sys
from datetime import datetime, timezone

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from spectral_utils.paper_exact import evaluator as EV        # noqa: E402
from spectral_utils.paper_exact.gates import Gate             # noqa: E402
from spectral_utils.processbench import SUBSETS, load_processbench  # noqa: E402

NO_ERROR = EV.NO_ERROR

REGRESSION_ANCHORS = {
    "ours": 0.3662, "max_entropy": 0.3614, "mind_the_gap": 0.2646,
    "prm_qwen25math7b": 0.7280, "critic_qwen72b": 0.5895,
}

#: method -> (access tier, labels/training, model passes, fidelity)
METHOD_TIERS = {
    "ours":              ("one-trace logprob", "none", 1, "our method"),
    "max_entropy":       ("one-trace logprob", "none", 1, "adapted-common-protocol"),
    "mind_the_gap":      ("one-trace logprob", "none", 1, "paper-specified"),
    "uprm_judge":        ("one-trace logprob", "none", 1, "paper-specified-partial"),
    "prm_qwen25math7b":  ("supervised PRM", "step-level PRM800K", 1, "official-exact"),
    "critic_qwen72b":    ("external judge", "none (8-sample vote)", 8, "official-exact"),
}


def canonical_rows() -> list:
    """The official 3,400 rows, in a stable, reproducible order.

    Ordering is (subset, dataset index) rather than whatever each artifact happened to
    iterate — two artifacts built in different orders must land on the same row here or the
    "identical population" claim is empty.
    """
    rows = []
    for subset in SUBSETS:
        for i, r in enumerate(load_processbench(subset, None)):
            rows.append({
                "row_id": f"{subset}:{r.get('id', i)}",
                "subset": subset, "index": i,
                "label": int(r["label"]),
                "generator": r.get("generator"),
                "n_steps": len(r["steps"]),
            })
    return rows


# ── artifact readers ────────────────────────────────────────────────────────────
#
# Each reader returns {row_id: predicted_first_error_step_or_-1}. They are deliberately
# permissive about *where* the prediction lives and strict about the row identity: a
# prediction that cannot be tied to a canonical row_id is dropped and counted, never
# positionally guessed onto a row.

def _iter_pkl_entries(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, dict):
                yield k, v
    elif isinstance(obj, list):
        for k, v in enumerate(obj):
            if isinstance(v, dict):
                yield k, v


def read_prediction_pkls(pattern: str, subset_from_name: bool = True) -> dict:
    """Read {row_id: prediction} from a glob of per-subset prediction pkls."""
    preds, unmatched = {}, 0
    for path in sorted(glob.glob(pattern)):
        base = os.path.basename(path)
        subset = next((s for s in SUBSETS if s in base), None) if subset_from_name else None
        for key, entry in _iter_pkl_entries(path):
            pred = entry.get("prediction", entry.get("pred", entry.get("pred_step")))
            rid = entry.get("id", entry.get("row_id"))
            sub = entry.get("subset", subset)
            if sub is None:
                unmatched += 1
                continue
            row_id = f"{sub}:{rid}" if rid is not None else f"{sub}:{key}"
            preds[row_id] = (int(pred) if pred is not None else None)
    return {"predictions": preds, "n_unmatched": unmatched}


def inventory(roots) -> dict:
    """Report what per-row prediction artifacts exist and what schema they carry.

    Run this before building. Guessing an artifact's schema is how a table silently ends up
    scoring the wrong column, and this project has already spent a step (193) chasing
    exactly that class of staleness.
    """
    found = {}
    for root in roots:
        for path in sorted(glob.glob(os.path.join(root, "**", "*.pkl"), recursive=True)):
            base = os.path.basename(path)
            if not any(t in base.lower() for t in ("pb_", "processbench", "uprm", "prm", "critic")):
                continue
            try:
                n, keys, sample = 0, set(), None
                for k, v in _iter_pkl_entries(path):
                    n += 1
                    keys |= set(v)
                    if sample is None:
                        sample = {kk: (str(vv)[:60]) for kk, vv in list(v.items())[:8]}
                    if n >= 200:
                        break
                found[path] = {"n_entries_sampled": n, "keys": sorted(keys),
                               "has_prediction": bool(keys & {"prediction", "pred", "pred_step"}),
                               "has_label": "label" in keys,
                               "size_mb": round(os.path.getsize(path) / 1e6, 1),
                               "sample": sample}
            except Exception as e:  # noqa: BLE001
                found[path] = {"error": repr(e)[:200]}
    return found


def build_table(rows, methods: dict, gate: Gate) -> dict:
    """methods: name -> {row_id: prediction}. Returns the long table plus per-subset stats."""
    by_id = {r["row_id"]: r for r in rows}
    long, per_method = [], {}

    for name, preds in methods.items():
        covered = [rid for rid in by_id if rid in preds]
        coverage = len(covered) / max(1, len(by_id))
        gate.check(f"coverage_{name}", coverage >= 0.99,
                   f"{len(covered)}/{len(by_id)} canonical rows ({coverage:.3f})")
        per_subset = {}
        for subset in SUBSETS:
            ids = [r["row_id"] for r in rows if r["subset"] == subset]
            p = [preds.get(rid) for rid in ids]
            l = [by_id[rid]["label"] for rid in ids]
            per_subset[subset] = EV.processbench_f1(p, l)
            per_subset[subset]["sla"] = EV.mind_the_gap_sla(p, l)["sla"]
            per_subset[subset]["sla_tol1"] = EV.mind_the_gap_sla(p, l, tolerance=1)["sla"]
        tier = METHOD_TIERS.get(name, ("unknown", "unknown", None, "unlabelled"))
        per_method[name] = {
            "per_subset": per_subset,
            "macro_f1": EV.macro_f1(per_subset),
            "coverage": coverage,
            "access_tier": tier[0], "labels_or_training": tier[1],
            "model_passes": tier[2], "fidelity": tier[3],
            "regression_anchor": REGRESSION_ANCHORS.get(name),
        }
        for rid in by_id:
            long.append({"row_id": rid, "subset": by_id[rid]["subset"],
                         "label": by_id[rid]["label"], "method": name,
                         "prediction": preds.get(rid)})
    return {"long": long, "per_method": per_method}


def paired_bootstrap_vs(per_method, methods: dict, rows, reference: str, gate: Gate) -> dict:
    """Paired question-level bootstrap of each method's macro F1 against `reference`."""
    if reference not in methods:
        return {}
    by_id = {r["row_id"]: r for r in rows}

    def macro_from(payloads):
        per = {}
        for subset in SUBSETS:
            sel = [p for p in payloads if p["subset"] == subset]
            if sel:
                per[subset] = EV.processbench_f1([p["pred"] for p in sel],
                                                 [p["label"] for p in sel])
        return EV.macro_f1(per) if per else float("nan")

    out = {}
    ref_groups = {rid: {"subset": by_id[rid]["subset"], "label": by_id[rid]["label"],
                        "pred": methods[reference].get(rid)} for rid in by_id}
    for name, preds in methods.items():
        if name == reference:
            continue
        groups = {rid: {"subset": by_id[rid]["subset"], "label": by_id[rid]["label"],
                        "pred": preds.get(rid)} for rid in by_id}
        out[name] = EV.paired_grouped_bootstrap(groups, ref_groups, macro_from, n_boot=1000)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--roots", nargs="+",
                    default=["/shared/cycle2_tau_averbuch_prj/omrisegev1/results"])
    ap.add_argument("--inventory", action="store_true")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "paper_exact", "l0"))
    ap.add_argument("--reference", default="max_entropy",
                    help="method the paired bootstrap compares against")
    ap.add_argument("--strict-tiers", action="store_true", default=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.inventory:
        inv = inventory(args.roots)
        path = os.path.join(args.out, "L0_INVENTORY.json")
        with open(path, "w") as f:
            json.dump(inv, f, indent=2, default=str)
        for p, d in inv.items():
            flag = "PRED" if d.get("has_prediction") else "    "
            print(f"{flag} {d.get('size_mb', '?'):>7} MB  {p}")
            if d.get("keys"):
                print(f"        keys: {', '.join(d['keys'][:14])}")
        print(f"\ninventory -> {path}")
        print("\nWire each PRED artifact into MANIFEST_SOURCES below, then rerun without "
              "--inventory. Do not guess a schema.")
        return

    gate = Gate("L0-shared-processbench-table", args.out)
    rows = canonical_rows()
    gate.check("canonical_population", len(rows) == 3400,
               f"{len(rows)} rows (official ProcessBench is 3,400)")
    counts = {s: sum(1 for r in rows if r["subset"] == s) for s in SUBSETS}
    print(f"[l0] canonical rows: {counts}", flush=True)

    #: Wire in whatever the inventory found. Each entry is name -> reader callable.
    MANIFEST_SOURCES = {
        "uprm_judge": lambda: read_prediction_pkls(
            os.path.join(args.roots[0], "paper_exact", "l1_uprm_judge_full",
                         "shards", "*.pkl")),
    }
    methods, sources = {}, {}
    for name, reader in MANIFEST_SOURCES.items():
        try:
            res = reader()
            if res["predictions"]:
                methods[name] = res["predictions"]
                sources[name] = {"n": len(res["predictions"]),
                                 "n_unmatched": res["n_unmatched"]}
                print(f"[l0] {name}: {len(res['predictions'])} predictions", flush=True)
            else:
                print(f"[l0] {name}: no predictions found (stage not run yet)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[l0] {name}: reader failed: {e!r}", flush=True)

    gate.check("at_least_one_method", bool(methods),
               f"{len(methods)} methods wired: {sorted(methods)}"
               if methods else "no per-row predictions available yet — run --inventory and "
                               "wire MANIFEST_SOURCES, or wait for L1/L2 to land")
    if not methods:
        gate.finish(raise_on_fail=False)
        return

    table = build_table(rows, methods, gate)
    paired = paired_bootstrap_vs(table["per_method"], methods, rows, args.reference, gate)

    for name, st in table["per_method"].items():
        anchor = st["regression_anchor"]
        print(f"[l0] {name:<20} macro F1 = {st['macro_f1']:.4f}"
              + (f"   [anchor {anchor:.4f}]" if anchor else "")
              + f"   tier={st['access_tier']}", flush=True)
        if anchor is not None:
            gate.check(f"anchor_{name}", abs(st["macro_f1"] - anchor) < 0.02,
                       f"macro F1 {st['macro_f1']:.4f} vs recorded anchor {anchor:.4f}")

    report = {
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evaluator_revision": EV.EVALUATOR_REVISION,
        "n_rows": len(rows), "subset_counts": counts,
        "sources": sources,
        "per_method": table["per_method"],
        "paired_vs_reference": {"reference": args.reference, "deltas": paired},
        "regression_anchors": REGRESSION_ANCHORS,
        "note": "Access tiers must accompany every number. SLA is reported per method but "
                "belongs in a separate panel: it is computed on erroneous traces only and "
                "is not on the same population as ProcessBench F1.",
    }
    with open(os.path.join(args.out, "L0_TABLE.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    import csv
    with open(os.path.join(args.out, "L0_LONG.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["row_id", "subset", "label", "method", "prediction"])
        w.writeheader()
        w.writerows(table["long"])
    print(f"\nL0 -> {args.out}")
    gate.finish(raise_on_fail=False)


if __name__ == "__main__":
    main()
