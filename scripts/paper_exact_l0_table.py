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
from spectral_utils.paper_exact.manifest import sha256_file   # noqa: E402
from spectral_utils.processbench import SUBSETS, load_processbench  # noqa: E402

NO_ERROR = EV.NO_ERROR

REGRESSION_ANCHORS = {
    "ours": 0.3662, "max_entropy": 0.3614, "mind_the_gap": 0.2646,
    "prm_qwen25math7b": 0.7280, "critic_qwen72b": 0.5895,
}

#: Provenance is a SEPARATE axis from fidelity, and Codex §8 Q2 requires it on every row.
#:
#: `fidelity` answers "how faithfully does this reproduce the paper's protocol?".
#: `provenance` answers "was this produced under the paper_exact_acquisition_v1 contract,
#: with an immutable hashed manifest?". A row can be faithful and yet pre-contract: the
#: released-PRM and critic ceilings were run before the contract existed, so no immutable
#: RUN_MANIFEST governs them.
#:
#: Q2's explicit instruction: mark those `pre-contract provenance` and DO NOT fabricate a
#: retroactive manifest. Backfilling a manifest onto an artifact whose exact tree, revision
#: and dataset order were never recorded would assert a chain of custody that does not
#: exist — the label is the honest artifact, not a reconstructed hash.
PROVENANCE_CONTRACT = "paper-exact-contract"      # governed by an immutable RUN_MANIFEST
PROVENANCE_PRE = "pre-contract provenance"        # predates the contract; hashed, not manifested
PROVENANCE_QUOTED = "published-value-only"        # never rerun here; quoted from the paper

#: method -> {access tier, labels/training, model passes, fidelity, provenance}
METHOD_TIERS = {
    "ours": dict(access_tier="one-trace logprob", labels_or_training="none", model_passes=1,
                 fidelity="our method", provenance=PROVENANCE_PRE),
    "max_entropy": dict(access_tier="one-trace logprob", labels_or_training="none",
                        model_passes=1, fidelity="adapted-common-protocol",
                        provenance=PROVENANCE_PRE),
    "mind_the_gap": dict(access_tier="one-trace logprob", labels_or_training="none",
                         model_passes=1, fidelity="paper-specified",
                         provenance=PROVENANCE_PRE),
    "uprm_judge": dict(access_tier="one-trace logprob", labels_or_training="none",
                       model_passes=1, fidelity="paper-specified-partial",
                       provenance=PROVENANCE_CONTRACT),
    "prm_qwen25math7b": dict(access_tier="supervised PRM",
                             labels_or_training="step-level PRM800K", model_passes=1,
                             fidelity="official-exact", provenance=PROVENANCE_PRE),
    "critic_qwen72b": dict(access_tier="external judge",
                           labels_or_training="none (8-sample vote)", model_passes=8,
                           fidelity="official-exact", provenance=PROVENANCE_PRE),
}

UNKNOWN_TIER = dict(access_tier="unknown", labels_or_training="unknown", model_passes=None,
                    fidelity="unlabelled", provenance="unknown")


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


def _find_run_manifest(path: str, stop_at: str = "/") -> str:
    """Walk up from `path` looking for a paper_exact RUN_MANIFEST.json.

    Provenance is DETECTED, not declared. An artifact is contract-governed only if an
    immutable manifest actually sits above it on disk; asserting the label in a table by
    hand is exactly the sort of unverified claim this contract exists to prevent.
    """
    d = os.path.dirname(os.path.abspath(path))
    while True:
        cand = os.path.join(d, "RUN_MANIFEST.json")
        if os.path.exists(cand):
            return cand
        parent = os.path.dirname(d)
        if parent == d or d == stop_at:
            return ""
        d = parent


def describe_source(path: str) -> dict:
    """Hash and stamp one source file. Q2: 'inventory and hash every source'."""
    man = _find_run_manifest(path)
    entry = {
        "path": path,
        "sha256": sha256_file(path),
        "size_bytes": os.path.getsize(path),
        "mtime_utc": datetime.fromtimestamp(os.path.getmtime(path),
                                            timezone.utc).isoformat(timespec="seconds"),
        "run_manifest": man or None,
        "provenance": PROVENANCE_CONTRACT if man else PROVENANCE_PRE,
    }
    if man:
        try:
            with open(man) as f:
                m = json.load(f)
            entry["manifest_schema"] = m.get("schema")
            entry["manifest_run_id"] = m.get("run_id")
            entry["manifest_model_revision"] = m.get("model_revision")
            entry["manifest_fidelity"] = m.get("fidelity")
        except Exception as e:  # noqa: BLE001
            entry["manifest_error"] = repr(e)[:200]
            entry["provenance"] = PROVENANCE_PRE   # unreadable manifest governs nothing
    return entry


def read_prediction_pkls(pattern: str, subset_from_name: bool = True) -> dict:
    """Read {row_id: prediction} from a glob of per-subset prediction pkls."""
    preds, unmatched, srcs = {}, 0, []
    for path in sorted(glob.glob(pattern)):
        srcs.append(describe_source(path))
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
    return {"predictions": preds, "n_unmatched": unmatched, "sources": srcs}


def read_prediction_shards(run_dir: str) -> dict:
    """Read {row_id: prediction} from a paper_exact sharded acquisition directory.

    Uses the contract's own reader so the INDEX/STATUS hash chain is verified rather than
    bypassed by globbing raw shard files.
    """
    from spectral_utils.paper_exact.shards import iter_run_dirs, read_shards, verify_shards

    preds, unmatched, srcs = {}, 0, []
    dirs = iter_run_dirs(run_dir)
    for d in dirs:
        ver = verify_shards(d)
        man = _find_run_manifest(os.path.join(d, "INDEX.jsonl"))
        srcs.append({"path": d, "shard_verification": ver, "run_manifest": man or None,
                     "provenance": PROVENANCE_CONTRACT if man else PROVENANCE_PRE})
        for rec in read_shards(d):
            pred = rec.get("prediction", rec.get("pred", rec.get("pred_step")))
            sub, rid = rec.get("subset"), rec.get("id", rec.get("row_id"))
            if sub is None or rid is None:
                unmatched += 1
                continue
            preds[f"{sub}:{rid}"] = (int(pred) if pred is not None else None)
    return {"predictions": preds, "n_unmatched": unmatched, "sources": srcs}


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
                # Hashed at inventory time too, so the schema report and the table are
                # provably talking about the same bytes.
                found[path] = {**describe_source(path),
                               "n_entries_sampled": n, "keys": sorted(keys),
                               "has_prediction": bool(keys & {"prediction", "pred", "pred_step"}),
                               "has_label": "label" in keys,
                               "size_mb": round(os.path.getsize(path) / 1e6, 1),
                               "sample": sample}
            except Exception as e:  # noqa: BLE001
                found[path] = {"error": repr(e)[:200]}
    return found


#: Candidate locations per method, most-specific first. Paths are DISCOVERED (first glob that
#: matches a real file wins) rather than hardcoded, because the pre-contract artifacts predate
#: any naming convention. Discovery applies to *where* a file is; the schema still comes from
#: the permissive readers above, which count every prediction they cannot tie to a row_id.
SOURCE_CANDIDATES = {
    "uprm_judge": [("shards", "paper_exact/l1_uprm_judge_full")],
    "ours":         [("pkl", "**/pb_*ours*.pkl"), ("pkl", "**/processbench_ours*.pkl")],
    "max_entropy":  [("pkl", "**/pb_*max_entropy*.pkl"), ("pkl", "**/pb_*maxent*.pkl")],
    "mind_the_gap": [("pkl", "**/pb_*mind*gap*.pkl"), ("pkl", "**/pb_*evidence*drop*.pkl")],
    "prm_qwen25math7b": [("pkl", "**/pb_*prm*.pkl")],
    "critic_qwen72b":   [("pkl", "**/pb_*critic*.pkl")],
}


def manifest_sources(args) -> dict:
    """name -> zero-arg reader. Resolves SOURCE_CANDIDATES against --roots, plus --source."""
    overrides = {}
    for spec in (args.source or []):
        if ":" not in spec:
            raise SystemExit(f"--source must be name:kind:path, got {spec!r}")
        name, kind, path = spec.split(":", 2)
        overrides.setdefault(name, []).append((kind, path))

    out = {}
    for name, cands in {**SOURCE_CANDIDATES, **overrides}.items():
        chosen = None
        for kind, rel in cands:
            for root in args.roots:
                path = rel if os.path.isabs(rel) else os.path.join(root, rel)
                if kind == "shards" and os.path.isdir(path):
                    chosen = ("shards", path)
                elif kind == "pkl" and glob.glob(path, recursive=True):
                    chosen = ("pkl", path)
                if chosen:
                    break
            if chosen:
                break
        if not chosen:
            continue
        kind, path = chosen
        out[name] = ((lambda p: (lambda: read_prediction_shards(p)))(path) if kind == "shards"
                     else (lambda p: (lambda: read_prediction_pkls(p)))(path))
    return out


def build_table(rows, methods: dict, gate: Gate, observed_provenance: dict = None) -> dict:
    """methods: name -> {row_id: prediction}. Returns the long table plus per-subset stats.

    The SLA numbers are computed here but returned in a SEPARATE panel, never merged into the
    F1 block: ProcessBench F1 is computed on all 3,400 rows, while Mind-the-Gap's SLA is
    computed on the erroneous traces only. Two different populations in adjacent columns of
    one table invite exactly the comparison that is not valid.
    """
    by_id = {r["row_id"]: r for r in rows}
    observed_provenance = observed_provenance or {}
    long, per_method, sla_panel = [], {}, {}

    for name, preds in methods.items():
        covered = [rid for rid in by_id if rid in preds]
        coverage = len(covered) / max(1, len(by_id))
        gate.check(f"coverage_{name}", coverage >= 0.99,
                   f"{len(covered)}/{len(by_id)} canonical rows ({coverage:.3f})")
        per_subset, sla_subset = {}, {}
        for subset in SUBSETS:
            ids = [r["row_id"] for r in rows if r["subset"] == subset]
            p = [preds.get(rid) for rid in ids]
            l = [by_id[rid]["label"] for rid in ids]
            per_subset[subset] = EV.processbench_f1(p, l)
            sla_subset[subset] = {
                "sla": EV.mind_the_gap_sla(p, l)["sla"],
                "sla_tol1": EV.mind_the_gap_sla(p, l, tolerance=1)["sla"],
                "n_erroneous": int(sum(1 for x in l if int(x) != NO_ERROR)),
            }
        tier = dict(METHOD_TIERS.get(name, UNKNOWN_TIER))

        # A declared provenance that disagrees with what is on disk is a gate failure, not a
        # value to quietly prefer. The observed label wins; the disagreement is reported.
        declared = tier["provenance"]
        obs = observed_provenance.get(name)
        if obs and obs != declared:
            gate.check(f"provenance_declared_matches_disk_{name}", False,
                       f"declared {declared!r} but sources on disk say {obs!r}")
            tier["provenance"] = obs
            tier["provenance_declared"] = declared
        elif obs:
            gate.check(f"provenance_declared_matches_disk_{name}", True, f"{obs}")

        per_method[name] = {
            "per_subset": per_subset,
            "macro_f1": EV.macro_f1(per_subset),
            "coverage": coverage,
            "regression_anchor": REGRESSION_ANCHORS.get(name),
            **tier,
        }
        sla_panel[name] = {"per_subset": sla_subset,
                           "access_tier": tier["access_tier"],
                           "provenance": tier["provenance"]}
        for rid in by_id:
            long.append({"row_id": rid, "subset": by_id[rid]["subset"],
                         "label": by_id[rid]["label"], "method": name,
                         "prediction": preds.get(rid)})
    return {"long": long, "per_method": per_method, "sla_panel": sla_panel}


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
    ap.add_argument("--source", action="append", metavar="NAME:KIND:PATH",
                    help="wire a source explicitly, e.g. "
                         "ours:pkl:/path/pb_*_ours.pkl or uprm_judge:shards:/path/run_dir. "
                         "Repeatable; overrides SOURCE_CANDIDATES for that method.")
    ap.add_argument("--strict-tiers", action="store_true", default=True,
                    help="rank only within an access tier (default, and the only safe mode)")
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
            if d.get("provenance"):
                print(f"        prov: {d['provenance']}  sha256: {d.get('sha256','?')[:16]}")
            if d.get("keys"):
                print(f"        keys: {', '.join(d['keys'][:14])}")
        print(f"\ninventory -> {path}")
        print("\nWire each PRED artifact with --source NAME:pkl:<path> (or add it to "
              "SOURCE_CANDIDATES), then rerun without --inventory. Do not guess a schema.")
        return

    gate = Gate("L0-shared-processbench-table", args.out)
    rows = canonical_rows()
    gate.check("canonical_population", len(rows) == 3400,
               f"{len(rows)} rows (official ProcessBench is 3,400)")
    counts = {s: sum(1 for r in rows if r["subset"] == s) for s in SUBSETS}
    print(f"[l0] canonical rows: {counts}", flush=True)

    methods, sources, observed_provenance = {}, {}, {}
    for name, reader in manifest_sources(args).items():
        try:
            res = reader()
            srcs = res.get("sources") or []
            if res["predictions"]:
                methods[name] = res["predictions"]
                # A method is contract-governed only if EVERY source behind it is. One
                # pre-contract file in the set makes the whole row pre-contract.
                labels = {s.get("provenance") for s in srcs} or {PROVENANCE_PRE}
                observed_provenance[name] = (PROVENANCE_CONTRACT
                                             if labels == {PROVENANCE_CONTRACT}
                                             else PROVENANCE_PRE)
                sources[name] = {"n": len(res["predictions"]),
                                 "n_unmatched": res["n_unmatched"],
                                 "observed_provenance": observed_provenance[name],
                                 "files": srcs}
                print(f"[l0] {name}: {len(res['predictions'])} predictions "
                      f"[{observed_provenance[name]}]", flush=True)
            else:
                print(f"[l0] {name}: no predictions found (stage not run yet)", flush=True)
                sources[name] = {"n": 0, "status": "absent", "files": srcs}
        except Exception as e:  # noqa: BLE001
            print(f"[l0] {name}: reader failed: {e!r}", flush=True)
            sources[name] = {"n": 0, "status": "reader_failed", "error": repr(e)[:300]}

    gate.check("at_least_one_method", bool(methods),
               f"{len(methods)} methods wired: {sorted(methods)}"
               if methods else "no per-row predictions available yet — run --inventory and "
                               "wire manifest_sources(), or wait for L1/L2 to land")
    if not methods:
        gate.finish(raise_on_fail=False)
        return

    table = build_table(rows, methods, gate, observed_provenance)
    paired = paired_bootstrap_vs(table["per_method"], methods, rows, args.reference, gate)

    for name, st in table["per_method"].items():
        anchor = st["regression_anchor"]
        print(f"[l0] {name:<20} macro F1 = {st['macro_f1']:.4f}"
              + (f"   [anchor {anchor:.4f}]" if anchor else "")
              + f"   tier={st['access_tier']}  prov={st['provenance']}", flush=True)
        if anchor is not None:
            gate.check(f"anchor_{name}", abs(st["macro_f1"] - anchor) < 0.02,
                       f"macro F1 {st['macro_f1']:.4f} vs recorded anchor {anchor:.4f}")

    # --strict-tiers: rank only WITHIN an access tier. A single ordered column mixing a
    # step-supervised PRM, an 8-pass 72B critic and a one-pass logprob statistic is the one
    # output of this script that would actively mislead, so it is not produced at all.
    tiers = {}
    for name, st in table["per_method"].items():
        tiers.setdefault(st["access_tier"], []).append((st["macro_f1"], name))
    ranking_within_tier = {t: [n for _, n in sorted(v, reverse=True)] for t, v in tiers.items()}
    gate.check("no_cross_tier_ranking", bool(args.strict_tiers),
               f"ranked within {len(tiers)} access tier(s) only: "
               + "; ".join(f"{t}: {' > '.join(v)}" for t, v in ranking_within_tier.items()))

    report = {
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evaluator_revision": EV.EVALUATOR_REVISION,
        "n_rows": len(rows), "subset_counts": counts,
        "sources": sources,
        "per_method": table["per_method"],
        "ranking_within_tier": ranking_within_tier,
        "paired_vs_reference": {"reference": args.reference, "deltas": paired},
        "regression_anchors": REGRESSION_ANCHORS,
        "provenance_note":
            "Rows labelled 'pre-contract provenance' were produced before "
            "paper_exact_acquisition_v1 existed. Their source files are hashed here, but no "
            "immutable RUN_MANIFEST governs them, and none was fabricated retroactively: the "
            "exact tree, model revision and dataset order were not recorded at the time, so a "
            "backfilled manifest would assert a chain of custody that does not exist.",
        "note": "Access tiers must accompany every number. Mind-the-Gap SLA is in "
                "L0_SLA_PANEL.json, not this table: it is computed on erroneous traces only "
                "and is not on the same population as ProcessBench F1.",
    }
    with open(os.path.join(args.out, "L0_TABLE.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    with open(os.path.join(args.out, "L0_SLA_PANEL.json"), "w") as f:
        json.dump({"written_utc": report["written_utc"],
                   "population": "erroneous traces only — NOT comparable to ProcessBench F1",
                   "panel": table["sla_panel"]}, f, indent=2, default=float)
    import csv
    with open(os.path.join(args.out, "L0_LONG.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["row_id", "subset", "label", "method", "prediction"])
        w.writeheader()
        w.writerows(table["long"])
    print(f"\nL0 -> {args.out}")
    gate.finish(raise_on_fail=False)


if __name__ == "__main__":
    main()
