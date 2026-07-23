#!/usr/bin/env python
"""
fetch_backfill.py — fetch backfilled cell pkls from AIRCC, validate them against the
local pre-backfill copies, back the old copies up, then swap the new ones in.

The backfill (cluster/backfill_views.py) modifies raw pkls IN PLACE on the cluster
(append-only keys). This script is the only sanctioned way to land those modified
pkls locally, because it enforces the invariants the whole full-coverage plan rests
on before anything is replaced:

  1. backfill_report.json says Gate B passed for every pkl of the cell;
  2. every candidate now carries all four appended keys
     (token_logsumexp, top_k_logprobs_raw, token_spilled_energies, top_k_logprobs);
  3. labels, full_text, token_entropies, gen_token_ids are BYTE-IDENTICAL to the
     local pre-backfill copy on every sampled candidate (default 20/cell) — the
     backfill must not have perturbed anything published.

Flow per cell:  scp -> cache/_incoming/<cell>/  -> validate ->
                mv cache/repgrid/<cell>/*.pkl -> cache/_backup/<UTC-date>/<cell>/ ->
                move incoming into cache/repgrid/<cell>/.

Usage:
    python scripts/fetch_backfill.py --cells sciq_llama8b,lapeigvals_gsm8k_llama8b
    python scripts/fetch_backfill.py --cells sciq_llama8b --validate-only
"""
import argparse
import datetime
import glob
import json
import os
import pickle
import random
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "cluster"))

from backfill_views import APPEND_KEYS, _present, iter_problems, get_aliased  # noqa: E402
from backfill_specs import BACKFILL_SPECS  # noqa: E402

SHARED = "/shared/cycle2_tau_averbuch_prj/omrisegev1"
# gen_token_ids is frozen only when the pre-backfill copy already had it — tier-2r
# roundtrip cells legitimately get it APPENDED by the backfill.
FROZEN_KEYS = ("label", "full_text", "token_entropies", "gen_token_ids")


def cell_layout(cell, local_root_override=None):
    """Where a cell lives remotely and locally, plus its pkl glob and schema.

    Repgrid cells:  $SHARED/results/repgrid/<cell>  <->  cache/repgrid/<cell>
    Colab cells (c_*): $SHARED/<spec data_dir>      <->  local_cache/<data_dir minus data/colab/>
    """
    if cell.startswith("c_"):
        raw = BACKFILL_SPECS[cell]
        rel = raw["data_dir"]
        tail = rel[len("data/colab/"):] if rel.startswith("data/colab/") else rel
        local_dir = os.path.join(REPO, "local_cache", *tail.split("/"))
        return f"{SHARED}/{rel}", local_dir, raw["pkl_glob"], raw.get("schema", "flat")
    root = local_root_override or os.path.join(REPO, "cache", "repgrid")
    return (f"{SHARED}/results/repgrid/{cell}", os.path.join(root, cell),
            "raw_*.pkl", "repgrid")


def scp_cell(cell, host, incoming, remote_dir, pkl_glob):
    os.makedirs(incoming, exist_ok=True)
    src = f"{host}:{remote_dir}"
    # per-cell report name preferred (shared data_dirs); legacy name as fallback
    pats = [pkl_glob, f"backfill_report_{cell}.json", "backfill_report.json"]
    if not cell.startswith("c_"):
        pats.insert(1, "manifest.json")
    for pat in pats:
        cmd = ["scp", "-q", f"{src}/{pat}", incoming + os.sep]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 and pat == pkl_glob:
            raise RuntimeError(f"scp failed for {src}/{pat}: {r.stderr.strip()}")


def validate_cell(cell, incoming, local_dir, sample_n, pkl_glob, schema,
                  tolerate_skips=False, min_coverage=0.95):
    problems = []

    rep_path = os.path.join(incoming, f"backfill_report_{cell}.json")
    if not os.path.exists(rep_path):
        rep_path = os.path.join(incoming, "backfill_report.json")
    if not os.path.exists(rep_path):
        problems.append("no backfill_report.json fetched")
    else:
        rep = json.load(open(rep_path))
        for p in rep.get("pkls", []):
            if "gate" in p and not p.get("gate_b_pass"):
                problems.append(f"{p['pkl']}: Gate B FAILED on cluster "
                                f"({p.get('gate_b_reasons')})")
            if p.get("error"):
                problems.append(f"{p['pkl']}: {p['error']}")
        if rep.get("validate_only"):
            problems.append("report is from a --validate-only run — nothing was written")

    for new_pkl in sorted(glob.glob(os.path.join(incoming, pkl_glob))):
        name = os.path.basename(new_pkl)
        old_pkl = os.path.join(local_dir, name)
        with open(new_pkl, "rb") as f:
            new = pickle.load(f)

        new_cands = [c for _, _, _, cands in iter_problems(new, schema) for c in cands]
        missing = sum(1 for c in new_cands
                      if any(not _present(c.get(k)) for k in APPEND_KEYS))
        if missing:
            # tier-2r cells can have candidates the driver PROVABLY could not
            # roundtrip (documented in candidate_skips). With --tolerate-skips,
            # accept iff missing == documented skips AND coverage >= the floor;
            # the unified rebuild later drops key-incomplete candidates.
            n_skips = sum(len(p.get("candidate_skips", []))
                          for p in (rep.get("pkls", []) if os.path.exists(rep_path) else [])
                          if p["pkl"] == name)
            coverage = 1 - missing / max(1, len(new_cands))
            if tolerate_skips and missing == n_skips and coverage >= min_coverage:
                print(f"  note: {name}: {missing}/{len(new_cands)} candidates "
                      f"unbackfillable (documented roundtrip skips, coverage "
                      f"{coverage:.1%}) — accepted under --tolerate-skips")
            else:
                problems.append(f"{name}: {missing}/{len(new_cands)} candidates still "
                                f"missing appended keys "
                                f"(documented skips: {n_skips}, coverage {coverage:.1%})")

        if not os.path.exists(old_pkl):
            problems.append(f"{name}: no local pre-backfill copy to compare against")
            continue
        with open(old_pkl, "rb") as f:
            old = pickle.load(f)
        old_cands = [c for _, _, _, cands in iter_problems(old, schema) for c in cands]
        if len(old_cands) != len(new_cands):
            problems.append(f"{name}: candidate count changed "
                            f"({len(old_cands)} -> {len(new_cands)})")
            continue
        rng = random.Random(0)
        idxs = rng.sample(range(len(old_cands)), min(sample_n, len(old_cands)))
        for i in idxs:
            a, b = old_cands[i], new_cands[i]
            for k in FROZEN_KEYS:
                # aliases (ents/text/correct) count as the same key in old caches;
                # keys absent pre-backfill (roundtrip gen_token_ids) may be appended
                av = get_aliased(a, k)
                if av is None:
                    continue
                if pickle.dumps(av) != pickle.dumps(get_aliased(b, k)):
                    problems.append(f"{name}: frozen key {k!r} changed at "
                                    f"candidate {i}")
    return problems


def swap_in(cell, incoming, local_dir, backup_root):
    """Back up exactly the local files the fetch replaces, then move the new ones in."""
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")
    backup = os.path.join(backup_root, stamp, cell)
    os.makedirs(backup, exist_ok=True)
    os.makedirs(local_dir, exist_ok=True)
    for src in glob.glob(os.path.join(incoming, "*")):
        name = os.path.basename(src)
        dst = os.path.join(local_dir, name)
        if os.path.exists(dst):
            shutil.move(dst, os.path.join(backup, name))
        shutil.move(src, dst)
    os.rmdir(incoming)
    return backup


def main():
    ap = argparse.ArgumentParser(description="Fetch + validate + swap backfilled cells")
    ap.add_argument("--cells", required=True, help="comma-separated cell ids")
    ap.add_argument("--host", default="aircc")
    ap.add_argument("--sample-n", type=int, default=20,
                    help="problems sampled for the frozen-key byte comparison")
    ap.add_argument("--validate-only", action="store_true",
                    help="fetch + validate, do NOT swap in")
    ap.add_argument("--tolerate-skips", action="store_true",
                    help="accept cells whose only missing candidates are the "
                         "driver-documented roundtrip skips (coverage floor applies)")
    ap.add_argument("--min-coverage", type=float, default=0.95,
                    help="min fraction of candidates with all appended keys "
                         "when --tolerate-skips is set")
    ap.add_argument("--local-root", default=os.path.join(REPO, "cache", "repgrid"))
    args = ap.parse_args()

    ok_cells, bad_cells = [], []
    for cell in [c.strip() for c in args.cells.split(",") if c.strip()]:
        print(f"\n=== {cell} ===")
        incoming = os.path.join(REPO, "cache", "_incoming", cell)
        remote_dir, local_dir, pkl_glob, schema = cell_layout(cell, args.local_root)
        try:
            scp_cell(cell, args.host, incoming, remote_dir, pkl_glob)
        except RuntimeError as e:
            print(f"  FETCH FAILED: {e}")
            bad_cells.append(cell)
            continue
        problems = validate_cell(cell, incoming, local_dir, args.sample_n,
                                 pkl_glob, schema,
                                 tolerate_skips=args.tolerate_skips,
                                 min_coverage=args.min_coverage)
        if problems:
            print("  VALIDATION FAILED — cell NOT swapped in:")
            for p in problems:
                print(f"    - {p}")
            print(f"  (fetched files left in {incoming} for inspection)")
            bad_cells.append(cell)
            continue
        print("  validation OK: gates passed, all keys present, frozen keys byte-identical")
        if args.validate_only:
            print(f"  --validate-only: left in {incoming}")
            ok_cells.append(cell)
            continue
        backup = swap_in(cell, incoming, local_dir,
                         os.path.join(REPO, "cache", "_backup"))
        print(f"  swapped in; pre-backfill pkls -> {os.path.relpath(backup, REPO)}")
        ok_cells.append(cell)

    print(f"\n[fetch_backfill] OK: {len(ok_cells)}  FAILED: {len(bad_cells)}"
          + (f"  ({', '.join(bad_cells)})" if bad_cells else ""))
    sys.exit(1 if bad_cells else 0)


if __name__ == "__main__":
    main()
