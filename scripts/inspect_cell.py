#!/usr/bin/env python
"""
inspect_cell.py — standard schema report for a replication-grid raw pkl.

Replaces the throwaway `python -c "import pickle; ..."` spelunking that gets rewritten every
time a fresh cell lands. Prints, for a cell: N problems, K per problem, label distribution,
trace lengths, a per-candidate key-presence line (the 7 rich-save keys + the Step-161 energy
keys + judge labels + hidden state), and — via the real offline loader — the extractable
feature set + valid-rate for the canonical subsets. Reads the sibling manifest.json for
model / dataset / published Y / head_to_head. No torch, no model load.

Usage:
    python scripts/inspect_cell.py <raw_*.pkl | cache/repgrid/<preset_dir>>
"""
import argparse
import glob
import json
import os
import pickle
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from spectral_utils.repgrid_scoring import load_repgrid_cell, subset_matrix

# Canonical rich-save schema (CLAUDE.md "save everything, derive later") + the extras.
BASE_KEYS = ["full_text", "token_entropies", "token_spilled_energies", "token_offsets",
             "top_k_logprobs", "gen_token_ids", "label"]
ENERGY_KEYS = ["token_logsumexp", "top_k_logprobs_raw"]      # Step-161 raw-energy capture
JUDGE_KEYS = ["label_judged", "label_lexical"]               # LLM-judge second pass
HIDDEN_KEYS = ["hidden_middle_last"]                          # INSIDE capture

GOOD_5 = ["epr", "low_band_power", "sw_var_peak", "cusum_max", "spectral_entropy"]
GOOD_5_ENERGY = GOOD_5 + ["epr_energy", "min_energy", "sw_var_peak_energy", "cusum_max_energy"]
GOOD_5_LOGPROB = GOOD_5 + ["mean_top1_logprob", "logprob_margin", "mean_logprob_entropy"]


def resolve(path):
    """Return (pkl_path, manifest_or_None). Accepts a pkl file or a preset directory."""
    if os.path.isdir(path):
        pkls = sorted(glob.glob(os.path.join(path, "raw_*.pkl")))
        if not pkls:
            sys.exit(f"no raw_*.pkl in {path}")
        pkl = pkls[0]
    else:
        pkl = path
    man_path = os.path.join(os.path.dirname(pkl), "manifest.json")
    man = json.load(open(man_path)) if os.path.exists(man_path) else None
    return pkl, man


def _present(v):
    """A key counts as present when it exists and is not None/empty."""
    if v is None:
        return False
    if isinstance(v, (list, tuple, dict, str)):
        return len(v) > 0
    return True


def main():
    ap = argparse.ArgumentParser(description="Schema report for a replication-grid raw pkl.")
    ap.add_argument("path", help="raw_*.pkl file or a cache/repgrid/<preset> directory")
    args = ap.parse_args()

    pkl, man = resolve(args.path)
    size_mb = os.path.getsize(pkl) / 1e6
    print(f"== {os.path.relpath(pkl, REPO)}  ({size_mb:.1f} MB) ==")
    if man:
        pub = man.get("published") or {}
        print(f"   model={man.get('model')}  dataset={man.get('dataset')}  "
              f"N={man.get('n_samples')} K={man.get('k')} T={man.get('temps')}")
        print(f"   published Y={pub.get('value')} ({pub.get('method','')})  "
              f"head_to_head={man.get('head_to_head')}  judge={man.get('judge')}")

    with open(pkl, "rb") as f:
        data = pickle.load(f)

    # ── structure + K ──────────────────────────────────────────────────────────
    n_problems = len(data)
    ks = [len(data[i]["candidates"]) for i in data]
    cands = [c for i in data for c in data[i]["candidates"]]
    n_cand = len(cands)
    print(f"\n   problems={n_problems}  candidates={n_cand}  "
          f"K/problem: min={min(ks)} max={max(ks)} "
          f"{'(uniform)' if min(ks) == max(ks) else '(RAGGED)'}")

    # ── labels ─────────────────────────────────────────────────────────────────
    labels = [bool(c.get("label", False)) for c in cands]
    pos = sum(labels)
    neg = n_cand - pos
    acc = pos / max(n_cand, 1)
    print(f"   labels: acc={acc:.3f}  pos={pos}  neg={neg}  minority={min(pos, neg)}")
    if any("label_lexical" in c for c in cands):
        lex = [bool(c.get("label_lexical", False)) for c in cands]
        agree = sum(a == b for a, b in zip(labels, lex)) / max(n_cand, 1)
        print(f"   judge vs lexical agreement={agree:.3f}  (lexical acc={sum(lex)/max(n_cand,1):.3f})")

    # ── trace lengths ──────────────────────────────────────────────────────────
    tl = sorted(len(c.get("token_entropies", []) or []) for c in cands)
    if tl:
        mean = sum(tl) / len(tl)
        median = tl[len(tl) // 2]
        short = sum(1 for x in tl if x < 8)   # extract_all_features returns None below 8
        print(f"   trace len: mean={mean:.1f} median={median} min={tl[0]} max={tl[-1]}  "
              f"<8 tok (no spectral): {short}/{n_cand} ({short/max(n_cand,1):.0%})")

    # ── key presence (fraction of candidates carrying each key) ─────────────────
    def presence_line(title, keys):
        cells = []
        for k in keys:
            frac = sum(_present(c.get(k)) for c in cands) / max(n_cand, 1)
            tag = "yes" if frac > 0.999 else ("no " if frac < 0.001 else f"{frac:.2f}")
            cells.append(f"{k}={tag}")
        print(f"   {title:8s} " + "  ".join(cells))

    print("\n   key presence (per candidate):")
    presence_line("base", BASE_KEYS)
    presence_line("energy", ENERGY_KEYS)
    presence_line("judge", JUDGE_KEYS)
    presence_line("hidden", HIDDEN_KEYS)

    # ── feature extractability via the real offline loader ──────────────────────
    print("\n   feature extraction (via load_repgrid_cell):")
    cell = load_repgrid_cell(pkl)
    print(f"      available features ({len(cell['available'])}): {', '.join(cell['available'])}")
    for name, subset in [("GOOD_5", GOOD_5), ("GOOD_5+energy", GOOD_5_ENERGY),
                         ("GOOD_5+logprob", GOOD_5_LOGPROB)]:
        present = [f for f in subset if f in cell["available"]]
        if len(present) < 3:
            print(f"      {name:16s} unavailable ({len(present)}/{len(subset)} feats present)")
            continue
        _, valid = subset_matrix(cell["rows"], present)
        vr = int(valid.sum()) / max(len(cell["rows"]), 1)
        print(f"      {name:16s} {len(present)}/{len(subset)} feats, valid rows={int(valid.sum())} "
              f"({vr:.2f}) of {len(cell['rows'])}")


if __name__ == "__main__":
    main()
