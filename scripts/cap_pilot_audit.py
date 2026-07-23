#!/usr/bin/env python
"""
cap_pilot_audit.py — pick max_new from an uncensored-cap pilot.

Answers "what should max_new really be?" for a cell, from a pilot run whose cap was set
far above any plausible answer length (so the length distribution is NOT censored).

Reports, per cell:
  1. trace-length quantiles;
  2. for each candidate cap, the fraction of traces that WOULD be truncated, and the
     label rate among truncated vs completed traces — the truncation-label leakage
     confound (a cap that truncates mostly-wrong or mostly-right traces makes trace
     length a label proxy, which is exactly what we must not feed the detector);
  3. a repetition score per trace, to separate genuine long reasoning from degenerate
     loops. A looping trace is unbounded — no cap fixes it, so it must not drive the
     choice of cap (cf. cluster/presets.py on the ARS math500 cell, where the mn8192
     pilot tails showed repeat-frac <= 0.08 and the capped traces were therefore real).
  4. a recommended cap: the smallest candidate cap leaving <= --target-pinned of the
     NON-degenerate traces truncated.

Usage:
    python scripts/cap_pilot_audit.py <raw_*.pkl | pilot_dir> [more dirs...]
    python scripts/cap_pilot_audit.py results/pilot_cap/*_mn32768_pilot --target-pinned 0.02
"""
import argparse
import glob
import json
import os
import pickle
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CANDIDATE_CAPS = [1024, 2048, 4096, 8192, 16384, 32768]
NGRAM = 20            # loop detector window; R1 repetition loops are far shorter than this
REPEAT_TAIL = 2000    # score repetition on the trace tail, where loops actually appear


def resolve(path):
    """Accept a raw_*.pkl or a preset directory; return (pkl_path, manifest_or_None)."""
    if os.path.isdir(path):
        pkls = sorted(glob.glob(os.path.join(path, "raw_*.pkl")))
        if not pkls:
            return None, None
        pkl = pkls[0]
    else:
        pkl = path
    man_path = os.path.join(os.path.dirname(pkl), "manifest.json")
    man = json.load(open(man_path)) if os.path.exists(man_path) else None
    return pkl, man


def quantile(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    idx = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def repeat_frac(ids, n=NGRAM, tail=REPEAT_TAIL):
    """1 - distinct/total n-grams over the trace tail. ~0 = genuine text, ->1 = loop."""
    seq = list(ids or [])[-tail:]
    if len(seq) < n + 1:
        return 0.0
    grams = [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def trace_len(c):
    """Generated length in tokens; token_entropies is one entry per generated token."""
    te = c.get("token_entropies") or []
    if te:
        return len(te)
    return len(c.get("gen_token_ids") or [])


def audit(pkl, man, target_pinned, repeat_thresh):
    with open(pkl, "rb") as f:
        data = pickle.load(f)

    cands = [c for i in data for c in data[i]["candidates"]]
    if not cands:
        print("   (no candidates)")
        return

    run_cap = (man or {}).get("max_new")
    lens = [trace_len(c) for c in cands]
    labels = [bool(c.get("label", False)) for c in cands]
    reps = [repeat_frac(c.get("gen_token_ids")) for c in cands]
    degen = [r >= repeat_thresh for r in reps]
    n = len(cands)
    n_deg = sum(degen)

    slen = sorted(lens)
    print(f"   problems={len(data)}  candidates={n}  run cap(max_new)={run_cap}")
    print(f"   length: p50={quantile(slen,.50)}  p75={quantile(slen,.75)}  "
          f"p90={quantile(slen,.90)}  p95={quantile(slen,.95)}  p99={quantile(slen,.99)}  "
          f"max={slen[-1]}  mean={sum(lens)/n:.0f}")

    # Did the PILOT itself censor? If so every number below is a lower bound.
    if run_cap:
        at_cap = sum(1 for x in lens if x >= run_cap)
        flag = "  <-- PILOT CAP STILL BINDING; raise it and re-run" if at_cap else ""
        print(f"   traces at the pilot cap: {at_cap}/{n} ({at_cap/n:.1%}){flag}")

    print(f"   repetition (n={NGRAM}-gram over last {REPEAT_TAIL} tok): "
          f"degenerate (>= {repeat_thresh}) {n_deg}/{n} ({n_deg/n:.1%})")
    if n_deg:
        dl = sorted(l for l, d in zip(lens, degen) if d)
        print(f"     degenerate lengths: p50={quantile(dl,.50)} max={dl[-1]} "
              f"— unbounded by nature; excluded from the cap choice")

    # ── would-be truncation at each candidate cap ──────────────────────────────
    print(f"\n   {'cap':>7}  {'pinned(all)':>12}  {'pinned(clean)':>14}  "
          f"{'acc|truncated':>14}  {'acc|complete':>13}  {'leak Δ':>7}")
    rec = None
    for cap in CANDIDATE_CAPS:
        if run_cap and cap > run_cap:
            continue  # beyond what the pilot observed — no evidence either way
        trunc = [l >= cap for l in lens]
        n_tr = sum(trunc)
        clean_tr = sum(1 for t, d in zip(trunc, degen) if t and not d)
        n_clean = n - n_deg
        frac_clean = clean_tr / n_clean if n_clean else float("nan")

        acc_tr = (sum(1 for t, y in zip(trunc, labels) if t and y) / n_tr) if n_tr else float("nan")
        n_ok = n - n_tr
        acc_ok = (sum(1 for t, y in zip(trunc, labels) if not t and y) / n_ok) if n_ok else float("nan")
        delta = (acc_tr - acc_ok) if (n_tr and n_ok) else float("nan")

        if rec is None and frac_clean == frac_clean and frac_clean <= target_pinned:
            rec = cap
        mark = " <-" if rec == cap else ""
        print(f"   {cap:>7}  {n_tr:>4}/{n} {n_tr/n:>5.1%}  {clean_tr:>5}/{n_clean} "
              f"{frac_clean:>6.1%}  {acc_tr:>14.3f}  {acc_ok:>13.3f}  {delta:>+7.3f}{mark}")

    print()
    if rec:
        print(f"   RECOMMENDED max_new = {rec}  "
              f"(<= {target_pinned:.0%} of non-degenerate traces truncated)")
    else:
        print(f"   NO candidate cap gets non-degenerate truncation under {target_pinned:.0%} "
              f"— the tail is longer than every cap tested; raise the pilot cap.")
    print("   'leak Δ' = acc(truncated) - acc(complete). Large |Δ| means the cap itself "
          "encodes the label;\n   a cap is only safe when it is both rare AND non-selective.")


def main():
    ap = argparse.ArgumentParser(description="Choose max_new from an uncensored-cap pilot.")
    ap.add_argument("paths", nargs="+", help="raw_*.pkl files or pilot directories")
    ap.add_argument("--target-pinned", type=float, default=0.02,
                    help="max acceptable truncated fraction among non-degenerate traces")
    ap.add_argument("--repeat-thresh", type=float, default=0.30,
                    help="repeat-frac at or above which a trace counts as a degenerate loop")
    args = ap.parse_args()

    for path in args.paths:
        pkl, man = resolve(path)
        if pkl is None:
            print(f"== {path}\n   (no raw_*.pkl yet — still running?)\n")
            continue
        size_mb = os.path.getsize(pkl) / 1e6
        print(f"== {path}  ({os.path.basename(pkl)}, {size_mb:.1f} MB)")
        if man:
            print(f"   model={man.get('model')}  dataset={man.get('dataset')}  "
                  f"K={man.get('k')}  T={man.get('temps')}")
        audit(pkl, man, args.target_pinned, args.repeat_thresh)
        print()


if __name__ == "__main__":
    main()
