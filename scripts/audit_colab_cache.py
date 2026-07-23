#!/usr/bin/env python
"""
audit_colab_cache.py — Drive raw-pkl key audit for the full-coverage plan
(HANDOFF_full_coverage.md step 1; blocking input for the Colab-era backfill specs).

Runs in ONE Colab cell after `drive.mount` (no torch, no model loads — numpy only,
which Colab has). Also runs locally against any directory (that is how it is smoke-
tested: point --root at cache/repgrid).

For every candidate-like pkl under the roots it reports: schema kind, N/K, key
presence per candidate (the inspect_cell.py groups + prompt/context keys that decide
RAG reconstructibility), trace lengths, total generation tokens (the GPU-hour input),
temperature (parsed from path, with source), a model guess (from path tokens), and a
recovery-tier classification:

  complete — token_logsumexp present (nothing to backfill)
  tier1    — top_k_logprobs and/or token_spilled_energies already saved (lp/xlp/ΔE
             extractable OFFLINE; Z_n still needs the GPU pass unless complete)
  tier2    — gen_token_ids saved -> teacher-forced backfill recovers EVERYTHING
  tier2r   — no gen_token_ids but full_text saved -> backfill possible only if the
             full_text re-tokenization roundtrips (verify per tokenizer; risky)
  tier3    — neither -> only re-generation or drop (decision: Omri, post-audit)

Colab usage:
    from google.colab import drive; drive.mount('/content/drive')
    !cd /content/hallucination_detection && python scripts/audit_colab_cache.py \
        --root /content/drive/MyDrive --scan \
        --out /content/drive/MyDrive/hallucination_detection/coverage_audit.csv

Local usage (smoke):
    python scripts/audit_colab_cache.py --root cache/repgrid --scan --out -
"""
import argparse
import csv
import os
import pickle
import re
import sys

BASE_KEYS = ["full_text", "token_entropies", "token_spilled_energies", "token_offsets",
             "top_k_logprobs", "gen_token_ids", "label"]
ENERGY_KEYS = ["token_logsumexp", "top_k_logprobs_raw"]
PROMPT_KEYS = ["prompt", "prompt_token_ids", "question", "context", "passages",
               "retrieved", "docs"]

# Known/likely Drive cache dirs (relative to --root, typically MyDrive). --scan walks
# everything under --root instead; this list only orders/annotates the output.
KNOWN_SOURCES = [
    "epr_spectral_gpqa_72b",
    "spectral_phase9_cache",
    "hallucination_detection/cache",
    "hallucination_detection/consolidated_results",
    "spectral_phase10_cache",
    "epr_spectral_gsm8k_vs_lapei",
]

_TEMP_PATTERNS = [
    (re.compile(r"[_/\\]T[= ]?([0-9]+(?:\.[0-9]+)?)(?:[_/\\.]|$)"), "path_T"),
    (re.compile(r"temp[_= ]?([0-9]+(?:\.[0-9]+)?)", re.I), "path_temp"),
]

_MODEL_TOKENS = re.compile(
    r"(qwen[0-9.\-]*[a-z0-9.\-]*|llama[-_0-9.]*[a-z0-9.\-]*|mistral[a-z0-9.\-]*|"
    r"phi[-_0-9.]*[a-z0-9.\-]*|deepseek[a-z0-9.\-]*|opt-[0-9]+b|gemma[a-z0-9.\-]*|"
    r"nemo[a-z0-9.\-]*|r1[-_]?distill[a-z0-9.\-]*)", re.I)


def parse_temp(path):
    for pat, src in _TEMP_PATTERNS:
        m = pat.search(path)
        if m:
            return float(m.group(1)), src
    return None, "unknown"


def guess_model(path):
    hits = _MODEL_TOKENS.findall(path)
    return hits[-1] if hits else ""


def _present(v):
    if v is None:
        return False
    if isinstance(v, (list, tuple, dict, str)):
        return len(v) > 0
    return True


def sniff_candidates(obj):
    """Return (schema, [(entry_idx, cand_dict), ...]) or (None, []) if not
    candidate-like. Handles the repgrid {idx:{candidates:[...]}} schema, flat
    {key: cand} dicts, and flat [cand, ...] lists."""
    if isinstance(obj, dict) and obj:
        vals = list(obj.values())
        if all(isinstance(v, dict) and "candidates" in v for v in vals[:20]):
            out = []
            for i, v in obj.items():
                for c in v.get("candidates", []):
                    if isinstance(c, dict):
                        out.append((i, c))
            return "repgrid", out
        if all(isinstance(v, dict) for v in vals[:20]) and any(
                ("token_entropies" in v or "full_text" in v) for v in vals[:20]):
            return "flat_dict", [(i, v) for i, v in obj.items()]
        # one level of nesting: {run_key: {idx: cand}} or {run_key: [cand,...]}
        nested = []
        for k, v in obj.items():
            s, cands = sniff_candidates(v)
            if s:
                nested.extend(cands)
        if nested:
            return "nested_dict", nested
    if isinstance(obj, list) and obj and all(isinstance(v, dict) for v in obj[:20]) \
            and any(("token_entropies" in v or "full_text" in v) for v in obj[:20]):
        return "flat_list", list(enumerate(obj))
    return None, []


def classify_tier(frac):
    if frac.get("token_logsumexp", 0) > 0.99:
        return "complete"
    if frac.get("gen_token_ids", 0) > 0.99:
        return "tier2"
    if frac.get("top_k_logprobs", 0) > 0.99 or frac.get("token_spilled_energies", 0) > 0.99:
        # offline-extractable keys exist, but full recovery still needs a roundtrip
        return "tier1" if frac.get("full_text", 0) < 0.99 else "tier1+2r"
    if frac.get("full_text", 0) > 0.99:
        return "tier2r"
    return "tier3"


def audit_pkl(path, root, max_gb):
    size_mb = os.path.getsize(path) / 1e6
    rel = os.path.relpath(path, root)
    row = {"pkl": rel, "size_mb": round(size_mb, 1)}
    if size_mb > max_gb * 1000:
        row["notes"] = f"skipped: > {max_gb} GB (use --max-gb to raise)"
        return row
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
    except Exception as e:
        row["notes"] = f"unreadable: {type(e).__name__}: {e}"
        return row

    schema, cands_idx = sniff_candidates(obj)
    if not schema:
        row["notes"] = "not candidate-like (feature/result pkl?)"
        return row
    cands = [c for _, c in cands_idx]
    idxs = [i for i, _ in cands_idx]
    n_problems = len(set(idxs))
    n_cand = len(cands)

    frac = {}
    for k in BASE_KEYS + ENERGY_KEYS + PROMPT_KEYS:
        frac[k] = sum(_present(c.get(k)) for c in cands) / max(n_cand, 1)

    lens = [len(c.get("gen_token_ids") or c.get("token_entropies") or [])
            for c in cands]
    lens_nz = [x for x in lens if x]
    temp, temp_src = parse_temp(path)

    row.update({
        "schema": schema,
        "n_problems": n_problems,
        "n_candidates": n_cand,
        "k": round(n_cand / max(n_problems, 1), 2),
        "model_guess": guess_model(path),
        "temp": temp,
        "temp_source": temp_src,
        "mean_trace": round(sum(lens_nz) / max(len(lens_nz), 1), 1),
        "sum_gen_tokens": sum(lens),
        "tier": classify_tier(frac),
        "notes": "",
    })
    for k in BASE_KEYS + ENERGY_KEYS + PROMPT_KEYS:
        row[f"has_{k}"] = round(frac[k], 3)
    return row


def find_pkls(root, scan):
    seen = []
    if scan:
        for dirpath, dirnames, filenames in os.walk(root):
            # never descend into HF caches / checkpoints — huge and never candidate pkls
            dirnames[:] = [d for d in dirnames
                           if d not in ("hf_cache", "hf_cache_flat", ".git",
                                        "__pycache__", "checkpoints")]
            for fn in filenames:
                if fn.endswith(".pkl") and not fn.endswith(".tmp"):
                    seen.append(os.path.join(dirpath, fn))
    else:
        for src in KNOWN_SOURCES:
            d = os.path.join(root, src)
            if not os.path.isdir(d):
                continue
            for dirpath, _, filenames in os.walk(d):
                for fn in filenames:
                    if fn.endswith(".pkl") and not fn.endswith(".tmp"):
                        seen.append(os.path.join(dirpath, fn))
    return sorted(seen)


def main():
    ap = argparse.ArgumentParser(description="Drive raw-pkl coverage audit")
    ap.add_argument("--root", required=True,
                    help="scan root (Colab: /content/drive/MyDrive)")
    ap.add_argument("--scan", action="store_true",
                    help="walk the whole root instead of only KNOWN_SOURCES")
    ap.add_argument("--max-gb", type=float, default=8.0,
                    help="skip pkls larger than this (Colab RAM guard)")
    ap.add_argument("--out", default="coverage_audit.csv",
                    help="output CSV path, or - for stdout")
    args = ap.parse_args()

    pkls = find_pkls(args.root, args.scan)
    print(f"[audit] {len(pkls)} pkl(s) under {args.root}", file=sys.stderr)

    rows = []
    for i, p in enumerate(pkls):
        print(f"[audit] ({i+1}/{len(pkls)}) {os.path.relpath(p, args.root)}",
              file=sys.stderr, flush=True)
        rows.append(audit_pkl(p, args.root, args.max_gb))

    fields = ["pkl", "size_mb", "schema", "n_problems", "n_candidates", "k",
              "model_guess", "temp", "temp_source", "mean_trace", "sum_gen_tokens",
              "tier"] + [f"has_{k}" for k in BASE_KEYS + ENERGY_KEYS + PROMPT_KEYS] \
             + ["notes"]
    out = sys.stdout if args.out == "-" else open(args.out, "w", newline="",
                                                  encoding="utf-8")
    w = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    if out is not sys.stdout:
        out.close()
        print(f"[audit] wrote {args.out} ({len(rows)} rows)", file=sys.stderr)

    # summary to stderr for the Colab cell output
    tiers = {}
    for r in rows:
        tiers[r.get("tier", "unreadable/skipped")] = \
            tiers.get(r.get("tier", "unreadable/skipped"), 0) + 1
    print(f"[audit] tier summary: {tiers}", file=sys.stderr)


if __name__ == "__main__":
    main()
