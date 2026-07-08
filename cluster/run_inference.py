#!/usr/bin/env python
"""
Standalone GPU inference driver for the AIRCC cluster (Slurm + Pyxis, B200).

Runs K-candidate sampling per problem and saves the richest raw form per candidate
(full_text, token_entropies, token_spilled_energies, token_offsets, top_k_logprobs,
gen_token_ids, label — plus token_logsumexp / hidden_middle_last when a preset enables
them) so every future feature/baseline can be derived offline — including offline
re-grading, since full_text is kept.

Two ways to run:
  1. Replication-grid preset (recommended) — every paper-matched parameter comes from
     cluster/presets.py; individual flags still override for pilots:
        python cluster/run_inference.py --preset inside_coqa_llama7b
        python cluster/run_inference.py --preset inside_coqa_llama7b --n-samples 30  # pilot
  2. Ad-hoc dataset (the original math path):
        python cluster/run_inference.py --dataset aime24 --temps 0.2,0.6,1.0 \
            --k 8 --n-samples 30 --max-new 3072 --out /shared/.../results/edis_aime24

Guardrails from the EDIS/AIME24 test run (replication-grid plan):
  - MAX_NEW is per-preset, never a blanket 1024 (1024 truncated the entropy trace).
  - Each completed (dataset, T) cell is checked against an accuracy-band gate: a cell
    whose accuracy floors/ceilings (AIME24 x Qwen-1.5B hit ~2%) is flagged REJECT —
    its AUROC is not trustworthy.
  - A manifest.json with full provenance is written next to the pkls.

Preemption-safe: Slurm sends SIGTERM 15 minutes before SIGKILL (--signal=B:TERM@900
in the sbatch template). The handler sets a flag checked after every generation; the
driver checkpoints atomically and exits 0. The job is auto-requeued and resumes exactly
where it stopped — completed problems are skipped, partial ones filled per candidate.
"""
import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from types import SimpleNamespace

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # so `import presets` works

import torch

from presets import get_preset, list_presets
from spectral_utils import load_model, generate_full, load_cache, save_cache_atomic
from spectral_utils.data_loaders import (
    load_gsm8k, gsm8k_prompt, is_correct_gsm8k,
    load_math500, math_prompt, is_correct_math,
    load_amc23, amc23_prompt, is_correct_amc23,
    load_aime24, aime24_prompt, is_correct_aime24,
    load_hotpotqa, hotpotqa_prompt, is_correct_hotpotqa,
    load_trivia_qa, trivia_qa_prompt, is_correct_trivia_qa,
    load_webq, webq_prompt, is_correct_webq,
    load_coqa, coqa_prompt, is_correct_coqa,
    load_squad_v2, squad_v2_prompt, is_correct_squad_v2,
    load_nq_open, nq_open_prompt, is_correct_nq_open,
    load_truthfulqa, truthfulqa_prompt, is_correct_truthfulqa,
    load_sciq, sciq_prompt, is_correct_sciq,
)

# dataset -> (loader(n, split) -> list[row], prompt_fn(row) -> str, grader(gen, row) -> bool)
# All loaders share a (n_samples, split) signature; QA prompts/graders take the row dict.
DATASETS = {
    "gsm8k":     (lambda n, split="test":       load_gsm8k(split=split)[:n], gsm8k_prompt, is_correct_gsm8k),
    "math500":   (lambda n, split=None:         load_math500(n),   math_prompt,  is_correct_math),
    "amc23":     (lambda n, split=None:         load_amc23(n),     amc23_prompt, is_correct_amc23),
    "aime24":    (lambda n, split=None:         load_aime24(n),    aime24_prompt, is_correct_aime24),
    "hotpotqa":  (lambda n, split="validation": load_hotpotqa(n),  hotpotqa_prompt,
                  lambda gen, row: is_correct_hotpotqa(gen, row.get("answer", ""))),
    "trivia_qa": (lambda n, split="validation": load_trivia_qa(n, split),
                  lambda row: trivia_qa_prompt(row["question"]), is_correct_trivia_qa),
    "webq":      (lambda n, split="test":       load_webq(n, split),
                  lambda row: webq_prompt(row["question"]), is_correct_webq),
    "coqa":      (lambda n, split="validation": load_coqa(n, split),     coqa_prompt,     is_correct_coqa),
    "squad_v2":  (lambda n, split="validation": load_squad_v2(n, split), squad_v2_prompt, is_correct_squad_v2),
    "nq_open":   (lambda n, split="validation": load_nq_open(n, split),  nq_open_prompt,  is_correct_nq_open),
    "truthfulqa":(lambda n, split="validation": load_truthfulqa(n, split), truthfulqa_prompt, is_correct_truthfulqa),
    "sciq":      (lambda n, split="validation": load_sciq(n, split),     sciq_prompt,     is_correct_sciq),
}

STOP = {"flag": False}


def _on_sigterm(signum, frame):
    STOP["flag"] = True
    print("[driver] SIGTERM received — will checkpoint after current generation", flush=True)


def question_text(row) -> str:
    if isinstance(row, dict):
        return row.get("question") or row.get("problem") or row.get("query") or ""
    return str(row)


def accuracy_gate(labels, acc_band, min_minority):
    """Accuracy-band gate (U2). A cell's AUROC is only trustworthy when both classes are
    populated. Returns (ok, stats_dict)."""
    n = len(labels)
    pos = sum(1 for x in labels if x)
    neg = n - pos
    acc = pos / n if n else 0.0
    minority = min(pos, neg)
    lo, hi = acc_band
    reasons = []
    if not (lo <= acc <= hi):
        reasons.append(f"acc {acc:.3f} outside [{lo},{hi}]")
    if minority < min_minority:
        reasons.append(f"minority class {minority} < {min_minority}")
    ok = not reasons
    return ok, {"accuracy": acc, "pos": pos, "neg": neg, "minority": minority,
                "gate_ok": ok, "gate_reasons": reasons}


def run_temp(mdl, tok, rows, prompt_fn, grader, temp, cfg, out_path):
    """Run one (dataset, temperature) cell. Returns (completed, stats|None).
    completed=False means preempted mid-run (Slurm will requeue)."""
    cache = load_cache(out_path)
    n_done = sum(1 for e in cache.values() if len(e["candidates"]) >= cfg.k)
    print(f"\n=== T={temp} -> {out_path} ({n_done}/{len(rows)} problems already complete) ===",
          flush=True)

    for idx, row in enumerate(rows):
        entry = cache.setdefault(idx, {
            "question": question_text(row),
            "gold_row": row,
            "candidates": [],
        })
        dirty = False
        while len(entry["candidates"]) < cfg.k:
            if STOP["flag"]:
                save_cache_atomic(cache, out_path)
                print(f"PREEMPTED — checkpoint saved at T={temp} problem={idx} "
                      f"candidate={len(entry['candidates'])}", flush=True)
                return False, None
            t0 = time.time()
            r = generate_full(
                mdl, tok, prompt_fn(row), temperature=temp,
                max_new_tokens=cfg.max_new,
                logprob_top_k=cfg.logprob_top_k,
                gen_top_p=cfg.gen_top_p, gen_top_k=cfg.gen_top_k,
                capture_logsumexp=cfg.capture.get("logsumexp", False),
                capture_hidden=cfg.capture.get("hidden", False),
                hidden_layer=cfg.capture.get("hidden_layer"),
                capture_attention=cfg.capture.get("attention", False),
                capture_layer_fft=cfg.capture.get("layer_fft", False),
            )
            r["label"] = bool(grader(r["full_text"], row))
            entry["candidates"].append(r)
            print(f"[T={temp}] problem {idx + 1}/{len(rows)} cand "
                  f"{len(entry['candidates'])}/{cfg.k}: {len(r['token_entropies'])} tok, "
                  f"label={int(r['label'])}, {time.time() - t0:.1f}s", flush=True)
            dirty = True
        if dirty and (idx + 1) % cfg.checkpoint_every == 0:
            save_cache_atomic(cache, out_path)

    save_cache_atomic(cache, out_path)
    labels = [c["label"] for e in cache.values() for c in e["candidates"]]
    lens = [len(c["token_entropies"]) for e in cache.values() for c in e["candidates"]]
    mean_trace = sum(lens) / max(len(lens), 1)
    ok, gate = accuracy_gate(labels, cfg.acc_band, cfg.min_minority)
    verdict = "VALID" if ok else "REJECT"
    print(f"=== T={temp} DONE: {len(cache)} problems x {cfg.k} candidates | "
          f"accuracy {gate['accuracy']:.3f} | mean trace {mean_trace:.0f} tok | "
          f"GATE {verdict} (pos={gate['pos']}, neg={gate['neg']}, minority={gate['minority']}) ===",
          flush=True)
    if not ok:
        print(f"[GATE] T={temp} REJECT — {'; '.join(gate['gate_reasons'])}. "
              f"AUROC on this cell is not trustworthy; re-scope (larger paper model / "
              f"easier split) before treating it as a data cell.", flush=True)
    stats = {"temp": temp, "pkl": os.path.basename(out_path),
             "n_problems": len(cache), "k": cfg.k, "mean_trace": mean_trace, **gate}
    return True, stats


def write_manifest(out_dir, cfg, cells):
    """Provenance manifest next to the pkls (U-plan data-org)."""
    manifest = {
        "preset_id": cfg.preset_id,
        "paper": cfg.paper,
        "model": cfg.model,
        "dataset": cfg.dataset,
        "split": cfg.split,
        "n_samples": cfg.n_samples,
        "k": cfg.k,
        "temps": cfg.temps,
        "gen_top_p": cfg.gen_top_p,
        "gen_top_k": cfg.gen_top_k,
        "max_new": cfg.max_new,
        "logprob_top_k": cfg.logprob_top_k,
        "capture": cfg.capture,
        "acc_band": list(cfg.acc_band),
        "min_minority": cfg.min_minority,
        "published": cfg.published,
        "notes": cfg.notes,
        "job_id": os.environ.get("SLURM_JOB_ID", ""),
        "seed": cfg.seed,
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cells": cells,
    }
    path = os.path.join(out_dir, "manifest.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    os.replace(tmp, path)
    print(f"[driver] manifest -> {path}", flush=True)


def build_cfg(args):
    """Merge preset (if any) with explicit CLI overrides. CLI wins when provided."""
    base = get_preset(args.preset) if args.preset else {}

    def pick(cli_val, key, default):
        if cli_val is not None:
            return cli_val
        return base.get(key, default)

    dataset = args.dataset or base.get("dataset")
    if dataset is None:
        raise SystemExit("must pass --preset or --dataset")
    if dataset not in DATASETS:
        raise SystemExit(f"unknown dataset {dataset!r}; known: {sorted(DATASETS)}")

    temps_str = args.temps
    if temps_str is not None:
        temps = [float(t) for t in temps_str.split(",")]
    else:
        temps = base.get("temps", [1.0])

    return SimpleNamespace(
        preset_id=args.preset or "",
        paper=base.get("paper", ""),
        model=pick(args.model, "model", "Qwen/Qwen2.5-Math-1.5B-Instruct"),
        dataset=dataset,
        split=pick(args.split, "split", None),
        n_samples=pick(args.n_samples, "n_samples", 30),
        k=pick(args.k, "k", 8),
        temps=temps,
        max_new=pick(args.max_new, "max_new", 1024),
        logprob_top_k=pick(args.logprob_top_k, "logprob_top_k", 50),
        gen_top_p=base.get("gen_top_p"),
        gen_top_k=base.get("gen_top_k", 50),
        capture=base.get("capture", {}),
        acc_band=tuple(base.get("acc_band", (0.20, 0.85))),
        min_minority=base.get("min_minority", 30),
        published=base.get("published", {}),
        notes=base.get("notes", ""),
        checkpoint_every=args.checkpoint_every,
        seed=args.seed,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--preset", default=None, choices=list_presets() + [None],
                    help="replication-grid preset id (see cluster/presets.py)")
    ap.add_argument("--dataset", default=None, choices=sorted(DATASETS) + [None])
    ap.add_argument("--model", default=None)
    ap.add_argument("--split", default=None)
    ap.add_argument("--temps", default=None, help="comma-separated sampling temperatures")
    ap.add_argument("--k", type=int, default=None, help="candidates per problem")
    ap.add_argument("--n-samples", type=int, default=None, help="number of problems")
    ap.add_argument("--max-new", type=int, default=None)
    ap.add_argument("--logprob-top-k", type=int, default=None,
                    help="top-K logprobs saved per token (0 disables)")
    ap.add_argument("--out", default=None,
                    help="output dir (default: results_<preset|dataset> under CWD)")
    ap.add_argument("--checkpoint-every", type=int, default=1,
                    help="save cache every N completed problems")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = build_cfg(args)
    signal.signal(signal.SIGTERM, _on_sigterm)
    torch.manual_seed(cfg.seed)

    out_dir = args.out or f"results_{cfg.preset_id or cfg.dataset}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"[driver] preset={cfg.preset_id or '(none)'} model={cfg.model} "
          f"dataset={cfg.dataset} split={cfg.split} N={cfg.n_samples} K={cfg.k} "
          f"temps={cfg.temps} max_new={cfg.max_new} capture={cfg.capture}", flush=True)
    if torch.cuda.is_available():
        print(f"[driver] GPU: {torch.cuda.get_device_name(0)} "
              f"capability={torch.cuda.get_device_capability(0)}", flush=True)
    else:
        print("[driver] WARNING: no CUDA device — running on CPU", flush=True)

    loader, prompt_fn, grader = DATASETS[cfg.dataset]
    rows = loader(cfg.n_samples, cfg.split)
    write_manifest(out_dir, cfg, cells=[])  # provenance up front; refreshed per cell
    mdl, tok = load_model(cfg.model)

    cells = []
    for temp in cfg.temps:
        out_path = os.path.join(out_dir, f"raw_{cfg.dataset}_T{temp}.pkl")
        completed, stats = run_temp(mdl, tok, rows, prompt_fn, grader, temp, cfg, out_path)
        if not completed:
            sys.exit(0)  # preempted: clean exit, Slurm requeues and we resume
        cells.append(stats)
        write_manifest(out_dir, cfg, cells)  # refresh with this cell's gate verdict

    print(f"\nALL TEMPS COMPLETE for {cfg.dataset} -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
