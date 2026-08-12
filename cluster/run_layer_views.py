#!/usr/bin/env python
"""Per-layer white-box telemetry driver — writes the depth field for existing cells.

Teacher-forced, nothing is generated.  For every candidate that already carries
``gen_token_ids``, one forward pass records the per-layer / per-module logit-lens field
and the residual-stream geometry defined in ``cluster/layer_lens.py``.

DESIGN COMMITMENTS (deliberate, do not "simplify" these away)

1. **Writes to a SIDECAR, never into the canonical pkl.**  The published caches stay
   byte-identical; the depth field is 100+ MB per cell and would bloat every sync and
   every LFS chunk.  The sidecar is keyed by (problem_idx, cand_idx) so it re-joins the
   cache positionally.

2. **Gate B is reused unchanged and is still blocking.**  A wrong chat template
   produces a plausible but systematically shifted field that is invisible downstream.
   The gate compares recomputed top-15 final-layer entropies against the SAVED trace,
   with the tolerances calibrated in job 123504.  It costs nothing extra here because
   the final-layer logits come out of the same forward pass.

3. **No pooling, no view definition, no layer selection happens on the GPU.**  Those
   are the research questions; they are answered on CPU against this file.

Usage:
    python cluster/run_layer_views.py --cells lapeigvals_gsm8k_llama8b --dry-run
    python cluster/run_layer_views.py --cells spilled_triviaqa_llama8b --validate-only
    python cluster/run_layer_views.py --cells lapeigvals_gsm8k_llama8b,se_squad_v2_llama8b
"""
import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

try:
    import transformers.modeling_utils as _mu
    _mu.check_torch_load_is_safe = lambda *a, **k: None
except Exception:
    pass

from backfill_specs import resolve_spec, list_backfill_cells, BACKFILL_SPECS
from backfill_views import (
    GateStats, build_prompt_ids, build_warpers, candidate_gen_ids,
    candidate_quantities, gate_b_verdict, get_aliased, iter_problems, _git_sha,
    _stub_pcre,
)
from layer_lens import (
    MODULES, QUANTITIES, COV_EIGS_R, HID_PROJ_DIM, HID_PROJ_SEED,
    ModuleTap, candidate_layer_field, make_hid_proj, resolve_stack,
    verify_residual_reconstruction,
)
from spectral_utils import load_model, load_cache, save_cache_atomic, free_memory

EXIT_INCOMPLETE = 85
STOP = {"flag": False}
SIDECAR_VERSION = "layer-lens-v1"


def _on_sigterm(signum, frame):
    STOP["flag"] = True
    print("[layers] SIGTERM received — will checkpoint after current problem", flush=True)


def sidecar_path(spec, temp):
    return os.path.join(spec.data_dir, f"layer_views_T{temp}.pkl")


def process_pkl(mdl, tok, spec, temp, pkl_path, args):
    """Gate + extract one raw pkl into its sidecar. Returns (completed, report)."""
    try:
        layers, _, _ = resolve_stack(mdl)
    except RuntimeError as e:
        # Unsupported family (OPT's decoder.layers/fc1/fc2, fused blocks, ...) —
        # report and move on rather than failing the whole job.
        print(f"[layers] {spec.cell_id}: UNSUPPORTED ARCHITECTURE — {e}", flush=True)
        return True, {"pkl": os.path.basename(pkl_path), "temp": temp,
                      "arch_check_pass": False, "arch_error": str(e),
                      "aborted": True, "gate_b_pass": None}
    L = len(layers)
    hidden_size = mdl.config.hidden_size
    hid_proj = make_hid_proj(hidden_size, mdl.device, dim=args.proj_dim)

    cache = load_cache(pkl_path)
    if not cache:
        return True, {"pkl": os.path.basename(pkl_path), "error": "empty cache"}

    side_path = sidecar_path(spec, temp)
    side = load_cache(side_path) or {}
    if side.get("_meta", {}).get("proj_seed", HID_PROJ_SEED) != HID_PROJ_SEED:
        raise RuntimeError(f"{side_path} was written with a different projection seed")

    warpers = build_warpers(temp, spec.warp_base["top_k"], spec.warp_base["top_p"])
    rep_pen = spec.warp_base.get("rep_penalty")
    allow_rt = bool(getattr(spec, "allow_roundtrip", False))

    problems = list(iter_problems(cache, spec.schema))
    if args.limit:
        problems = problems[:args.limit]

    gate = GateStats("token_entropies")
    arch_check = {"done": False}
    t0 = time.time()
    n_written = len([k for k in side if k != "_meta"])
    n_tokens = 0
    since_ckpt = 0
    skips = []

    def flush(done):
        side["_meta"] = {
            "version": SIDECAR_VERSION, "cell_id": spec.cell_id, "model": spec.model,
            "temp": temp, "n_layers": L, "hidden_size": hidden_size,
            "modules": list(MODULES), "quantities": list(QUANTITIES),
            "proj_seed": HID_PROJ_SEED, "proj_dim": args.proj_dim,
            "cov_eigs_r": args.cov_eigs_r, "dtype": "float16",
            "source_pkl": os.path.basename(pkl_path),
            "git_sha": _git_sha(), "job_id": os.environ.get("SLURM_JOB_ID", ""),
            "complete": bool(done),
            "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        save_cache_atomic(side, side_path)

    for pi, (idx, gold_row, question, cands) in enumerate(problems):
        todo = [(ci, c) for ci, c in enumerate(cands)
                if f"{idx}:{ci}" not in side
                and (get_aliased(c, "gen_token_ids") or allow_rt)]
        gating = gate.n_traces < args.gate_n
        if not todo and not gating:
            continue
        if STOP["flag"]:
            flush(False)
            print(f"[layers] PREEMPTED — checkpoint at problem {idx}", flush=True)
            return False, None
        try:
            prompt_ids = build_prompt_ids(tok, spec.prompt_recipe, gold_row, question,
                                          cands[0], idx=idx)
        except ValueError as e:
            skips.append({"idx": idx, "reason": f"prompt reconstruction: {e}"})
            continue

        for ci, c in todo:
            ids, source, _ = candidate_gen_ids(tok, c, allow_rt)
            if ids is None:
                skips.append({"idx": idx, "cand": ci, "reason": source})
                continue
            if args.max_gen_tokens and len(ids) > args.max_gen_tokens:
                ids = ids[:args.max_gen_tokens]
            item = {"prompt_ids": prompt_ids, "gen_ids": ids}
            plen, tgen = len(prompt_ids), len(ids)

            dev = mdl.device
            seq = torch.tensor([prompt_ids + ids], dtype=torch.long, device=dev)
            attn = torch.ones_like(seq)
            with ModuleTap(layers) as tap, torch.no_grad():
                out = mdl(input_ids=seq, attention_mask=attn, use_cache=False,
                          output_hidden_states=True)
                gen = torch.tensor(ids, dtype=torch.long, device=dev)

                # Architecture guard, once per pkl, on the live model+dtype. A failure
                # aborts THIS cell only — a family whose submodules do not decompose
                # the way MODULES assumes (OPT's fc1/fc2, any fused block) must not
                # take down the other cells sharing the job.
                if arch_check["done"] is False:
                    try:
                        arch_check.update(verify_residual_reconstruction(
                            mdl, tap, out.hidden_states, out.logits, tol=args.arch_tol))
                    except RuntimeError as e:
                        print(f"[layers] {spec.cell_id}: ARCHITECTURE GUARD FAIL — {e}",
                              flush=True)
                        return True, {"pkl": os.path.basename(pkl_path), "temp": temp,
                                      "arch_check_pass": False, "arch_error": str(e),
                                      "aborted": True, "gate_b_pass": None}
                    arch_check["done"] = True
                    print(f"[layers] {spec.cell_id}: architecture guard OK "
                          f"(residual identity {arch_check['residual_identity_max_abs']:.2e}, "
                          f"lens {arch_check['lens_max_abs']:.2e})", flush=True)

                # Gate B rides along on the final-layer logits of this same pass.
                if gate.n_traces < args.gate_n:
                    raw = out.logits[0, plen - 1: plen - 1 + tgen]
                    q = candidate_quantities(raw, gen, warpers, 1, 1,
                                             rep_penalty=rep_pen, prompt_ids=prompt_ids)
                    saved_h = (get_aliased(c, "token_entropies") or [])[:tgen]
                    gate.add(saved_h, q["token_entropies_recomputed"])
                    del raw

                if not args.validate_only:
                    field = candidate_layer_field(
                        mdl, tap, out.hidden_states, gen, plen, tgen, hid_proj,
                        cov_eigs_r=args.cov_eigs_r, batch_index=0)
                    field["n_gen_tokens"] = tgen
                    field["label"] = get_aliased(c, "label")
                    side[f"{idx}:{ci}"] = field
                    n_written += 1
                    n_tokens += tgen
                del out
            if args.gate_n and gate.n_traces == args.gate_n and not args.validate_only:
                ok, reasons = gate_b_verdict(gate.summary(), args.tol_median,
                                             args.tol_first, args.min_frac_close)
                if not ok:
                    # "nothing kept" has to be true on DISK, not just in memory: the
                    # gate verdict lands at gate_n candidates but --checkpoint-every
                    # may already have flushed a partial sidecar, and a gate-failed
                    # partial that survives is indistinguishable from a good one to
                    # everything downstream. Quarantine rather than delete so the
                    # evidence is still there to diagnose.
                    quarantined = None
                    if os.path.exists(side_path):
                        quarantined = side_path + ".GATE_B_FAILED_DO_NOT_USE"
                        os.replace(side_path, quarantined)
                    print(f"[layers] {spec.cell_id} T={temp}: GATE-B FAIL "
                          f"({'; '.join(reasons)}) — aborting cell"
                          + (f", partial sidecar quarantined -> "
                             f"{os.path.basename(quarantined)}" if quarantined
                             else ", nothing written"), flush=True)
                    return True, {"pkl": os.path.basename(pkl_path), "temp": temp,
                                  "gate": gate.summary(), "gate_b_pass": False,
                                  "gate_b_reasons": reasons, "aborted": True,
                                  "quarantined": quarantined}

        since_ckpt += 1
        if since_ckpt >= args.checkpoint_every:
            flush(False)
            since_ckpt = 0
            print(f"[layers] {spec.cell_id} T={temp}: checkpoint at problem {idx} "
                  f"({n_written} candidates, {n_tokens} tokens)", flush=True)
        if args.validate_only and gate.n_traces >= args.gate_n:
            break

    summ = gate.summary()
    ok, reasons = gate_b_verdict(summ, args.tol_median, args.tol_first,
                                 args.min_frac_close)
    print(f"[gate] {spec.cell_id} T={temp}: median|dH|="
          f"{summ.get('median_abs', float('nan')):.2e} "
          f"frac_close={summ.get('frac_close', float('nan')):.3f} "
          f"GATE-B {'PASS' if ok else 'FAIL'}", flush=True)

    report = {"pkl": os.path.basename(pkl_path), "temp": temp, "gate": summ,
              "gate_b_pass": ok, "gate_b_reasons": reasons,
              "validate_only": bool(args.validate_only),
              "n_candidates": n_written, "n_tokens": n_tokens,
              "n_layers": L, "arch_check": arch_check,
              "skips": skips[:50], "n_skips": len(skips),
              "seconds": round(time.time() - t0, 1)}
    if not args.validate_only:
        flush(True)
        report["sidecar"] = side_path
        report["sidecar_mb"] = round(os.path.getsize(side_path) / 1e6, 1)
        print(f"[layers] {spec.cell_id} T={temp} DONE: {n_written} candidates, "
              f"{n_tokens} tokens, {report['sidecar_mb']} MB -> {side_path}", flush=True)
    return True, report


def dry_run(spec, args):
    """No model load: count candidates and estimate the sidecar size."""
    n_cand = n_tok = 0
    for temp, pkl_path in spec.pkls:
        cache = load_cache(pkl_path)
        for idx, gold_row, question, cands in iter_problems(cache, spec.schema):
            for c in cands:
                if not get_aliased(c, "gen_token_ids"):
                    continue
                t = len(get_aliased(c, "gen_token_ids"))
                if args.max_gen_tokens:
                    t = min(t, args.max_gen_tokens)
                n_cand += 1
                n_tok += t
    # 4 quantities x 3 modules x L layers x T tokens, float16, L unknown -> assume 32
    mb = 4 * len(MODULES) * 32 * n_tok * 2 / 1e6
    print(f"  {spec.cell_id:34s} {n_cand:6d} candidates  {n_tok/1e3:8.1f}k tokens  "
          f"~{mb:7.1f} MB sidecar (at L=32)")
    return n_cand, n_tok


def main():
    ap = argparse.ArgumentParser(description="Per-layer logit-lens telemetry",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--cells", default=None, help="comma-separated cell ids")
    ap.add_argument("--data-root", default="/shared/cycle2_tau_averbuch_prj/omrisegev1")
    ap.add_argument("--validate-only", action="store_true",
                    help="run Gate B only, write no field")
    ap.add_argument("--limit", type=int, default=None, help="max problems per pkl")
    ap.add_argument("--max-gen-tokens", type=int, default=1024,
                    help="truncate very long generations (0 = no cap)")
    ap.add_argument("--checkpoint-every", type=int, default=25)
    ap.add_argument("--gate-n", type=int, default=50)
    ap.add_argument("--tol-median", type=float, default=2e-2)
    ap.add_argument("--tol-first", type=float, default=5e-2)
    ap.add_argument("--min-frac-close", type=float, default=0.90)
    ap.add_argument("--proj-dim", type=int, default=HID_PROJ_DIM)
    ap.add_argument("--cov-eigs-r", type=int, default=COV_EIGS_R)
    ap.add_argument("--attn", default="eager", choices=["sdpa", "eager"])
    ap.add_argument("--arch-tol", type=float, default=5e-2,
                    help="architecture guard: max residual-identity / relative lens "
                         "deviation before the run aborts (bf16 forward noise is well "
                         "inside this; a wrong tap is orders of magnitude outside)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for cid in list_backfill_cells():
            print(f"  {cid:36s} origin={BACKFILL_SPECS[cid]['origin']}")
        return
    if not args.cells:
        raise SystemExit("pass --cells id1,id2,... (or --list)")

    specs = [resolve_spec(c.strip(), args.data_root)
             for c in args.cells.split(",") if c.strip()]

    if args.dry_run:
        tot_c = tot_t = 0
        for s in specs:
            c, t = dry_run(s, args)
            tot_c += c
            tot_t += t
        print(f"\n  TOTAL {tot_c} candidates, {tot_t/1e6:.2f}M tokens")
        return

    if any("awq" in s.model.lower() or "gptq" in s.model.lower() for s in specs):
        _stub_pcre()
    signal.signal(signal.SIGTERM, _on_sigterm)
    if torch.cuda.is_available():
        print(f"[layers] GPU: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("[layers] WARNING: no CUDA — running on CPU", flush=True)

    specs.sort(key=lambda s: (s.model, s.dtype))
    current = None
    mdl = tok = None
    reports = []
    for spec in specs:
        if (spec.model, spec.dtype) != current:
            if mdl is not None:
                del mdl, tok
                free_memory()
            print(f"\n[layers] loading {spec.model} (attn={args.attn}, "
                  f"dtype={spec.dtype})", flush=True)
            mdl, tok = load_model(spec.model, attn_impl=args.attn, dtype=spec.dtype)
            current = (spec.model, spec.dtype)
        print(f"\n=== {spec.cell_id} ({len(spec.pkls)} pkl(s)) ===", flush=True)
        for temp, pkl_path in spec.pkls:
            done, rep = process_pkl(mdl, tok, spec, temp, pkl_path, args)
            if not done:
                print("[layers] INCOMPLETE — resubmit the same args to resume",
                      flush=True)
                sys.exit(EXIT_INCOMPLETE)
            rep["cell_id"] = spec.cell_id
            reports.append(rep)
        path = os.path.join(spec.data_dir, f"layer_views_report_{spec.cell_id}.json")
        with open(path + ".tmp", "w") as f:
            json.dump({"cell_id": spec.cell_id, "model": spec.model,
                       "version": SIDECAR_VERSION, "git_sha": _git_sha(),
                       "job_id": os.environ.get("SLURM_JOB_ID", ""),
                       "pkls": [r for r in reports if r["cell_id"] == spec.cell_id]},
                      f, indent=2, default=str)
        os.replace(path + ".tmp", path)
        print(f"[layers] report -> {path}", flush=True)

    arch = [r["cell_id"] for r in reports if r.get("arch_check_pass") is False]
    gate = [r["cell_id"] for r in reports if r.get("gate_b_pass") is False]
    ok = [r["cell_id"] for r in reports if r.get("gate_b_pass") is True]
    print(f"\n[layers] ALL CELLS PROCESSED — {len(ok)} extracted"
          + (f", GATE-B FAILED: {gate}" if gate else "")
          + (f", UNSUPPORTED ARCHITECTURE: {arch}" if arch else ""), flush=True)


if __name__ == "__main__":
    main()
