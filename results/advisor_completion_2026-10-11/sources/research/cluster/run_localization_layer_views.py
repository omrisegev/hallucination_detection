#!/usr/bin/env python
"""Per-layer white-box field for the LOCALIZATION rows (ProcessBench + PRMBench), teacher-forced.

Purpose (Omri, 2026-09-10): fuse the per-token entropy that now leads the answer-only
localization benchmark with the white-box per-layer views that were captured only for the
final-answer cells (``cluster/run_layer_views.py``). Nothing is generated: one forward pass
over each provided chain, exactly the pass ``run_teacher_forced.py`` / ``run_prmbench_teacher_forced.py``
already make, but with ``output_hidden_states=True`` and the ``layer_lens`` module taps, so
the per-layer logit-lens quantities are recorded per token.

DESIGN COMMITMENTS
1. Reuses ``build_items`` from the two teacher-forced drivers (same chat template, same
   chain construction, same step spans, same alignment gate) so the token axis is identical
   to the existing telemetry rows and joins by ``(cell, row id)``.
2. Writes one compressed npz per row into ``<out>/rows/`` (resumable, atomic rename), plus a
   manifest. Default is the FULL field: four lens quantities at all three taps
   (``--modules attn,mlp,resid``) for every layer, plus resid_norm, cov_eigs and hid_proj.
   Bytes (validated against the 14 existing sidecars to <0.1%):
       per token     = n_taps * 4 quantities * L * 2 B  +  resid_norm L * 2 B
       per candidate = cov_eigs L*r*4 B  +  hid_proj L*D*2 B
   At L=36, n_taps=3, r=32, D=256: 944 B/token + 23 KB/row, i.e. ~6.9 GB for all
   6,968,779 tokens / 13,769 rows across both models and both tasks. Disk is not the
   constraint; GPU time is — the lens runs norm+lm_head 3*L=108 times per token.
3. Gate: the final-layer lens entropy of every row must reproduce the saved
   ``token_entropies`` of the existing telemetry pkl for that row (median abs diff below
   ``--tol-median``); rows failing the gate are still written but flagged, and the run
   aborts if more than ``--max-gate-fail`` fraction fails in the first ``--gate-n`` rows.
4. SIGTERM -> finish the current batch, write, exit 85 (chain with afterany, see cluster/README.md).

Usage (on the cluster, inside the pytorch container):
    python cluster/run_localization_layer_views.py --task processbench --subset gsm8k \
        --model Qwen/Qwen3-8B --telemetry $R/pb_qwen3_8b/gsm8k.pkl --out $R/pb_layer_views_qwen3_8b/gsm8k
    python cluster/run_localization_layer_views.py --task prmbench --model Qwen/Qwen3-8B \
        --telemetry $R/prmbench_qwen3_8b_telemetry_full/prmbench_telemetry.pkl --out $R/prmbench_layer_views_qwen3_8b
    python cluster/run_localization_layer_views.py --smoke      # CPU, tiny random model, no download
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

STOP = {"flag": False}


def _on_sigterm(signum, frame):
    STOP["flag"] = True
    print("[layer-views-loc] SIGTERM received; will checkpoint after the current batch", flush=True)


def batched(items, max_batch_tokens, max_batch=4):
    order = sorted(items, key=lambda it: len(it["prompt_ids"]) + len(it["gen_ids"]))
    batch, longest = [], 0
    for it in order:
        n = len(it["prompt_ids"]) + len(it["gen_ids"])
        if batch and (max(longest, n) * (len(batch) + 1) > max_batch_tokens or len(batch) >= max_batch):
            yield batch
            batch, longest = [], 0
        batch.append(it)
        longest = max(longest, n)
    if batch:
        yield batch


def field_for_batch(mdl, batch, modules, quantities, hid_proj, cov_eigs_r):
    """Run one teacher-forced batch with taps and reduce each row to its per-layer field."""
    import torch
    from layer_lens import forward_batch_layers, candidate_layer_field, MODULES, QUANTITIES
    results = []
    for out, tap in forward_batch_layers(mdl, batch):
        for b, it in enumerate(batch):
            plen, tgen = len(it["prompt_ids"]), len(it["gen_ids"])
            field = candidate_layer_field(mdl, tap, out.hidden_states, it["gen_ids"], plen, tgen,
                                          hid_proj, cov_eigs_r=cov_eigs_r, batch_index=b)
            keep = {}
            for q in quantities:
                arr = field[q]                                   # [3 modules, L, T]
                keep[q] = arr[[MODULES.index(m) for m in modules]]
            keep["resid_norm"] = field["resid_norm"]
            # candidate_layer_field pays for the Gram eigendecomposition and the random
            # projection on every row whether or not they are kept; discarding them made
            # the geometry family (INSIDE / EigenScore / HaloScope) unreachable from this
            # output for no saving.  [L,r] + [L,D] per row is ~0.27 GB over 13,769 rows.
            keep["cov_eigs"] = field["cov_eigs"]
            keep["hid_proj"] = field["hid_proj"]
            # Gate input: the final-layer residual lens entropy in the cached top-15
            # renormalised form.  lens_H is FULL-vocabulary and is a different statistic —
            # comparing it to the saved token_entropies would fail the gate by construction.
            keep["final_lens_H"] = field["final_lens_H_top15"]
            keep["final_lens_H_fullvocab"] = field["lens_H"][MODULES.index("resid"), -1].astype(np.float32)
            results.append(keep)
        del out, tap
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return results


def load_rows_and_items(cfg, tok):
    if cfg.task == "processbench":
        from run_teacher_forced import build_items, load_processbench
        rows = load_processbench(cfg.subset, cfg.n_samples)
        items = build_items(tok, rows)
        ids = [r.get("id") for r in rows]
    else:
        from run_prmbench_teacher_forced import build_items
        from spectral_utils.prmbench import load_prmbench
        meta, _ = load_prmbench(cfg.n_samples)
        items = build_items(tok, meta)
        ids = [m["idx"] for m in meta]
    for it, rid in zip(items, ids):
        it["row_id"] = str(rid)
    return items


def saved_entropies(telemetry_path):
    if not telemetry_path:
        return {}
    from spectral_utils import load_cache
    cache = load_cache(telemetry_path)
    out = {}
    for k, v in cache.items():
        rid = v.get("id", v.get("idx", k))
        out[str(rid)] = np.asarray(v["token_entropies"], float)
    return out


def main():
    ap = argparse.ArgumentParser(description="Per-layer lens field for localization rows")
    ap.add_argument("--task", choices=["processbench", "prmbench"], default="processbench")
    ap.add_argument("--subset", default="gsm8k")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument("--telemetry", default=None, help="existing telemetry pkl for the same rows (gate)")
    ap.add_argument("--out", required=False)
    ap.add_argument("--modules", default="attn,mlp,resid")
    ap.add_argument("--quantities", default="lens_H,lens_logp_tgt,lens_logp_top1,lens_kl_final")
    ap.add_argument("--max-batch-tokens", type=int, default=6000)
    ap.add_argument("--max-batch", type=int, default=4)
    ap.add_argument("--gate-n", type=int, default=50)
    ap.add_argument("--tol-median", type=float, default=2e-2)
    ap.add_argument("--max-gate-fail", type=float, default=0.1)
    ap.add_argument("--cov-eigs-r", type=int, default=32)
    ap.add_argument("--proj-dim", type=int, default=256)
    ap.add_argument("--smoke", action="store_true", help="CPU tiny random model, two fake rows")
    cfg = ap.parse_args()
    signal.signal(signal.SIGTERM, _on_sigterm)
    import torch
    from layer_lens import make_hid_proj, MODULES, QUANTITIES
    modules = tuple(cfg.modules.split(",")); quantities = tuple(cfg.quantities.split(","))
    assert all(m in MODULES for m in modules) and all(q in QUANTITIES for q in quantities)

    if cfg.smoke:
        sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
        from smoke_layer_lens import tiny_model
        mdl = tiny_model()
        items = [{"idx": i, "row_id": f"smoke-{i}", "prompt_ids": list(range(5, 5 + 7 + i)),
                  "gen_ids": list(range(20, 20 + 30 + 5 * i)), "step_token_spans": [(0, 10), (10, 30 + 5 * i)]} for i in range(2)]
        hid_proj = make_hid_proj(mdl.config.hidden_size, mdl.device, dim=cfg.proj_dim)
        fields = field_for_batch(mdl, items, modules, quantities, hid_proj, cfg.cov_eigs_r)
        for it, f in zip(items, fields):
            shapes = {k: v.shape for k, v in f.items()}
            assert f["lens_H"].shape == (len(modules), len(mdl.model.layers), len(it["gen_ids"])), shapes
            assert all(np.isfinite(v.astype(np.float32)).all() for v in f.values()), "non-finite field"
            print("smoke row", it["row_id"], shapes)
        print("SMOKE OK"); return

    if not cfg.out:
        ap.error("--out is required unless --smoke is given")
    from spectral_utils import load_model
    os.makedirs(os.path.join(cfg.out, "rows"), exist_ok=True)
    mdl, tok = load_model(cfg.model, quantize_4bit=False)
    mdl.eval()
    items = load_rows_and_items(cfg, tok)
    saved = saved_entropies(cfg.telemetry)
    hid_proj = make_hid_proj(mdl.config.hidden_size, mdl.device, dim=cfg.proj_dim)
    done = {p[:-4] for p in os.listdir(os.path.join(cfg.out, "rows")) if p.endswith(".npz")}
    todo = [it for it in items if it["row_id"] not in done]
    print(f"[layer-views-loc] {len(items)} rows, {len(done)} done, {len(todo)} to go; modules={modules} quantities={quantities}", flush=True)
    gate_stats = {"checked": 0, "failed": 0}
    manifest = {"task": cfg.task, "subset": cfg.subset, "model": cfg.model, "modules": modules, "quantities": quantities,
                "started": datetime.now(timezone.utc).isoformat(), "n_rows": len(items),
                # provenance: these rows are meant to be frozen evidence, so the settings
                # that shape the field travel with them (run_layer_views.py does the same).
                "proj_dim": cfg.proj_dim, "cov_eigs_r": cfg.cov_eigs_r,
                "max_batch_tokens": cfg.max_batch_tokens, "max_batch": cfg.max_batch,
                "tol_median": cfg.tol_median, "telemetry": cfg.telemetry,
                "n_layers": int(len(mdl.model.layers)), "hidden_size": int(mdl.config.hidden_size),
                "job_id": os.environ.get("SLURM_JOB_ID"), "git_sha": os.environ.get("GIT_SHA")}
    n_done = 0; t_start = time.time()
    for batch in batched(todo, cfg.max_batch_tokens, cfg.max_batch):
        if STOP["flag"]:
            json.dump({**manifest, "gate": gate_stats, "status": "PREEMPTED", "rows_done": len(done)}, open(os.path.join(cfg.out, "MANIFEST.json"), "w"), indent=1)
            print("PREEMPTED; exiting 85", flush=True); sys.exit(85)
        t0 = time.time()
        fields = field_for_batch(mdl, batch, modules, quantities, hid_proj, cfg.cov_eigs_r)
        for it, f in zip(batch, fields):
            flag = None
            if it["row_id"] in saved:
                ref = saved[it["row_id"]]
                if len(ref) == len(f["final_lens_H"]):
                    med = float(np.median(np.abs(ref - f["final_lens_H"])))
                    gate_stats["checked"] += 1
                    if not med < cfg.tol_median:
                        gate_stats["failed"] += 1; flag = f"GATE_MEDIAN_ABS_DIFF={med:.4f}"
                else:
                    gate_stats["checked"] += 1; gate_stats["failed"] += 1; flag = "GATE_LENGTH_MISMATCH"
            path = os.path.join(cfg.out, "rows", it["row_id"] + ".npz"); tmp = path + ".tmp.npz"
            np.savez_compressed(tmp, step_starts=np.asarray([s[0] for s in it["step_token_spans"]]),
                                step_ends=np.asarray([s[1] for s in it["step_token_spans"]]),
                                gen_token_ids=np.asarray(it["gen_ids"], np.int32), gate_flag=np.asarray(flag or ""), **f)
            os.replace(tmp, path); done.add(it["row_id"]); n_done += 1
            # A gate that never fires is not a gate that passed.  The row-id join differs
            # between the two tasks (ProcessBench rows carry "id", PRMBench rows "idx"),
            # so a key mismatch would otherwise sail through as a vacuous success.
            if saved and n_done >= cfg.gate_n and gate_stats["checked"] == 0:
                json.dump({**manifest, "gate": gate_stats, "status": "GATE_VACUOUS"}, open(os.path.join(cfg.out, "MANIFEST.json"), "w"), indent=1)
                print(f"GATE VACUOUS: telemetry supplied but 0 of the first {n_done} rows joined by row id", flush=True)
                sys.exit(2)
            if gate_stats["checked"] == cfg.gate_n and gate_stats["failed"] > cfg.max_gate_fail * cfg.gate_n:
                json.dump({**manifest, "gate": gate_stats, "status": "GATE_FAILED"}, open(os.path.join(cfg.out, "MANIFEST.json"), "w"), indent=1)
                print(f"GATE FAILED: {gate_stats}", flush=True); sys.exit(2)
        print(f"[layer-views-loc] batch of {len(batch)} in {time.time()-t0:.1f}s ({len(done)}/{len(items)}), gate {gate_stats}", flush=True)
    json.dump({**manifest, "gate": gate_stats, "status": "COMPLETE", "rows_done": len(done), "seconds": time.time() - t_start},
              open(os.path.join(cfg.out, "MANIFEST.json"), "w"), indent=1)
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
