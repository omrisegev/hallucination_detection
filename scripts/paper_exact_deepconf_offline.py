#!/usr/bin/env python
"""
Offline DeepConf replay: every variant, table and online curve, from a saved acquisition.

Handoff §M2: "Perform all 64 fresh resampling repetitions and K/filter/budget variants
offline without more generation." This script is that step. It reads the shards written by
`cluster/run_paper_exact_deepconf.py` and produces:

  1. the equality audit against the pinned official confidence function (§M1 gate);
  2. the offline table — statistic x filter percentile x K, averaged over 64 fresh
     resamplings, with the cutoff recomputed inside each working set;
  3. the online table — warm-up N_init=16, percentile threshold, early termination replayed
     token by token against the saved confidence, consensus beta and budget stops;
  4. accuracy-versus-total-generated-tokens, the paper's own axis.

Why replay instead of generate: DeepConf's online rule only ever *truncates* a trace that
the offline pool already contains in full, so terminating at step t is exactly equivalent
to reading the first t tokens of a saved trace. That equivalence is what makes one 1.8B-token
pool serve every variant. It does NOT hold for REFRAIN or LEASH, which change what is
generated next — those need real branched generations, which is why S1/S2 run on GPU.

Usage:
    python scripts/paper_exact_deepconf_offline.py --run $SHARED/results/paper_exact/m2_full \
        --out results/paper_exact/m2_offline
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from spectral_utils.paper_exact import deepconf as DC          # noqa: E402
from spectral_utils.paper_exact import evaluator as EV         # noqa: E402
from spectral_utils.paper_exact.gates import Gate              # noqa: E402
from spectral_utils.paper_exact.manifest import load_manifest   # noqa: E402
from spectral_utils.paper_exact.shards import read_shards       # noqa: E402

#: Paper regression references for Qwen3-8B x AIME24 (Appendix Tables 5-10).
PAPER_REFERENCE = {
    "maj@512": {"accuracy": 80.0, "tokens": 2.32e8},
    "online_low": {"accuracy": 86.5, "tokens": 0.90e8, "token_change": -0.611},
    "online_high": {"accuracy": 80.4, "tokens": 1.33e8, "token_change": -0.428},
    "offline_top25": {"accuracy": 86.9},
    "Mean@10": {"accuracy": 86.7},
    "Tail(2k)@10": {"accuracy": 86.7},
    "L(2K)@10": {"accuracy": 86.7},
    "B(10%)@10": {"accuracy": 86.7},
}


def load_pool(run_dir: str, verify: bool = True) -> dict:
    """question_id -> list of trace dicts, each with confidence, answer, correctness, length."""
    pool = {}
    for rec in read_shards(run_dir, verify=verify):
        conf = np.asarray(rec["channels"]["deepconf_conf"], dtype=np.float64)
        pool.setdefault(rec["question_id"], []).append({
            "trace_key": rec["trace_key"],
            "conf": conf,
            "answer": rec.get("pred_answer"),
            "is_correct": bool(rec.get("correct", False)),
            "n_tokens": int(rec.get("n_tokens", len(conf))),
            "gold": rec.get("gold_answer"),
            "parse_status": rec.get("parse_status"),
            "retains_raw_top_k": rec.get("retains_raw_top_k", False),
            "raw_top_k_logprobs": rec.get("raw_top_k_logprobs"),
            "conf_variant": rec.get("conf_variant"),
            "conf_topk": rec.get("conf_topk", DC.DEFAULT_CONF_TOPK),
        })
    return pool


# ── 1. equality audit ───────────────────────────────────────────────────────────

def equality_audit(pool: dict, manifest: dict, gate: Gate) -> dict:
    """Recompute confidence from the retained raw top-50 and compare to what was stored.

    This is the §M1 gate. It proves that the scalar `deepconf_conf` channel — the only thing
    retained for most of the pool — is exactly what the pinned function computes from raw
    logprobs. Passing it is what licenses calling the pool's numbers DeepConf rather than a
    named proxy.
    """
    audited, results = 0, []
    for qid, traces in pool.items():
        for t in traces:
            if not t["retains_raw_top_k"] or t["raw_top_k_logprobs"] is None:
                continue
            lp = np.asarray(t["raw_top_k_logprobs"]["logprobs"], dtype=np.float64)
            recomputed = DC.trace_token_confidence(
                lp, variant=t["conf_variant"] or "paper_eq2",
                conf_topk=int(t["conf_topk"]), sampled_first=False)
            res = DC.equality_audit(t["conf"], recomputed,
                                    logits_stage=manifest.get("logits_stage", "unknown"))
            results.append({"trace_key": t["trace_key"], **res})
            audited += 1
    n_bad = sum(1 for r in results if not r["passed"])
    gate.check("deepconf_equality_audit", audited > 0 and n_bad == 0,
               f"{audited - n_bad}/{audited} audited traces reproduce the pinned function"
               if audited else "NO audit-sample traces retained raw top-50",
               detail=[r for r in results if not r["passed"]][:5])
    gate.check("logits_stage_is_raw", manifest.get("logits_stage") == "raw",
               f"logits_stage={manifest.get('logits_stage')!r}; anything but 'raw' means "
               f"every number here is a named proxy, not exact DeepConf")
    return {"n_audited": audited, "n_failed": n_bad,
            "max_abs_diff": max((r.get("max_abs_diff", 0.0) for r in results), default=0.0),
            "passed": audited > 0 and n_bad == 0}


# ── 2. offline table ────────────────────────────────────────────────────────────

def offline_table(pool: dict, ks, etas, statistics, n_runs: int = 64, seed: int = 42) -> list:
    """Statistic x eta x K, averaged over `n_runs` fresh resamplings per question.

    Voting is confidence-weighted (the paper's `V(a) = sum_t C_t I(answer=a)`); `eta=None`
    with `weighted=False` is plain majority, the cons@K baseline.
    """
    rows = []
    for stat in statistics:
        for eta in etas:
            for K in ks:
                accs, toks, nq = [], [], 0
                for qid, traces in sorted(pool.items()):
                    if len(traces) < 2:
                        continue
                    weighted = eta is not None or stat != "majority"
                    res = DC.offline_resample(
                        traces, K=K, n_runs=n_runs, eta=eta,
                        weighted=(stat != "majority"),
                        statistic=(stat if stat != "majority" else "mean"),
                        seed=seed + hash(qid) % 1000 if False else seed)
                    accs.append(res["accuracy"])
                    toks.append(res["tokens"])
                    nq += 1
                if not accs:
                    continue
                label = ("maj" if stat == "majority" else stat) + (f"@{eta}" if eta else "")
                rows.append({
                    "label": label, "statistic": stat, "eta": eta, "K": K,
                    # Accuracy is the mean over questions of the per-question vote accuracy —
                    # the paper's unit. Pooling traces across questions instead would weight
                    # long-trace questions more heavily.
                    "accuracy": float(np.mean(accs)) * 100.0,
                    "total_tokens": float(np.sum(toks)),
                    "n_questions": nq, "n_runs": n_runs,
                })
    return rows


# ── 3. online replay ────────────────────────────────────────────────────────────

def online_replay(pool: dict, eta: float, budgets, n_runs: int = 64,
                  n_init: int = DC.DEFAULT_N_INIT, window: int = DC.DEFAULT_GROUP_WINDOW,
                  beta: float = DC.DEFAULT_BETA, seed: int = 42) -> list:
    """Algorithm 2 replayed against saved traces.

    Warm up on `n_init` traces at full length, set the threshold from their lowest-group
    confidences, then stream the remaining traces: a trace whose running group confidence
    drops below the threshold is terminated at that token and contributes only the tokens
    it actually generated. Sampling stops at consensus beta or the budget.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for B in budgets:
        accs, toks, aborted = [], [], []
        for _ in range(int(n_runs)):
            run_acc, run_tok, run_abort = [], 0.0, 0
            for qid, traces in sorted(pool.items()):
                order = rng.permutation(len(traces))
                warm = [traces[i] for i in order[:n_init]]
                thr = DC.online_threshold(
                    [DC.lowest_group_conf(t["conf"], window) for t in warm], eta=eta)
                votes, gold_of = {}, {}
                for t in warm:
                    votes[t["answer"]] = votes.get(t["answer"], 0.0) + float(
                        DC.lowest_group_conf(t["conf"], window))
                    gold_of[t["answer"]] = t["is_correct"]
                    run_tok += t["n_tokens"]
                for i in order[n_init:B]:
                    t = traces[i]
                    stop_at = _first_termination(t["conf"], thr, window)
                    if stop_at is not None:
                        run_tok += stop_at
                        run_abort += 1
                        continue          # an aborted trace does not vote
                    run_tok += t["n_tokens"]
                    votes[t["answer"]] = votes.get(t["answer"], 0.0) + float(
                        DC.lowest_group_conf(t["conf"], window))
                    gold_of[t["answer"]] = t["is_correct"]
                    if DC.consensus_reached(votes, beta):
                        break
                best = max(votes, key=votes.get) if votes else None
                run_acc.append(1.0 if gold_of.get(best, False) else 0.0)
            accs.append(float(np.mean(run_acc)) if run_acc else float("nan"))
            toks.append(run_tok)
            aborted.append(run_abort)
        rows.append({
            "variant": f"online_eta{int(eta)}", "eta": eta, "budget": B,
            "accuracy": float(np.nanmean(accs)) * 100.0,
            "total_tokens": float(np.mean(toks)),
            "mean_aborted_traces": float(np.mean(aborted)),
            "n_runs": n_runs, "n_init": n_init, "beta": beta, "window": window,
        })
    return rows


def _first_termination(conf, threshold: float, window: int):
    """First token index at which the running group confidence falls below the threshold.

    Vectorised over the trace's group confidences: group i covers tokens [i, i+window), so
    the trace is killed at token i+window. Returns None if it never drops.
    """
    if not np.isfinite(threshold):
        return None
    g = DC.group_confidences(conf, window)
    if g.size == 0:
        return None
    below = np.flatnonzero(g < threshold)
    if below.size == 0:
        return None
    w = int(min(window, len(conf)))
    return int(min(len(conf), below[0] + w))


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--run", required=True, help="acquisition run directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ks", default="32,64,128,256,512")
    ap.add_argument("--etas", default="10,90")
    ap.add_argument("--statistics",
                    default="majority,mean,lowest_group_2k,bottom_10pct,tail_2k")
    ap.add_argument("--n-runs", type=int, default=64)
    ap.add_argument("--no-verify", action="store_true",
                    help="skip shard re-hashing (only for iterating on a verified run)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    man = load_manifest(args.run)
    gate = Gate("M-deepconf-offline", args.out)

    print(f"[offline] loading pool from {args.run} ...", flush=True)
    pool = load_pool(args.run, verify=not args.no_verify)
    sizes = {q: len(v) for q, v in pool.items()}
    print(f"[offline] {len(pool)} questions, "
          f"{sum(sizes.values())} traces, min/max per question "
          f"{min(sizes.values(), default=0)}/{max(sizes.values(), default=0)}", flush=True)
    gate.check("pool_non_empty", bool(pool), f"{len(pool)} questions")
    gate.check("pool_balanced", len(set(sizes.values())) <= 1 or
               (min(sizes.values()) / max(sizes.values()) > 0.95),
               f"per-question pool sizes {min(sizes.values())}..{max(sizes.values())}")

    cov = EV.parser_coverage([t["parse_status"] for ts in pool.values() for t in ts])
    gate.check("parser_coverage", cov >= 0.95, f"{cov:.4f} of traces produced a boxed answer")

    audit = equality_audit(pool, man, gate)

    ks = [int(x) for x in args.ks.split(",") if x]
    ks = [k for k in ks if k <= max(sizes.values(), default=0)]
    etas = [None] + [float(x) for x in args.etas.split(",") if x]
    stats = [s.strip() for s in args.statistics.split(",") if s.strip()]

    print("[offline] offline table ...", flush=True)
    off = offline_table(pool, ks, etas, stats, n_runs=args.n_runs)
    print("[offline] online replay ...", flush=True)
    on = []
    for eta in (DC.ETA_LOW, DC.ETA_HIGH):
        on += online_replay(pool, eta=eta,
                            budgets=[b for b in DC.BUDGETS if b <= max(sizes.values())],
                            n_runs=min(args.n_runs, 16))

    report = {
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_dir": args.run, "run_id": man.get("run_id"),
        "model": man.get("model_id"), "model_revision": man.get("model_revision"),
        "fidelity": man.get("fidelity"), "logits_stage": man.get("logits_stage"),
        "evaluator_revision": EV.EVALUATOR_REVISION,
        "n_questions": len(pool), "pool_sizes": sizes,
        "equality_audit": audit,
        "offline_table": off, "online_table": on,
        "paper_reference": PAPER_REFERENCE,
        "note": "paper_reference values are regression targets, not acceptance gates; a "
                "deviation is diagnosed for provenance, never tuned away.",
    }
    path = os.path.join(args.out, "DEEPCONF_OFFLINE.json")
    with open(path + ".tmp", "w") as f:
        json.dump(report, f, indent=2, default=float)
    os.replace(path + ".tmp", path)

    print(f"\n{'label':<20} {'K':>6} {'acc%':>8} {'tokens':>14}")
    for r in sorted(off, key=lambda r: (r["label"], r["K"])):
        print(f"{r['label']:<20} {r['K']:>6} {r['accuracy']:>8.1f} {r['total_tokens']:>14.3e}")
    print(f"\n{'online':<20} {'B':>6} {'acc%':>8} {'tokens':>14} {'aborted':>9}")
    for r in on:
        print(f"{r['variant']:<20} {r['budget']:>6} {r['accuracy']:>8.1f} "
              f"{r['total_tokens']:>14.3e} {r['mean_aborted_traces']:>9.1f}")
    print(f"\nreport -> {path}")
    gate.finish(raise_on_fail=False)


if __name__ == "__main__":
    main()
