#!/usr/bin/env python
"""
M2 balanced operational checkpoint — a continue-versus-repair decision, CPU only.

Codex review addendum §8: at a balanced point in the DeepConf acquisition, verify the run's
mechanics before spending the rest of the budget. The listed obligations, each implemented
below as a named check:

    expected/observed keys by question and shard   duplicates and missing IDs
    shard hash verification                        parser coverage
    telemetry finiteness and alignment             raw-logit audit coverage
    trace-length distribution                      throughput, storage growth, updated ETA
    an offline smoke reproduction (labelled partial)
    DeepConf scalar agreement

WHAT THIS IS NOT
----------------
Performance against the paper's published numbers is NOT a stop criterion and is not consulted
by the verdict. Accuracy appears in a clearly separated `context_only` block, because a
checkpoint that could halt a run for scoring below a published table would be selecting the
acquisition on its outcome. The verdict depends only on mechanics: hashes, key identity,
alignment, coverage, determinism and resource safety.

BALANCE
-------
`--min-per-question` (default 256) is the precondition Codex named. The M2 driver strides units
in question-major order, so at any interruption the pool holds COMPLETE data for the first k
questions and none for the rest — the maximally unbalanced shape. This script therefore
reports whether the precondition holds and refuses to call the pool balanced when it does not,
rather than reporting per-question statistics that are really statistics about one question.

Usage:
    python scripts/paper_exact_m2_checkpoint.py \
        --run  $SH/results/paper_exact/m2_deepconf_full \
        --out  $SH/results/paper_exact/m2_checkpoint
    # deeper telemetry sample (reads shard pickles; slower)
    python scripts/paper_exact_m2_checkpoint.py --run ... --out ... --sample-per-part 8
"""
import argparse
import json
import os
import pickle
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from spectral_utils.paper_exact import deepconf as DC       # noqa: E402
from spectral_utils.paper_exact import evaluator as EV      # noqa: E402
from spectral_utils.paper_exact.gates import Gate           # noqa: E402
from spectral_utils.paper_exact.shards import iter_run_dirs, verify_shards  # noqa: E402

#: Channels the DeepConf acquisition must carry for every token of every trace.
REQUIRED_CHANNELS = ("raw_entropy", "raw_logprob_sampled", "raw_pmax", "deepconf_conf")


# ── index-level accounting (cheap: no shard pickle is opened) ───────────────────

def read_index(path: str) -> list:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def account_keys(run: str) -> dict:
    """Expected-versus-observed key accounting, per question and per shard directory.

    Expected keys come from each worker's own RUN_MANIFEST (`expected_traces` plus the
    recorded shard/n_shards), never from a recomputation here: a checkpoint that rederived
    the expected set would agree with itself even if the driver's striding were wrong.
    """
    parts, observed, per_part, dup_within = {}, set(), {}, []
    for d in iter_run_dirs(run):
        name = os.path.basename(d)
        index = read_index(os.path.join(d, "INDEX.jsonl"))
        keys = [k for e in index for k in e["keys"]]
        seen = Counter(keys)
        dup_within += [f"{name}:{k}x{c}" for k, c in seen.items() if c > 1]
        man_path = os.path.join(d, "RUN_MANIFEST.json")
        man = {}
        if os.path.exists(man_path):
            with open(man_path) as f:
                man = json.load(f)
        per_part[name] = {
            "n_shards": len(index),
            "n_committed": len(keys),
            "n_unique": len(seen),
            "expected_traces": man.get("expected_traces"),
            "shard": (man.get("extra") or {}).get("shard", man.get("shard")),
            "n_shards_declared": (man.get("extra") or {}).get("n_shards", man.get("n_shards")),
            "bytes": sum(e.get("bytes", 0) for e in index),
            "last_written_utc": index[-1]["written_utc"] if index else None,
        }
        parts[name] = set(seen)
        observed |= set(seen)

    # Cross-worker collisions are the failure the part_NN layout exists to prevent. Checking
    # for them is how we know the layout is actually holding.
    cross = []
    names = sorted(parts)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            both = parts[a] & parts[b]
            if both:
                cross.append(f"{a}&{b}: {len(both)} shared keys e.g. {sorted(both)[:3]}")

    per_question = Counter(k.rsplit("#", 1)[0] for k in observed)
    return {"per_part": per_part, "per_question": dict(per_question),
            "n_observed": len(observed), "duplicates_within_part": dup_within,
            "duplicate_keys_across_parts": cross,
            "expected_total": sum(v["expected_traces"] or 0 for v in per_part.values())}


# ── record-level checks (opens a sample of shard pickles) ───────────────────────

def sample_records(run: str, per_part: int) -> list:
    """Take the newest `per_part` records from each worker, verifying each shard's hash first.

    Newest rather than oldest on purpose: a drift that appeared partway through the run is
    invisible in the first shard every worker wrote.
    """
    from spectral_utils.paper_exact.manifest import sha256_file
    out = []
    for d in iter_run_dirs(run):
        index = read_index(os.path.join(d, "INDEX.jsonl"))
        for entry in reversed(index):
            path = os.path.join(d, entry["path"])
            if not os.path.exists(path):
                continue
            if sha256_file(path) != entry["sha256"]:
                raise RuntimeError(f"sha256 mismatch reading sample from {path}")
            with open(path, "rb") as f:
                recs = pickle.load(f)
            out += [{**r, "_part": os.path.basename(d)} for r in recs[-per_part:]]
            break
    return out


def check_records(recs: list, gate: Gate, audit_every: int) -> dict:
    """Alignment, finiteness, parser coverage, audit coverage, DeepConf scalar agreement."""
    misaligned, nonfinite, lens, statuses, n_audit = [], [], [], Counter(), 0
    stat_max_abs, conf_max_abs = 0.0, 0.0
    missing_channels = Counter()

    for r in recs:
        key = r.get("trace_key", "?")
        ch = r.get("channels") or {}
        n = int(r.get("n_tokens") or 0)
        lens.append(n)
        statuses[r.get("parse_status", "missing")] += 1

        for name in REQUIRED_CHANNELS:
            if name not in ch:
                missing_channels[name] += 1
                continue
            v = np.asarray(ch[name], dtype=np.float64)
            # Every channel is one scalar per generated token. A channel one element short or
            # long means the prefix truncation the whole detection lane relies on would read a
            # different token's value than it thinks it is reading.
            if v.shape[0] != n:
                misaligned.append(f"{key}:{name} len={v.shape[0]} vs n_tokens={n}")
            if v.size and not np.all(np.isfinite(v)):
                nonfinite.append(f"{key}:{name} {int((~np.isfinite(v)).sum())} non-finite")

        gen = r.get("gen_token_ids")
        if gen is not None and len(gen) != n:
            misaligned.append(f"{key}:gen_token_ids len={len(gen)} vs n_tokens={n}")

        # Stored trace statistics are a convenience; the offline lane recomputes them. If the
        # two ever disagree, every downstream table silently depends on which one it read.
        conf = np.asarray(ch.get("deepconf_conf", []), dtype=np.float64)
        stored = r.get("trace_statistics") or {}
        if conf.size and stored:
            for name, fn in DC.TRACE_STATISTICS.items():
                if name in stored:
                    got, want = float(fn(conf)), float(stored[name])
                    if np.isfinite(got) and np.isfinite(want):
                        stat_max_abs = max(stat_max_abs, abs(got - want))

        # Audit traces retain the raw top-k arrays. Recomputing confidence from them is the
        # only check that the per-token channel really came from RAW logits, which is the
        # condition the DeepConf name is licensed on.
        if r.get("retains_raw_top_k") and "raw_top_k_logprobs" in r:
            n_audit += 1
            try:
                rc = DC.trace_token_confidence(
                    r["raw_top_k_logprobs"]["logprobs"] if isinstance(
                        r["raw_top_k_logprobs"], dict) else r["raw_top_k_logprobs"],
                    variant=r.get("conf_variant", "paper_eq2"),
                    conf_topk=int(r.get("conf_topk", DC.DEFAULT_CONF_TOPK)))
                m = min(len(rc), conf.size)
                if m:
                    conf_max_abs = max(conf_max_abs,
                                       float(np.max(np.abs(np.asarray(rc[:m]) - conf[:m]))))
            except Exception as e:  # noqa: BLE001
                misaligned.append(f"{key}:raw_audit_recompute_failed {e!r}"[:160])

    gate.check("channels_present", not missing_channels,
               "all required channels on every sampled trace" if not missing_channels
               else f"missing: {dict(missing_channels)}")
    gate.check("telemetry_alignment", not misaligned,
               f"{len(recs)} sampled traces aligned" if not misaligned
               else f"{len(misaligned)} misalignments e.g. {misaligned[:3]}")
    gate.check("telemetry_finite", not nonfinite,
               "all channel values finite" if not nonfinite
               else f"{len(nonfinite)} non-finite e.g. {nonfinite[:3]}")

    covered = sum(v for k, v in statuses.items() if k in ("boxed", "fallback_number"))
    cov = covered / max(1, len(recs))
    # A parser that silently fails is indistinguishable from a model that is wrong, so the
    # coverage floor is a mechanical gate. `boxed` versus `fallback_number` is reported
    # separately because a run leaning on the fallback is answering a different question.
    gate.check("parser_coverage", cov >= 0.90,
               f"{covered}/{len(recs)} parsed ({cov:.3f}); statuses {dict(statuses)}")

    gate.check("raw_logit_audit_coverage", n_audit >= 1,
               f"{n_audit} audit traces in the sample (stride --audit-every {audit_every})")
    gate.check("deepconf_stored_statistics_agree", stat_max_abs < 1e-9,
               f"max |recomputed - stored| over {len(DC.TRACE_STATISTICS)} statistics "
               f"= {stat_max_abs:.3e}")
    if n_audit:
        gate.check("deepconf_conf_matches_raw_logits", conf_max_abs < 1e-5,
                   f"max |conf(raw top-k) - stored channel| = {conf_max_abs:.3e}")

    lens = [x for x in lens if x]
    dist = {}
    if lens:
        q = np.percentile(lens, [5, 25, 50, 75, 95]).tolist()
        dist = {"n": len(lens), "min": min(lens), "max": max(lens),
                "mean": round(statistics.fmean(lens), 1),
                "p5": q[0], "p25": q[1], "median": q[2], "p75": q[3], "p95": q[4]}
    return {"parse_statuses": dict(statuses), "parser_coverage": cov,
            "n_audit_traces": n_audit, "trace_length_distribution": dist,
            "stored_statistic_max_abs_diff": stat_max_abs,
            "raw_logit_conf_max_abs_diff": conf_max_abs if n_audit else None,
            "stop_reasons": dict(Counter(r.get("stop_reason", "?") for r in recs))}


def throughput_and_eta(run: str, acct: dict) -> dict:
    """Per-worker tok/s, bytes per trace, projected storage and remaining wall."""
    rates, per_part = [], {}
    for d in iter_run_dirs(run):
        name = os.path.basename(d)
        tp_path = os.path.join(d, "THROUGHPUT.json")
        tp = {}
        if os.path.exists(tp_path):
            with open(tp_path) as f:
                tp = json.load(f)
        if tp.get("tokens_per_s"):
            rates.append(float(tp["tokens_per_s"]))
        per_part[name] = tp
    done = acct["n_observed"]
    expected = acct["expected_total"] or 0
    bytes_done = sum(v["bytes"] for v in acct["per_part"].values())
    per_trace = bytes_done / max(1, done)
    agg = sum(rates) if rates else None

    # ETA from realized bytes and the mean trace length implied by them, not from the plan's
    # estimate: the plan's estimate is what the M1 pilot already showed can be wrong by two
    # orders of magnitude.
    eta_h = None
    if agg and done:
        mean_tok = None
        for v in per_part.values():
            if v.get("mean_tokens_per_trace"):
                mean_tok = float(v["mean_tokens_per_trace"])
                break
        if mean_tok:
            eta_h = (expected - done) * mean_tok / agg / 3600.0
    return {
        "per_part_throughput": per_part,
        "n_workers_reporting": len(rates),
        "tokens_per_s_aggregate": round(agg, 1) if agg else None,
        "bytes_per_trace": int(per_trace),
        "bytes_committed": bytes_done,
        "projected_bytes_total": int(per_trace * expected) if expected else None,
        "fraction_complete": round(done / expected, 5) if expected else None,
        "eta_hours_remaining": round(eta_h, 1) if eta_h else None,
        "eta_note": "from realized aggregate tok/s and realized mean trace length; the "
                    "question-major stride means later questions may have different lengths, "
                    "so this is a projection, not a schedule.",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--run", required=True, help="M2 acquisition root (parent of part_NN)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-per-question", type=int, default=256,
                    help="Codex's balance precondition: completed traces per question")
    ap.add_argument("--sample-per-part", type=int, default=4,
                    help="records per worker for the deep telemetry checks")
    ap.add_argument("--audit-every", type=int, default=64)
    ap.add_argument("--skip-hash", action="store_true",
                    help="skip the full shard re-hash (fast; weakens the integrity check)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    gate = Gate("M2-operational-checkpoint", args.out)

    acct = account_keys(args.run)
    print(f"[m2ck] {acct['n_observed']} committed keys across "
          f"{len(acct['per_part'])} workers", flush=True)

    gate.check("workers_present", len(acct["per_part"]) > 0,
               f"{len(acct['per_part'])} worker directories with an INDEX.jsonl")
    gate.check("no_duplicate_keys_within_part", not acct["duplicates_within_part"],
               f"{len(acct['duplicates_within_part'])} duplicates"
               if acct["duplicates_within_part"] else "none")
    gate.check("no_duplicate_keys_across_parts", not acct["duplicate_keys_across_parts"],
               "; ".join(acct["duplicate_keys_across_parts"])[:300]
               if acct["duplicate_keys_across_parts"] else
               "part_NN isolation holding: no key written by two workers")

    over = {n: v for n, v in acct["per_part"].items()
            if v["expected_traces"] and v["n_unique"] > v["expected_traces"]}
    gate.check("no_worker_exceeds_its_expected_set", not over,
               f"workers over their expected count: {sorted(over)}" if over else
               "every worker within its declared expected_traces")

    counts = [v["n_unique"] for v in acct["per_part"].values()]
    spread = (max(counts) - min(counts)) if counts else 0
    # Workers are strided from one ordered unit list, so a large spread means one worker is
    # stalled or lost — worth seeing even though it is not by itself a repair trigger.
    gate.check("worker_progress_even", spread <= max(64, 0.25 * max(counts or [1])),
               f"committed per worker min={min(counts or [0])} max={max(counts or [0])} "
               f"spread={spread}")

    if not args.skip_hash:
        ver = verify_shards(args.run)
        gate.check("shard_hashes_verify", ver["ok"],
                   f"{ver['n_shards']} shards, {ver['n_traces']} traces, "
                   f"{ver['n_unique_keys']} unique keys"
                   + ("" if ver["ok"] else f"; problems {ver['problems'][:3]}"))
    else:
        ver = {"skipped": True}
        gate.check("shard_hashes_verify", False, "SKIPPED via --skip-hash — integrity unproven")

    # ── balance precondition ───────────────────────────────────────────────────
    pq = acct["per_question"]
    n_at = sum(1 for v in pq.values() if v >= args.min_per_question)
    balanced = bool(pq) and n_at == len(pq) and len(pq) > 1
    gate.check("balanced_checkpoint_precondition", balanced,
               f"{n_at}/{len(pq)} questions have >= {args.min_per_question} traces "
               f"(counts min={min(pq.values()) if pq else 0}, "
               f"max={max(pq.values()) if pq else 0}). "
               + ("balanced" if balanced else
                  "NOT balanced — the driver strides units in question-major order, so the "
                  "pool currently holds complete data for the leading questions and none for "
                  "the rest. Mechanical checks below are still valid; anything aggregated "
                  "across questions is not."))

    recs = sample_records(args.run, args.sample_per_part)
    print(f"[m2ck] deep-checking {len(recs)} sampled records", flush=True)
    rec_report = check_records(recs, gate, args.audit_every) if recs else {}
    gate.check("sample_nonempty", bool(recs), f"{len(recs)} records sampled")

    tp = throughput_and_eta(args.run, acct)

    # ── offline smoke reproduction, labelled partial ────────────────────────────
    # Codex asked for a reproduction "labelled partial". It is partial in a specific and
    # stated way: DeepConf's online rule needs n_init warmup traces plus a budget from
    # BUDGETS, and the committed pool is nowhere near either for most questions. Running it
    # anyway and printing a number would misrepresent a warmup as a result.
    smoke = {"label": "partial", "runnable": False, "reason": None}
    ready = [q for q, n in pq.items() if n >= max(DC.BUDGETS)]
    if ready:
        smoke.update(runnable=True, questions_ready=sorted(ready),
                     largest_runnable_budget=max(b for b in DC.BUDGETS
                                                 if b <= min(pq[q] for q in ready)),
                     note="Alg. 2 replay is runnable on these questions only; it is still "
                          "labelled partial because the pool is incomplete and the question "
                          "set is not the paper's 30.")
    else:
        smoke["reason"] = (f"no question yet holds {max(DC.BUDGETS)} traces "
                           f"(max is {max(pq.values()) if pq else 0}); every DeepConf budget "
                           f"in {DC.BUDGETS} exceeds the pool, so an Alg. 2 replay would "
                           f"report a warmup, not a result")
    gate.check("offline_smoke_reproduction_attempted", True, json.dumps(smoke)[:300])

    # ── context only: never consulted by the verdict ───────────────────────────
    graded = [r for r in recs if r.get("correct") is not None]
    context_only = {
        "_warning": "CONTEXT ONLY. Performance against the paper is NOT a stop criterion and "
                    "did not enter the verdict. This is a tiny newest-record sample from an "
                    "unbalanced pool and is not an accuracy estimate.",
        "n_graded_in_sample": len(graded),
        "n_correct_in_sample": int(sum(1 for r in graded if r["correct"])),
        "stop_reasons": rec_report.get("stop_reasons"),
    }

    g = gate.finish(raise_on_fail=False)

    # ── verdict: mechanics only ────────────────────────────────────────────────
    ADVISORY = {"balanced_checkpoint_precondition", "worker_progress_even",
                "raw_logit_audit_coverage", "offline_smoke_reproduction_attempted"}
    failures = [c["name"] for c in g["checks"] if not c["passed"]]
    blocking = [n for n in failures if n not in ADVISORY]
    verdict = "CONTINUE" if not blocking else "REPAIR"

    report = {
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run": args.run,
        "evaluator_revision": EV.EVALUATOR_REVISION,
        "verdict": verdict,
        "blocking_failures": blocking,
        "advisory_failures": [n for n in failures if n in ADVISORY],
        "verdict_basis": "mechanics only — hashes, key identity, alignment, parser coverage, "
                         "audit coverage, resource safety. Published-number comparison is "
                         "excluded by construction.",
        "key_accounting": acct,
        "shard_verification": ver,
        "record_checks": rec_report,
        "throughput_and_eta": tp,
        "offline_smoke": smoke,
        "context_only": context_only,
    }
    path = os.path.join(args.out, "M2_CHECKPOINT.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n[m2ck] verdict: {verdict}")
    if blocking:
        print(f"[m2ck] blocking: {', '.join(blocking)}")
    if report["advisory_failures"]:
        print(f"[m2ck] advisory (not blocking): {', '.join(report['advisory_failures'])}")
    print(f"[m2ck] progress {acct['n_observed']}/{acct['expected_total']} traces"
          f"  ETA {tp.get('eta_hours_remaining')} h"
          f"  projected {(tp.get('projected_bytes_total') or 0) / 1e9:.1f} GB")
    print(f"[m2ck] -> {path}")


if __name__ == "__main__":
    main()
