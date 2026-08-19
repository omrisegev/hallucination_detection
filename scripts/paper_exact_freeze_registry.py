#!/usr/bin/env python
"""
Freeze the prefix-detection claim registry, and exercise its structural gates.

Codex review addendum §8 (Q5): freeze the registry and "emit a frozen JSON rendering plus
hashes before opening comparison results". This script does exactly that and nothing else —
it computes no comparison, touches no evaluation label, and reads no acquisition outcome.

It also runs every structural gate that is checkable without labels, on synthetic input, so
the gates are demonstrably executable before the lane runs. Gates that genuinely require the
real fit are listed as NOT ESTABLISHED rather than skipped, because a skipped check reads
like a pass.

Usage:
    python scripts/paper_exact_freeze_registry.py --out results/paper_exact/prefix_lane
    python scripts/paper_exact_freeze_registry.py --verify-only
"""
import argparse
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from spectral_utils.paper_exact import claim_registry as CR   # noqa: E402
from spectral_utils.paper_exact import evaluator as EV        # noqa: E402
from spectral_utils.paper_exact.gates import Gate             # noqa: E402
from spectral_utils.paper_exact.telemetry import causal_prefix_channels  # noqa: E402


def _demo_score(channels, t):
    """A stand-in causal score for the structural gates.

    Deliberately uses a prefix-CENTRED cumulative sum — a statistic that *would* leak if it
    were computed on the completed trace and sliced. Rebuilding it from `row[:t]` is what
    makes suffix invariance hold, so the gate is testing something real.
    """
    ch = causal_prefix_channels(channels, t)
    h = np.asarray(ch["raw_entropy"], dtype=float)
    if h.size == 0:
        return 0.0
    cs = np.cumsum(h - h.mean())
    return float(np.max(np.abs(cs)) / h.size)


def run_structural_gates(gate: Gate, registry: dict) -> dict:
    rng = np.random.default_rng(7)
    budgets = registry["budgets"]["primary_absolute_tokens"]

    # suffix invariance: shared 512-token prefix, wildly different suffixes
    base = list(rng.normal(2.0, 0.5, 512))
    a = {"raw_entropy": base + list(rng.normal(6.0, 0.4, 200))}
    b = {"raw_entropy": base + list(rng.normal(-4.0, 3.0, 900))}
    ok, why = CR.gate_suffix_invariance(_demo_score, a, b, budgets)
    gate.check("suffix_invariance", ok, why)

    ok, why = CR.gate_tokenwise_vs_chunked(_demo_score, a, 256)
    gate.check("tokenwise_vs_chunked_replay", ok, why)

    # label permutation on a genuinely informative score
    y = (rng.random(400) < 0.4).astype(float)
    s = y * 1.2 + rng.normal(0, 1.0, 400)
    ok, why = CR.gate_label_permutation(y, s, EV.auroc)
    gate.check("label_permutation", ok, why)

    # feature-order invariance: an order-symmetric reducer must not care about stream order
    mat = rng.normal(size=(64, 12))
    names = [f"s{i}" for i in range(12)]
    ok, why = CR.gate_feature_order(
        lambda m, nm: np.max(np.abs(m - m.mean(axis=0)), axis=1), mat, names)
    gate.check("feature_order_perturbation", ok, why)

    ok, why = CR.gate_split_isolation(range(0, 40), range(40, 60), range(60, 100))
    gate.check("split_isolation", ok, why)

    # alarm horizon: calibrate on one half of the correct traces, verify on the other
    paths = [rng.normal(0, 1, 40) for _ in range(400)]
    cal = EV.calibrate_ever_alarm_threshold(paths[:200], target_fpr=0.05)
    ok, why = CR.gate_alarm_horizon(cal["threshold"], paths[200:], 0.05)
    gate.check("alarm_horizon_calibration", ok, why)

    groups = {q: list(rng.normal(rng.normal(0, 1), 0.05, 10)) for q in range(60)}
    ok, why = CR.gate_grouped_resampling(
        groups,
        lambda vals: float(np.mean(np.concatenate([np.asarray(v) for v in vals]))),
        lambda g, fn: EV.grouped_bootstrap(g, fn, n_boot=500))
    gate.check("grouped_resampling_validity", ok, why)

    lengths = rng.integers(50, 4000, 400).astype(float)
    ok, why = CR.gate_length_only_leakage(lengths, y, s, EV.auroc)
    gate.check("length_only_leakage", ok, why)

    for name, why in CR.DEFERRED_GATES.items():
        # Recorded as an explicit failure, not an omission: these must be established during
        # the lane run, and a silently missing check would read as a pass.
        gate.check(name, False, f"NOT ESTABLISHED — {why}")
    return {"deferred": sorted(CR.DEFERRED_GATES)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--registry", default=CR.REGISTRY_PATH)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "paper_exact",
                                                  "prefix_lane"))
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    registry = CR.load_registry(args.registry)
    problems = CR.verify_registry(registry)
    print(f"registry : {args.registry}")
    print(f"version  : {registry.get('registry_version')}")
    print(f"hash     : {CR.registry_hash(registry)}")
    print(f"nulls    : {len(registry.get('nulls') or {})} / {len(CR.CANONICAL_NULLS)} canonical")
    print(f"gates    : {len(registry.get('structural_gates') or {})} declared")
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("registry is internally consistent\n")

    if args.verify_only:
        return

    os.makedirs(args.out, exist_ok=True)
    gate = Gate("prefix-lane-structural-gates", args.out)
    run_structural_gates(gate, registry)
    # raise_on_fail=False: the two deferred gates fail by construction here. They are the
    # lane run's obligation, and this script's job is to record that obligation, not to
    # pretend it is discharged.
    g = gate.finish(raise_on_fail=False)

    frozen = CR.freeze(registry, args.out)
    print(f"\nfrozen   -> {frozen['path']}")
    print(f"hash     -> {frozen['registry_hash']}")
    established = [c["name"] for c in g["checks"] if c["passed"]]
    print(f"\nstructural gates established now ({len(established)}): {', '.join(established)}")
    print(f"still owed by the lane run: {', '.join(sorted(CR.DEFERRED_GATES))}")
    print("\nCite this hash in every prefix-lane result file. An edited registry changes the "
          "hash, which is the audit trail.")


if __name__ == "__main__":
    main()
