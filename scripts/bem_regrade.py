#!/usr/bin/env python
"""
bem_regrade.py — offline BEM correctness regrade for a SemGrad-protocol pkl.

Mirrors run_inference.py's --regrade / run_judge_pass pattern, but for BEM instead of an
LLM judge: no generation, no target model — reads full_text + gold_row["truthful_answers"]
already saved in the cache. Runs LOCALLY, never on the AIRCC cluster (see
spectral_utils/bem_scorer.py's module docstring for why BEM never needs the B200/NGC path).

One-time setup before this will actually run: `pip install tensorflow kagglehub`, plus a
Kaggle account/API token for kagglehub to download the BEM checkpoint (see
data/semgrad_protocol/PROVENANCE.md).

Usage:
    python scripts/bem_regrade.py <raw_semgrad_sciq_T0.0.pkl> [<raw_semgrad_truthfulqa_T0.0.pkl> ...]
    python scripts/bem_regrade.py --threshold 0.8 results/semgrad_pilot/raw_semgrad_sciq_T0.0.pkl
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from spectral_utils import load_cache, save_cache_atomic
from spectral_utils.bem_scorer import bem_label_cache, THRESHOLD


def regrade_one(path: str, threshold: float) -> None:
    cache = load_cache(path)
    if not cache:
        print(f"[bem_regrade] {path}: empty/missing cache, skipping")
        return

    before = [c.get("label", False) for e in cache.values() for c in e["candidates"]]
    before_acc = sum(before) / max(len(before), 1)

    n = bem_label_cache(
        cache, threshold=threshold,
        checkpoint=lambda c=cache, p=path: save_cache_atomic(c, p),
        checkpoint_every=25,
        on_progress=lambda idx, n: (
            print(f"[bem_regrade] scored {n} candidates (problem {idx})", flush=True)
            if n % 50 == 0 else None),
    )
    save_cache_atomic(cache, path)

    after = [c.get("label", False) for e in cache.values() for c in e["candidates"]]
    after_acc = sum(after) / max(len(after), 1)
    pos, neg = sum(after), len(after) - sum(after)
    print(f"[bem_regrade] {os.path.basename(path)}: {n} candidates newly scored | "
          f"interim ROUGE-L accuracy {before_acc:.3f} -> BEM accuracy {after_acc:.3f} "
          f"(threshold={threshold}) | pos={pos} neg={neg}")
    if min(pos, neg) < 30:
        print(f"[bem_regrade] WARNING: minority class {min(pos, neg)} < 30 -- not yet "
              f"eligible for a trustworthy AUROC (see external_data_collection_plan_2026.md's "
              f"pilot health-check bar).")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("pkls", nargs="+", help="raw_semgrad_*.pkl path(s) to regrade in place")
    ap.add_argument("--threshold", type=float, default=THRESHOLD,
                    help="BEM positive-class probability threshold (default: official 0.8)")
    args = ap.parse_args()
    for path in args.pkls:
        regrade_one(path, args.threshold)


if __name__ == "__main__":
    main()
