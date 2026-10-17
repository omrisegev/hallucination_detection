"""Independent bisect/order-count validation of saved-score diagnostics."""
from bisect import bisect_left, bisect_right
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def concordance(scores, labels):
    good = sorted(s for s, e in zip(scores, labels) if not e)
    bad = [s for s, e in zip(scores, labels) if e]
    if not good or not bad:
        return 0., 0
    # Every error is compared with sorted correct scores; equal values get half.
    wins = sum(bisect_left(good, s) + .5 * (bisect_right(good, s) - bisect_left(good, s)) for s in bad)
    return wins, len(good) * len(bad)


def main():
    package, output = map(Path, sys.argv[1:])
    expected = json.loads((output / "RESULTS.json").read_text())
    metadata = {r["idx"]: r for r in json.loads((package / "sources/prmscore_inputs/prmbench_meta.json").read_text())["rows"]}
    bank = np.load(package / "sources/prmscore_inputs/bank_N16.npz", allow_pickle=False)
    assert concordance([1, 1, 2], [True, False, False]) == (.5, 2)
    assert concordance([1, 1], [False, False]) == (0., 0)
    checks = {}
    for method, wanted in expected["summary"].items():
        observed = defaultdict(list)
        hits, step0_count = 0, 0
        error_count = 0
        for i, idx in enumerate(bank["idx"]):
            row = metadata[str(idx)]
            start, stop = bank["offsets"][i:i + 2]
            n = int(stop - start)
            v = list(range(n)) if method == "STEP_INDEX" else bank["S__" + method][start:stop].tolist()
            labels = [j + 1 in row["error_steps"] for j in range(n)]
            top = v.index(max(v))
            step0_count += int(top == 0)
            if any(labels):
                hits += int(labels[top])
                error_count += 1
            full_wins, full_pairs = concordance(v, labels)
            if full_pairs:
                observed["full"].append(full_wins / full_pairs)
            grouped = defaultdict(lambda: ([], []))
            for j, (score, label) in enumerate(zip(v, labels)):
                # Integer arithmetic independently implements the fixed 5 bins.
                group = (10 * j + 5) // (2 * n)
                grouped[group][0].append(score)
                grouped[group][1].append(label)
            wins, pairs = 0., 0
            for sub_scores, sub_labels in grouped.values():
                w, p = concordance(sub_scores, sub_labels)
                wins += w
                pairs += p
            if pairs:
                observed["matched"].append(wins / pairs)
                observed["wins"].append(wins)
                observed["pairs"].append(pairs)
        assert hits == wanted["top1_hits"]
        assert step0_count == wanted["n_top1_step0"]
        assert error_count == wanted["n_error_answers"]
        assert len(observed["full"]) == wanted["n_auc_eligible"]
        assert len(observed["matched"]) == wanted["n_same_bin_eligible"]
        assert sum(observed["pairs"]) == wanted["n_same_bin_pairs"]
        full_delta = abs(sum(observed["full"]) / len(observed["full"]) - wanted["within_auc"])
        matched_delta = abs(sum(observed["matched"]) / len(observed["matched"]) - wanted["same_bin_auc"])
        pooled_delta = abs(sum(observed["wins"]) / sum(observed["pairs"]) - wanted["same_bin_pair_pooled_auc"])
        assert max(full_delta, matched_delta, pooled_delta) < 1e-12
        checks[method] = dict(status="PASS", full_auc_abs_difference=full_delta,
                              same_bin_auc_abs_difference=matched_delta,
                              same_bin_pooled_auc_abs_difference=pooled_delta,
                              count_checks="top1 hits, step0 hits, eligibility, pairs all exact")
    report = dict(status="PASS", independent_arithmetic="Python bisect counts versus pairwise matrices/rank sums; integer position bins; list first-maximum versus numpy stable sort", 
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  results_sha256=hashlib.sha256((output / "RESULTS.json").read_bytes()).hexdigest(), checks=checks)
    (output / "INDEPENDENT_CHECK.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
