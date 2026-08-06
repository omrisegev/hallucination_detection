#!/usr/bin/env python3
"""Dataset-free gates for the fixed confidence-orientation feature contract."""

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from spectral_utils.feature_contract import (                         # noqa: E402
    CONFIDENCE_FEATURE_SIGNS_V1,
    FIXED_SIGN_CHANGES_V1,
    FIXED_STABLE_EXCLUDED_V1,
    LEGACY_FEATURE_SIGNS,
    confidence_oriented_matrix,
    confidence_sign_vector,
    consensus_anchor,
)
from spectral_utils.feature_utils import FEAT_NAMES                    # noqa: E402
from spectral_utils.subset_sweep import (                             # noqa: E402
    ALL_SIGNS,
    CANONICAL_POOL,
    EXTRA_VIEWS,
    LEGACY_ALL_SIGNS,
    REPGRID_VIEWS,
)


FAILED = []


def check(name, condition, detail=""):
    state = "PASS" if condition else "FAIL"
    print(f"  [{state}] {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        FAILED.append(name)


def main():
    print("=" * 76)
    print("UNIT GATES — fixed confidence-orientation feature contract")
    print("=" * 76)

    registered = set(FEAT_NAMES) | set(REPGRID_VIEWS)
    check("all 30 in-scope raw features are registered",
          registered == set(CONFIDENCE_FEATURE_SIGNS_V1),
          f"registered={len(CONFIDENCE_FEATURE_SIGNS_V1)} expected={len(registered)}")
    check("all signs are exactly +/-1",
          set(CONFIDENCE_FEATURE_SIGNS_V1.values()) == {-1, 1})
    check("legacy mapping is explicit and complete",
          set(LEGACY_FEATURE_SIGNS) == registered)
    check("thirteen historical directions changed",
          len(FIXED_SIGN_CHANGES_V1) == 13,
          ", ".join(sorted(FIXED_SIGN_CHANGES_V1)))
    check("stable schema quarantines exactly the registered four",
          FIXED_STABLE_EXCLUDED_V1 == {
              "pe_mean", "stft_spectral_entropy", "cusum_shift_idx", "rpdi"
          })
    check("canonical fixed-sign map has no implicit fallback",
          set(CANONICAL_POOL) == set(ALL_SIGNS))
    check("every canonical raw-feature consumer uses the v1 direction",
          all(ALL_SIGNS[name] == sign
              for name, sign in CONFIDENCE_FEATURE_SIGNS_V1.items()))
    check("canonical legacy map remains reproducible",
          set(CANONICAL_POOL) == set(LEGACY_ALL_SIGNS))
    check("out-of-scope derived views retain their historical direction",
          all(ALL_SIGNS[name] == LEGACY_ALL_SIGNS[name] == -1
              for name in EXTRA_VIEWS))

    names = list(CONFIDENCE_FEATURE_SIGNS_V1)
    raw = np.tile(np.arange(len(names), dtype=float), (7, 1))
    raw += np.arange(7, dtype=float)[:, None]
    all_matrix, all_names, all_signs = confidence_oriented_matrix(raw, names)
    stable_matrix, stable_names, stable_signs = confidence_oriented_matrix(
        raw, names, stable=True,
    )
    check("all-schema matrix uses every feature",
          all_matrix.shape == raw.shape and all_names == names)
    check("all-schema multiplication is exact",
          np.array_equal(all_matrix, raw * confidence_sign_vector(names)))
    check("stable schema removes only quarantined features",
          stable_matrix.shape[1] == len(names) - 4
          and not (set(stable_names) & FIXED_STABLE_EXCLUDED_V1))
    check("stable signs align with stable names",
          np.array_equal(stable_signs, confidence_sign_vector(stable_names)))

    rng = np.random.default_rng(17)
    shared = rng.normal(size=200)
    aligned = np.column_stack([shared + rng.normal(scale=.2, size=200)
                               for _ in range(8)])
    anchor = consensus_anchor(aligned)
    check("consensus anchor follows aligned views",
          all(np.corrcoef(anchor, aligned[:, j])[0, 1] > 0
              for j in range(aligned.shape[1])))

    try:
        confidence_sign_vector(["not_registered"])
    except KeyError:
        unknown_rejected = True
    else:
        unknown_rejected = False
    check("unknown features fail closed", unknown_rejected)

    print("\n" + "=" * 76)
    if FAILED:
        print(f"{len(FAILED)} FAILED: " + ", ".join(FAILED))
        raise SystemExit(1)
    print("ALL PASSED")


if __name__ == "__main__":
    main()
