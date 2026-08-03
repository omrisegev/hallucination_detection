#!/usr/bin/env python
"""
smoke_localization.py — known-answer gate for the Extension-F (step localization) modules.

Same role as `scripts/smoke_selectors.py` for the selector bench and `scripts/smoke_preset.py`
for cluster presets, and the same policy (Omri, 2026-07-17): **every new building block is
tested standalone on synthetic data with an obvious expected answer BEFORE it is integrated**
with the existing L-SML / U-PCR methods. CPU-only, seconds, exit non-zero on any failure.

Run:  python scripts/localization/smoke_localization.py
      python scripts/localization/smoke_localization.py evidence   # substring filter

Add a module here by giving it a module-level `smoke()` and listing it in `MODULES`.
"""
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

# (import path, human label). Import path is resolved relative to this directory or the repo.
MODULES = [
    ("evidence_drop", "Evidence Drop detector + the three paper baselines"),
    ("selective_metrics", "risk-coverage / AURC / selective accuracy / NP calibration"),
    ("spectral_utils.localization_data", "full-MATH loader helpers"),
]


def main() -> int:
    filt = sys.argv[1] if len(sys.argv) > 1 else ""
    failures = []
    ran = 0

    for mod_path, label in MODULES:
        if filt and filt not in mod_path and filt not in label:
            continue
        try:
            mod = __import__(mod_path, fromlist=["smoke"])
        except ImportError as e:
            failures.append((mod_path, f"import failed: {e}"))
            print(f"[FAIL] {mod_path:38s} import failed: {e}")
            continue
        fn = getattr(mod, "smoke", None)
        if fn is None:
            failures.append((mod_path, "no smoke() defined"))
            print(f"[FAIL] {mod_path:38s} no smoke() defined")
            continue
        ran += 1
        try:
            fn()
            print(f"[ok  ] {mod_path:38s} {label}")
        except Exception:
            failures.append((mod_path, traceback.format_exc()))
            print(f"[FAIL] {mod_path:38s} {label}")
            traceback.print_exc()

    print()
    if failures:
        print(f"SMOKE FAILED — {len(failures)} of {ran} module(s): "
              f"{', '.join(m for m, _ in failures)}")
        return 1
    print(f"ALL LOCALIZATION SMOKE TESTS PASS ({ran} modules)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
