"""Create an additive late integrity record after all structure/R1 work.

Never run against incomplete structure; never overwrite a record or scores.
Producer and audit source roots are separate because the repair is new code.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from spectral_utils.joint_lsml_integrity import create_late_record, verify_prelabel_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--producer-root", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    args = parser.parse_args()
    cells = [f"pb_{s}_{m}" for s in ("gsm8k", "math", "olympiadbench", "omnimath") for m in ("q4", "q8")]
    cells.append("prmbench_qwen3_8b")
    create_late_record(args.results_root, cells, producer_root=args.producer_root,
                       audit_root=REPO, amendment=args.amendment)
    result = verify_prelabel_record(args.results_root, cells)
    print(f"Verified {len(result['files'])} exact paths. Late provenance disclosed; no labels decoded.")


if __name__ == "__main__":
    main()
