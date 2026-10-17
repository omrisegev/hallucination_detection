"""Evaluate authenticated saved counts and build a standalone Hebrew report."""
import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "2")
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spectral_utils.error_count_evaluation import build_report, evaluate_package


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=ROOT/"results/label_free_error_count_v1_2026-10-12")
    parser.add_argument("--frozen", type=Path, default=ROOT/"results/advisor_completion_2026-10-11")
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if not args.report_only:
        status = evaluate_package(args.package, args.frozen, args.draws, args.seed)
        print("EVALUATION_COMPLETE", status["runtime_s"], flush=True)
    print("REPORT", build_report(args.package), flush=True)


if __name__ == "__main__":
    main()
