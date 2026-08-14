#!/usr/bin/env python3
"""Restore or verify the authenticated tokenizer inputs required by A6-S0a."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.a6_tokenizer_restore import (  # noqa: E402
    load_and_verify_restore,
    restore_all_three,
)


DEFAULT_OUT = REPO / "results" / "automatic_group_free_phase_a6_tokenizers_v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("restore", "verify"))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--rclone", default="rclone")
    args = parser.parse_args()
    if args.command == "restore":
        value = restore_all_three(args.out, rclone=args.rclone)
    else:
        value = load_and_verify_restore(args.out)
    print(json.dumps({
        "status": value["status"],
        "materialized_sha256": value["materialized_sha256"],
        "out": str(Path(args.out).resolve()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
