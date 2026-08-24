#!/usr/bin/env python3
"""Restricted worker for target-free EDIS/AIME method fitting.

The audit hook is installed before importing any scientific project module.
Only the sanitized code capsule, fit-safe input capsule, runtime libraries, and
the worker output directory are reachable after that point.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import os
from pathlib import Path
import sys


POLICY_ENV = "RECONSTRUCTION_EDIS_FIT_POLICY_B64"
encoded_policy = os.environ.pop(POLICY_ENV, None)
if not encoded_policy:
    raise RuntimeError("EDIS fit worker requires a controller audit policy")
policy = json.loads(base64.b64decode(encoded_policy).decode("utf-8"))

REPO = Path(__file__).resolve().parents[2]
firewall_path = REPO / "spectral_utils/reconstruction_benchmark/fit_firewall.py"
firewall_spec = importlib.util.spec_from_file_location(
    "_reconstruction_edis_boot_firewall", firewall_path
)
if firewall_spec is None or firewall_spec.loader is None:
    raise RuntimeError("cannot load the EDIS boot firewall")
boot_firewall = importlib.util.module_from_spec(firewall_spec)
firewall_spec.loader.exec_module(boot_firewall)
policy_sha256 = boot_firewall.install_fit_audit_hook(policy)
denial_probes = boot_firewall.run_forbidden_read_probes(policy)
del policy, encoded_policy

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import argparse  # noqa: E402

from spectral_utils.reconstruction_benchmark.edis_fit import run_fit_worker  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--build", required=True, choices=("A", "B"))
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--fit-root", required=True, type=Path)
    parser.add_argument("--cell", action="append", dest="cells")
    parser.add_argument("--method", action="append", dest="methods")
    args = parser.parse_args()
    run_fit_worker(
        release_id=args.release_id,
        build_id=args.build,
        input_root=args.input_root,
        fit_root=args.fit_root,
        repo=REPO,
        audit_policy_sha256=policy_sha256,
        denial_probes=denial_probes,
        requested_cells=args.cells,
        requested_methods=args.methods,
    )
    violations = boot_firewall.fit_firewall_violations()
    if violations:
        raise RuntimeError(
            f"EDIS boot firewall recorded {len(violations)} sticky violation(s)"
        )


if __name__ == "__main__":
    main()
