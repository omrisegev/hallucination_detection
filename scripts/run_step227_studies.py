#!/usr/bin/env python3
"""Orchestrator for the Step-227 secondary studies.

The registered dependency-fusion sweep must finish before ANY of these run: a concurrent fit
competes for CPU threads and memory bandwidth and can affect the registered run's timing and
stability.  That was a review requirement for the DEEM probe, extended by Omri to all three
studies because the residual study is ~48,000 decomposition refits.

Order, and why:

    0. synthetic admission           must be a full, source-matched PASS
    1. wait for the sweep            PID gone AND records.jsonl quiet AND summary.json written
    2. smoke both spectral studies   2 cells, tiny B — catches wiring bugs before the long run
    3. solver mechanism study        seconds for the factorial, ~1-3 h for the held-out part
    4. residual identifiability      B=1000 x 2 nulls x 24 cells
    5. DEEM soft-collapse probe      needs the finished checkpoint to pick its pilot cells

Every step's stdout goes to results/_step227_logs/.  A failing step stops the chain: a broken
gate must not be papered over by the next study's output.

Usage:
    python scripts/run_step227_studies.py --pid 809940
    python scripts/run_step227_studies.py --pid 809940 --skip-wait      # sweep already done
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDY = os.path.join(REPO, "results", "dependency_fusion_study")
LOGS = os.path.join(REPO, "results", "_step227_logs")
QUIET_SECONDS = 900
SYNTHETIC_SCRIPT = os.path.join(REPO, "scripts", "synthetic_dependency_fusion_validation.py")
SYNTHETIC_SUMMARY = os.path.join(
    REPO, "results", "synthetic_dependency_fusion", "summary.json",
)


def synthetic_admission_status():
    """Require the locally established known-truth gate before real-data work.

    The result is source-bound: editing the benchmark after running it makes the
    old PASS stale.  There is intentionally no command-line bypass.  A failed
    synthetic mechanism is a scientific stop, not an operational inconvenience.
    """
    if not os.path.exists(SYNTHETIC_SUMMARY):
        return False, "synthetic summary missing; run the full synthetic benchmark first"
    try:
        with open(SYNTHETIC_SUMMARY, encoding="utf-8") as handle:
            summary = json.load(handle)
    except Exception as exc:
        return False, f"synthetic summary unreadable: {type(exc).__name__}: {exc}"
    try:
        with open(SYNTHETIC_SCRIPT, "rb") as handle:
            current_hash = hashlib.sha256(handle.read()).hexdigest()
    except Exception as exc:
        return False, f"synthetic script unreadable: {type(exc).__name__}: {exc}"
    if summary.get("script_sha256") != current_hash:
        return False, "synthetic result is stale: script SHA-256 does not match"
    admission = summary.get("admission", {})
    if not admission.get("eligible_full_run"):
        return False, "synthetic result came from a quick/non-eligible run"
    if not admission.get("admission_pass"):
        failed = [gate.get("gate") for gate in admission.get("gates", [])
                  if not gate.get("pass")]
        return False, "synthetic admission failed: " + ", ".join(failed)
    return True, "full source-matched synthetic admission PASS"


def process_status(pid):
    """Return ``(alive, command)`` without assuming a particular OS.

    A PID that exists but cannot be inspected is treated as alive.  That is the
    safe failure mode for a gate whose only job is to avoid competing with the
    registered sweep.
    """
    if pid is None:
        return False, ""
    pid = int(pid)
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False, ""
        except PermissionError:
            return True, "<alive; command unavailable>"
        try:
            command = subprocess.run(
                ["ps", "-p", str(pid), "-o", "command="],
                capture_output=True, text=True, timeout=60, check=False,
            ).stdout.strip()
        except Exception:
            command = "<alive; command unavailable>"
        return True, command

    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {int(pid)}"],
                             capture_output=True, text=True, timeout=60).stdout
        return str(pid) in out, out.strip()
    except Exception:
        # On Windows an unavailable process query must block, not silently
        # declare the sweep dead.
        return True, "<process status unavailable>"


def process_alive(pid):
    """Backward-compatible boolean wrapper used by lightweight tests."""
    return process_status(pid)[0]


def _summary_matches_checkpoint(summary_path, records_path):
    """Require a readable report generated after the final checkpoint write."""
    if not os.path.exists(summary_path):
        return False, "summary.json not written — the sweep never reached reporting"
    if os.path.getmtime(summary_path) < os.path.getmtime(records_path):
        return False, "summary.json is older than records.jsonl"
    try:
        with open(summary_path, encoding="utf-8") as handle:
            summary = json.load(handle)
    except Exception as exc:
        return False, f"summary.json is unreadable: {type(exc).__name__}: {exc}"
    if not isinstance(summary.get("arms"), list) or "config_hash" not in summary:
        return False, "summary.json is not a completed dependency-fusion report"
    return True, "summary is fresh and readable"


def sweep_done(pid):
    alive, command = process_status(pid)
    if alive:
        suffix = f" ({command})" if command else ""
        return False, f"PID {pid} still alive{suffix}"
    records = os.path.join(STUDY, "records.jsonl")
    if not os.path.exists(records):
        return False, "records.jsonl missing"
    quiet = time.time() - os.path.getmtime(records)
    if quiet < QUIET_SECONDS:
        return False, f"records.jsonl written {quiet:.0f}s ago (< {QUIET_SECONDS}s)"
    summary = os.path.join(STUDY, "summary.json")
    valid, why = _summary_matches_checkpoint(summary, records)
    if not valid:
        return False, why
    return True, f"PID gone, quiet {quiet:.0f}s, {why}"


def run(name, argv):
    os.makedirs(LOGS, exist_ok=True)
    log = os.path.join(LOGS, f"{name}.log")
    print(f"\n=== {name} ===\n  {' '.join(argv)}\n  -> {log}", flush=True)
    started = time.time()
    with open(log, "w", encoding="utf-8") as handle:
        proc = subprocess.run(argv, cwd=REPO, stdout=handle,
                              stderr=subprocess.STDOUT, text=True)
    took = time.time() - started
    print(f"  exit={proc.returncode} after {took / 60:.1f} min", flush=True)
    if proc.returncode != 0:
        with open(log, encoding="utf-8") as handle:
            tail = handle.read()[-3000:]
        print(f"--- tail of {log} ---\n{tail}", flush=True)
        raise SystemExit(f"{name} failed with exit {proc.returncode}; chain stopped")
    return log


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, default=None,
                        help="PID of the running registered sweep")
    parser.add_argument("--data-dir", default=os.path.join(REPO, "local_cache"))
    parser.add_argument("--skip-wait", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--device", default="auto",
                        help="DEEM device: auto, cpu, cuda, or mps")
    args = parser.parse_args()

    py = sys.executable
    admitted, admission_why = synthetic_admission_status()
    print(f"synthetic admission: {'PASS' if admitted else 'BLOCKED'} — {admission_why}",
          flush=True)
    if not admitted:
        raise SystemExit(
            "real-data Step-227 studies are blocked until a revised method passes "
            "scripts/synthetic_dependency_fusion_validation.py"
        )

    if args.skip_wait:
        done, why = sweep_done(args.pid)
        print(f"sweep completion check: {'PASS' if done else 'BLOCKED'} — {why}", flush=True)
        if not done:
            raise SystemExit("--skip-wait skips polling, not the safety gate")
    else:
        print(f"waiting for the registered sweep (PID {args.pid}) to exit; "
              f"polling every {args.poll_seconds}s", flush=True)
        while True:
            done, why = sweep_done(args.pid)
            print(f"  [{time.strftime('%H:%M:%S')}] {'READY' if done else 'waiting'} — {why}",
                  flush=True)
            if done:
                break
            time.sleep(args.poll_seconds)

    if not args.skip_smoke:
        run("smoke_solver", [py, "scripts/solver_mechanism_study.py",
                             "--data-dir", args.data_dir, "--smoke"])
        run("smoke_residual", [py, "scripts/residual_identifiability_study.py",
                               "--data-dir", args.data_dir, "--smoke"])

    run("solver_mechanism", [py, "scripts/solver_mechanism_study.py",
                             "--data-dir", args.data_dir])
    run("residual_identifiability", [py, "scripts/residual_identifiability_study.py",
                                     "--data-dir", args.data_dir])
    deem_argv = [py, "scripts/deem_soft_collapse_probe.py",
                 "--data-dir", args.data_dir, "--device", args.device]
    if args.pid is not None:
        deem_argv.extend(["--sweep-pid", str(args.pid)])
    run("deem_probe", deem_argv)
    print("\nall three studies finished; results under results/solver_mechanism, "
          "results/residual_identifiability, results/deem_probe", flush=True)


if __name__ == "__main__":
    main()
