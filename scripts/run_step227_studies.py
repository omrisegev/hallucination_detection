#!/usr/bin/env python3
"""Orchestrator for the Step-227 secondary studies.

The registered dependency-fusion sweep must finish before ANY of these run: a concurrent fit
competes for CPU threads and memory bandwidth and can affect the registered run's timing and
stability.  That was a review requirement for the DEEM probe, extended by Omri to all three
studies because the residual study is ~48,000 decomposition refits.

Order, and why:

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
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDY = os.path.join(REPO, "results", "dependency_fusion_study")
LOGS = os.path.join(REPO, "results", "_step227_logs")
QUIET_SECONDS = 900


def process_alive(pid):
    if pid is None:
        return False
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {int(pid)}"],
                             capture_output=True, text=True, timeout=60).stdout
        return str(int(pid)) in out
    except Exception:
        return False


def sweep_done(pid):
    if process_alive(pid):
        return False, f"PID {pid} still alive"
    records = os.path.join(STUDY, "records.jsonl")
    if not os.path.exists(records):
        return False, "records.jsonl missing"
    quiet = time.time() - os.path.getmtime(records)
    if quiet < QUIET_SECONDS:
        return False, f"records.jsonl written {quiet:.0f}s ago (< {QUIET_SECONDS}s)"
    if not os.path.exists(os.path.join(STUDY, "summary.json")):
        return False, "summary.json not written — the sweep never reached its reporting stage"
    return True, f"PID gone, quiet {quiet:.0f}s, summary.json present"


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
    args = parser.parse_args()

    py = sys.executable
    if not args.skip_wait:
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
    run("deem_probe", [py, "scripts/deem_soft_collapse_probe.py",
                       "--data-dir", args.data_dir])
    print("\nall three studies finished; results under results/solver_mechanism, "
          "results/residual_identifiability, results/deem_probe", flush=True)


if __name__ == "__main__":
    main()
