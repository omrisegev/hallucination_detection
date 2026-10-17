"""
GATE.json — machine-checkable stage gates.

Handoff §6. Each stage emits a GATE.json whose checks are pass/fail with a reason, and
promotion smoke -> pilot -> full depends **only** on these: schema, hashes, causality,
parser coverage, determinism, checkpoint/resume, and resource safety. It never depends on
whether a method wins. Published values are regression targets, not promotion gates
(handoff §1) — so `Gate` has no way to express "the number looked right", on purpose.
"""
import json
import os
from datetime import datetime, timezone


class Gate:
    """Collect named checks, then write GATE.json and (optionally) fail the job.

    Usage::

        g = Gate("L1-uprm-judge", run_dir)
        g.check("manifest", not problems, f"{len(problems)} manifest problems", problems)
        g.check("parser_coverage", cov >= 0.99, f"coverage={cov:.4f} (need >= 0.99)")
        g.finish(raise_on_fail=True)
    """

    def __init__(self, stage: str, run_dir: str):
        self.stage = stage
        self.run_dir = run_dir
        self.checks = []

    def check(self, name: str, passed: bool, reason: str = "", detail=None):
        self.checks.append({
            "name": name,
            "passed": bool(passed),
            "reason": reason,
            "detail": detail,
        })
        flag = "PASS" if passed else "FAIL"
        print(f"[gate:{self.stage}] {flag} {name}: {reason}", flush=True)
        return bool(passed)

    @property
    def passed(self) -> bool:
        return all(c["passed"] for c in self.checks)

    @property
    def failures(self) -> list:
        return [c["name"] for c in self.checks if not c["passed"]]

    def finish(self, raise_on_fail: bool = True) -> dict:
        gate = {
            "stage": self.stage,
            "run_dir": self.run_dir,
            "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
            "passed": self.passed,
            "n_checks": len(self.checks),
            "failures": self.failures,
            "checks": self.checks,
        }
        write_gate(gate, self.run_dir, self.stage)
        if raise_on_fail and not self.passed:
            raise SystemExit(
                f"[gate:{self.stage}] FAILED: {self.failures}. "
                f"Promotion is blocked. Fix the cause — never relax the gate to reach a number."
            )
        return gate


def write_gate(gate: dict, run_dir: str, stage: str = None) -> str:
    os.makedirs(run_dir, exist_ok=True)
    stage = stage or gate.get("stage", "stage")
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in stage)
    path = os.path.join(run_dir, f"GATE_{safe}.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(gate, f, indent=2, default=str)
    os.replace(tmp, path)
    return path


def write_blocked_assets(run_dir: str, stage: str, missing: list, evidence: dict) -> str:
    """Emit BLOCKED_ASSETS.json instead of a number.

    Handoff §P0.4 / W1: when an official asset is unavailable, the honest output is a
    precise `blocked-assets` row. Substituting a different dataset, labeller or prompt to
    fill the cell would produce a number that reads as a reproduction and is not one.
    """
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "BLOCKED_ASSETS.json")
    payload = {
        "stage": stage,
        "fidelity": "blocked-assets",
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "missing_assets": list(missing),
        "evidence": dict(evidence),
        "note": "No substitute corpus, labeller, prompt or checkpoint may be used to fill "
                "this row. Publish the blocked-assets row.",
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp, path)
    return path
