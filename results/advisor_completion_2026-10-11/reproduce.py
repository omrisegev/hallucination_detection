"""Local CPU replay, using saved inputs and the existing project virtualenv."""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import subprocess
from pathlib import Path

from package_manifest import build, verify

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
PYTHON = ROOT / ".venv/bin/python"


def inventory() -> list:
    data = json.loads((PACKAGE / "report_data.json").read_text())
    rows = data["task_status"]
    required = {"1", "2", "3", "4", "5", "5א", "6", "7", "8", "9", "9א", "9ב", "10", "11", "12", "13"}
    assert {r["id"] for r in rows} == required and len(rows) == len(required)
    for row in rows:
        for key in ["requirement", "previous_state", "action", "status", "finding", "evidence"]:
            assert row.get(key), (row["id"], key)
        for evidence in row["evidence"]:
            assert (PACKAGE / evidence).is_file(), (row["id"], evidence)
    gate = json.loads((PACKAGE / "POPULATION_GATE.json").read_text())
    assert gate["status"] == "PASS" and (gate["n_answers"], gate["n_steps"], gate["n_groups"]) == (6969, 94203, 707)
    for name in ["statistics", "validation"]:
        assert json.loads((PACKAGE / name / "STATUS.json").read_text())["status"] == "COMPLETE"
    assert json.loads((PACKAGE / "controls/STATUS_all.json").read_text())["status"] == "PASS"
    assert json.loads((PACKAGE / "lr/ACCEPTANCE.json").read_text())["status"] == "PASS"
    return rows


def write_inventory(rows: list) -> None:
    (PACKAGE / "TASK_STATUS.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    keys = ["id", "requirement", "previous_state", "action", "status", "finding", "evidence"]
    with (PACKAGE / "TASK_STATUS.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, keys)
        writer.writeheader()
        writer.writerows({**r, "evidence": " | ".join(r["evidence"])} for r in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--verify", action="store_true")
    group.add_argument("--report", action="store_true")
    group.add_argument("--all", action="store_true")
    args = parser.parse_args()
    # Report construction and checkpoint resume both consume saved outputs.
    # Reject any changed result before generating a new manifest.
    identities = verify(full=True)
    rows = inventory()
    if args.verify:
        print(json.dumps({**identities, "task_count": len(rows), "population_and_numerical_status": "PASS"}, indent=2))
        return
    steps = []
    if args.all:
        steps += [
            [PACKAGE / "controls/run_controls.py", "--stage", "all"],
            [PACKAGE / "controls/validate_controls.py"],
            [PACKAGE / "lr/reproduce.py"],
            [PACKAGE / "analysis/run_statistics.py"],
            [PACKAGE / "analysis/run_validation.py"],
        ]
    steps += [[PACKAGE / "report/build_v11.py", "--final"],
              [PACKAGE / "report/validate_html.py"]]
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logpath = PACKAGE / "logs" / f"reproduce_{stamp}.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(PACKAGE), str(ROOT), env.get("PYTHONPATH", "")])
    env.update({"OPENBLAS_NUM_THREADS": "2", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
    with logpath.open("w") as log:
        for step in steps:
            command = [str(PYTHON)] + [str(value) for value in step]
            print("Running", step[0].relative_to(PACKAGE), flush=True)
            log.write(json.dumps(command) + "\n")
            log.flush()
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    write_inventory(inventory())
    build()
    print("Reproduction PASS; log", logpath)


if __name__ == "__main__":
    main()
