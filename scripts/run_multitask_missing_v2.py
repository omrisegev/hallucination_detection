#!/usr/bin/env python3
"""Run only the fixed development missing-channel audit for architecture v2."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.run_global_local_online_architecture_v2 as run  # noqa: E402
from scripts.run_multitask_sensitivity_v2 import _missing_sensitivity  # noqa: E402


def main() -> None:
    selection = json.load(open(run.OUT / "HEAD_SELECTION.json", encoding="utf-8"))["selected"]
    output = []
    for model_name, family in sorted(run.DEV_CELLS):
        rows = run.load_rows(run._cell_path(model_name, family))
        calibration, evaluation = run._split(rows)
        print(f"[missing] {model_name}/{family}", flush=True)
        models = run._fit_selected_cell(calibration, selection)
        output.extend(_missing_sensitivity(
            model_name, family, calibration, evaluation, models, selection
        ))
        run._write_csv(run.OUT / "MISSING_CHANNEL_SENSITIVITY.partial.csv", output)
    run._write_csv(run.OUT / "MISSING_CHANNEL_SENSITIVITY.csv", output)
    print(f"[done] wrote missing-channel audit to {run.OUT}", flush=True)


if __name__ == "__main__":
    main()
