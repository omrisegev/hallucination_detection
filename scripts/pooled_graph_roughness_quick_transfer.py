#!/usr/bin/env python3
"""Quick retrospective transfer diagnostic for the pooled-roughness pilot."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import roc_auc_score


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.harp_global_contribution_teacher import (  # noqa: E402
    load_llama_processbench_cells,
    load_qwen_processbench_cells,
    load_semgrad_cells,
)


DIRECTION = np.asarray([
    0.0016923250008277738,
    0.00027956399005031607,
    -0.002730373984985923,
    0.00242714551451718,
    -0.0017197369807337092,
    0.004675242384337348,
])
TRUST_FACTOR = 0.5


def score(cell):
    presence = np.asarray(cell["presence"], dtype=bool)
    correction = np.asarray(cell["residuals"]) @ DIRECTION
    scale = float(np.std(correction))
    if scale <= 1e-12:
        return np.asarray(cell["baseline"])
    correction *= TRUST_FACTOR / (int(np.sum(presence)) * scale)
    return np.asarray(cell["baseline"]) + correction


def main():
    cells = (
        load_llama_processbench_cells()
        + load_qwen_processbench_cells()
        + load_semgrad_cells()
    )
    rows = []
    for cell in cells:
        y = cell["correctness"]
        baseline = roc_auc_score(y, cell["baseline"])
        candidate = roc_auc_score(y, score(cell))
        rows.append({
            "domain": cell["domain"],
            "group": cell["group"],
            "cell": cell["cell"],
            "delta_pp": 100 * (candidate - baseline),
        })
    summaries = {}
    for domain in sorted({row["domain"] for row in rows}):
        selected = [row for row in rows if row["domain"] == domain]
        groups = sorted({row["group"] for row in selected})
        values = [
            np.mean([
                row["delta_pp"] for row in selected if row["group"] == group
            ])
            for group in groups
        ]
        summaries[domain] = {
            "equal_group_delta_pp": float(np.mean(values)),
            "positive_groups": int(np.sum(np.asarray(values) > 0)),
            "worst_group_pp": float(np.min(values)),
            "group_values_pp": dict(zip(groups, map(float, values))),
        }
    print(json.dumps({"summaries": summaries, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
