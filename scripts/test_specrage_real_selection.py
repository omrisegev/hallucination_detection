#!/usr/bin/env python3
"""Known-answer test for v2 real-runner arm/config/lambda selection."""

import os
import sys


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from scripts.specrage_upcr_real import (  # noqa: E402
    CANDIDATE_ARMS,
    CONFIGS,
    LAMBDAS,
    select_one_standard_error,
)


def main():
    cells = ("gsm8k_mock", "triviaqa_mock", "coqa_mock")
    rows = []
    for config in CONFIGS:
        for cell_index, cell in enumerate(cells):
            for lambda_ in LAMBDAS:
                rows.extend((
                    {
                        "cell": cell,
                        "config": config,
                        "arm": "deployed_upcr",
                        "lambda": lambda_,
                        "auroc": 0.70,
                    },
                    {
                        "cell": cell,
                        "config": config,
                        "arm": "dufs_liu",
                        "lambda": lambda_,
                        "auroc": 0.705,
                    },
                ))
                for arm in CANDIDATE_ARMS:
                    # Embedding is the known winner; sample is a valid but
                    # weaker candidate. Small cell variation supplies nonzero SE.
                    gain = 0.020 if arm == "specrage_embedding" else 0.010
                    gain += 0.001 * cell_index
                    rows.append({
                        "cell": cell,
                        "config": config,
                        "arm": arm,
                        "lambda": lambda_,
                        "auroc": 0.70 + gain,
                        "algebra_valid": True,
                        "graph_collapsed": False,
                        "orientation_failure": False,
                    })
    chosen, candidates = select_one_standard_error(rows, cells)
    assert chosen is not None
    assert chosen["arm"] == "specrage_embedding"
    assert chosen["config"] == "agreement_k15"
    assert chosen["lambda"] == 0.0
    assert len(candidates) == len(CONFIGS) * len(CANDIDATE_ARMS) * len(LAMBDAS)
    print("SPECRAGE REAL SELECTION TEST PASS")


if __name__ == "__main__":
    main()
