"""REWIND cut-point baseline + distance diagnostic on the Step-279 Localization lane.

CPU only.  Reuses the frozen fair-comparison lane end to end: the official 3,400
Llama-3.1-8B ProcessBench rows, the five localization folds, the threshold-only
cross-fit, and the paired grouped bootstrap.  Adds three REWIND rows (one per
cut-point stage, all with REWIND's best response-level detector, mean token
entropy) next to the frozen same-access methods, then measures how far each
method's locator lands from the gold first error on erroneous traces.

    .venv/bin/python scripts/run_rewind_cutpoint_processbench_v1.py [--bootstrap N]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tokenizers import Tokenizer  # noqa: E402

import scripts.build_fair_paper_exact_comparisons_v1 as fair  # noqa: E402
from spectral_utils.fair_comparisons.evaluator import (  # noqa: E402
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_BOOTSTRAP_SEED,
)
from spectral_utils.fair_comparisons.localization import (  # noqa: E402
    DEDICATED_METHOD_ID,
    GL_LIU_METHOD_ID,
    MAX_ENTROPY_METHOD_ID,
    MIND_GAP_METHOD_ID,
    UNIFIED28_METHOD_ID,
    bootstrap_localization_contrasts,
    crossfit_score_method,
    load_unified28_prethreshold_rows,
    native_mind_gap_metrics,
    replay_same_access_methods,
)
from spectral_utils.fair_comparisons.processbench import (  # noqa: E402
    PROCESSBENCH_SUBSETS,
    canonical_processbench_id,
)
from spectral_utils.rewind_cutpoint import rewind_row  # noqa: E402

TOKENIZER = (
    ROOT / "results/automatic_group_free_phase_a6_s0a_v1/inputs/llama31-8b-d4280566d175d597ec5d5294998b01f982c0729f56960601f889483643e1051f/tokenizer.json"
)
OUT = ROOT / "results" / "rewind_cutpoint_processbench_v1"
REWIND_IDS = {
    "aligned": "rewind_cutpoint",
    "gradient": "rewind_peak_gradient_noalign",
    "peak": "rewind_peak_only",
}
# Published Step-279 values; the replay must reproduce them before anything new is read.
EXPECTED_F1 = {UNIFIED28_METHOD_ID: 0.284832, DEDICATED_METHOD_ID: 0.326141}


def rewind_rows(pb, base_rows):
    """REWIND rows aligned 1:1 with an existing method's base fields."""

    tok = Tokenizer.from_file(str(TOKENIZER))
    raw = {}
    for subset, path in fair._pb_paths().items():
        import pickle

        with path.open("rb") as handle:
            cache = pickle.load(handle)
        for key, row in cache.items():
            raw[canonical_processbench_id(subset, str(row.get("id", key)))] = row
    out = {method: [] for method in REWIND_IDS.values()}
    diag = {}
    for base in base_rows:
        row = raw[base["row_id"]]
        result = rewind_row(row, tok.id_to_token)
        diag[base["row_id"]] = {**result, "n_steps": len(row["step_token_spans"])}
        for stage, method in REWIND_IDS.items():
            out[method].append(
                {
                    **{k: base[k] for k in (
                        "row_id", "group_id", "cell_id", "family", "subset", "first_error",
                        "stratify_label", "fold", "source_artifact_hash",
                    )},
                    "fit_hash": "rewind_no_fit",
                    "continuous_score": result["mean_entropy"],
                    "locator": int(result["cut_steps"][stage]),
                }
            )
    return out, diag


def distance_table(method_rows):
    """Locator error on erroneous traces only (detector-independent)."""

    table = []
    for method, rows in method_rows.items():
        err = [r for r in rows if int(r["first_error"]) != -1]
        d = np.asarray([int(r["locator"]) - int(r["first_error"]) for r in err])
        entry = {"method": method, "n_error_traces": len(err)}
        for subset in ("all",) + tuple(PROCESSBENCH_SUBSETS):
            sel = d if subset == "all" else np.asarray(
                [int(r["locator"]) - int(r["first_error"]) for r in err if r["family"] == subset]
            )
            entry[f"{subset}_exact"] = float(np.mean(sel == 0))
            if subset == "all":
                entry["within1"] = float(np.mean(np.abs(sel) <= 1))
                entry["early"] = float(np.mean(sel < 0))
                entry["late"] = float(np.mean(sel > 0))
                entry["median_abs"] = float(np.median(np.abs(sel)))
                entry["mean_signed"] = float(np.mean(sel))
        table.append(entry)
    return table


def positional_references(rows, diag):
    err = [r for r in rows if int(r["first_error"]) != -1]
    n = np.asarray([diag[r["row_id"]]["n_steps"] for r in err])
    gold = np.asarray([int(r["first_error"]) for r in err])
    return {
        "always_step0_exact": float(np.mean(gold == 0)),
        "uniform_random_exact": float(np.mean(1.0 / n)),
        "always_last_step_exact": float(np.mean(gold == n - 1)),
        "gold_relative_position_mean": float(np.mean(gold / np.maximum(n - 1, 1))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES)
    args = parser.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    pb = fair.build_pb_population_and_folds()
    population = fair._lane_population(pb["population"], fair.PB_LOCALIZATION_POPULATION)
    unified = load_unified28_prethreshold_rows(
        fair.UNIFIED_VALIDATION, pb["population"], folds=pb["localization_folds"]
    )
    replay = replay_same_access_methods(ROOT, pb["population"], folds=pb["localization_folds"])
    methods = {UNIFIED28_METHOD_ID: unified["rows"], **replay["methods"]}
    new_rows, diag = rewind_rows(pb, replay["methods"][MIND_GAP_METHOD_ID])
    methods.update(new_rows)
    # 2x2 decomposition: swap REWIND's detector and locator with the max-entropy row's.
    max_rows = {r["row_id"]: r for r in replay["methods"][MAX_ENTROPY_METHOD_ID]}
    methods["mean_entropy_step_top5mean"] = [
        {**r, "locator": max_rows[r["row_id"]]["locator"]} for r in new_rows["rewind_cutpoint"]
    ]
    methods["max_entropy_rewind_cut"] = [
        {**r, "continuous_score": max_rows[r["row_id"]]["continuous_score"]}
        for r in new_rows["rewind_cutpoint"]
    ]
    print(f"rows built in {time.time() - t0:.0f}s", flush=True)

    scored = {
        method: crossfit_score_method(rows, method_id=method, population_id=population.population_id)
        for method, rows in methods.items()
    }
    f1 = {m: s["metrics"] for m, s in scored.items()}
    for method, expected in EXPECTED_F1.items():
        got = float(f1[method]["equal_subset_macro_f1"])
        if abs(got - expected) > 5e-6:
            raise RuntimeError(f"{method}: replay F1 {got} != published {expected}")
    print("Step-279 F1 reproduced; scoring REWIND", flush=True)

    intervals = bootstrap_localization_contrasts(
        methods,
        contrasts=(
            ("rewind_cutpoint", MIND_GAP_METHOD_ID),
            ("rewind_cutpoint", UNIFIED28_METHOD_ID),
            (DEDICATED_METHOD_ID, "rewind_cutpoint"),
            (GL_LIU_METHOD_ID, "rewind_cutpoint"),
            ("mean_entropy_step_top5mean", "rewind_cutpoint"),
            ("rewind_cutpoint", "max_entropy_rewind_cut"),
            ("rewind_cutpoint", MAX_ENTROPY_METHOD_ID),
            ("rewind_cutpoint", "rewind_peak_only"),
        ),
        n_boot=args.bootstrap,
        seed=DEFAULT_BOOTSTRAP_SEED,
    )

    distances = distance_table(methods)
    stage_changes = {
        "peak_to_gradient_step_changed": float(np.mean(
            [d["cut_steps"]["peak"] != d["cut_steps"]["gradient"] for d in diag.values()])),
        "gradient_to_aligned_step_changed": float(np.mean(
            [d["cut_steps"]["gradient"] != d["cut_steps"]["aligned"] for d in diag.values()])),
        "aligned_found_delimiter": float(np.mean(
            [d["cut_tokens"]["aligned"] != d["cut_tokens"]["gradient"] for d in diag.values()])),
    }
    summary = {
        "schema": "rewind_cutpoint_processbench_v1",
        "population": population.population_id,
        "n_rows": len(population.ordered_ids),
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_seed": DEFAULT_BOOTSTRAP_SEED,
        "labels_used_in_score_or_locator": False,
        "deviations": [
            "cached top-K entropies instead of full-vocabulary entropies",
            "delimiters . , newline as substrings of byte-level BPE tokens",
            "cut token mapped to its ProcessBench step and read as the first error",
        ],
        "localization_f1": f1,
        "intervals": intervals,
        "native_sla": {m: native_mind_gap_metrics(rows) for m, rows in methods.items()},
        "positional_references": positional_references(methods["rewind_cutpoint"], diag),
        "rewind_stage_changes": stage_changes,
        "distance": distances,
        "runtime_s": time.time() - t0,
    }
    (OUT / "SUMMARY.json").write_text(json.dumps(summary, indent=2, default=float))
    with (OUT / "DISTANCE.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(distances[0]))
        writer.writeheader()
        writer.writerows(distances)
    with (OUT / "REWIND_ROWS.jsonl").open("w") as handle:
        for row_id, d in diag.items():
            handle.write(json.dumps({"row_id": row_id, **d}) + "\n")
    print(json.dumps({m: f1[m]["equal_subset_macro_f1"] for m in f1}, indent=1))
    print(f"done in {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
