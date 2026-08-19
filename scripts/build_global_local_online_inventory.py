#!/usr/bin/env python3
"""Build the frozen cache/evidence inventory for GLOBAL_LOCAL_ONLINE_IU_V1.

The script is deliberately read-only with respect to source caches.  It reads
small manifests and filesystem metadata only; it never unpickles a large cache
and never contacts or mutates Google Drive.  The Drive rows are a dated
snapshot of the read-only ``rclone lsf --format pst`` audit recorded in the
protocol cycle on 2026-08-16.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/global_local_online_iu_v1"


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _lfs_pointer(path: Path) -> bool:
    if not path.exists() or path.stat().st_size > 1024:
        return False
    try:
        return path.read_bytes().startswith(b"version https://git-lfs.github.com/spec/v1")
    except OSError:
        return False


def _first_artifact(directory: Path, names: list[str] | None = None) -> Path | None:
    candidates = []
    for pattern in names or ["*.pkl", "*.json"]:
        candidates.extend(sorted(directory.glob(pattern)))
    return candidates[0] if candidates else None


def _manifest_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest_path in sorted((ROOT / "dataset_cache/repgrid").glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        directory = manifest_path.parent
        model = manifest.get("model", "")
        dataset = manifest.get("dataset", "")
        protocol = manifest.get("protocol") or manifest.get("paper", "")
        cells = manifest.get("cells")
        if isinstance(cells, dict):
            # ProcessBench teacher-forced manifests use one mapping per subset.
            iterator = [
                {
                    "dataset": subset,
                    "n_problems": info.get("n_rows", ""),
                    "temp": 0.0,
                    "pkl": f"processbench_{subset}.pkl",
                    "mean_trace": info.get("mean_tokens", ""),
                }
                for subset, info in cells.items()
            ]
        elif isinstance(cells, list) and cells:
            iterator = cells
        else:
            artifact = _first_artifact(directory, ["*.pkl", "*.pkl.part-00"])
            iterator = [{
                "dataset": dataset,
                "n_problems": manifest.get("n_samples", ""),
                "temp": (manifest.get("temps") or [""])[0],
                "pkl": artifact.name if artifact else "",
                "mean_trace": "",
            }]
        for cell in iterator:
            artifact = directory / str(cell.get("pkl", ""))
            exists = artifact.exists()
            pointer = _lfs_pointer(artifact) if exists else False
            materialized = bool(exists and not pointer)
            is_pb = "processbench" in artifact.name or directory.name.startswith("pb_")
            is_rag = str(cell.get("dataset", dataset)).startswith("lciteeval_")
            capture = manifest.get("capture") or {}
            if materialized and is_pb:
                classification = "causal-prefix-valid"
                reason = "teacher-forced aligned token telemetry plus final-answer and first-error labels"
                role = "retrospective_localization_and_early"
            elif materialized and is_rag:
                classification = "causal-prefix-valid"
                reason = "aligned generated-token telemetry and answer label; RAG is outside current thesis scope"
                role = "inventory_only_out_of_scope"
            elif pointer:
                classification = "unusable"
                reason = "local artifact is a Git-LFS pointer; no download authorized"
                role = "remote_archive_only"
            elif not exists:
                classification = "unusable"
                reason = "artifact is not materialized locally"
                role = "manifest_only"
            elif capture.get("logsumexp"):
                classification = "causal-prefix-valid"
                reason = "manifest declares aligned token telemetry including logsumexp; row schema not reopened in this inventory"
                role = "retrospective_inventory"
            else:
                classification = "final-only"
                reason = "materialized artifact lacks a verified complete aligned prefix-telemetry contract"
                role = "retrospective_inventory"
            rows.append({
                "record_id": f"repgrid/{directory.name}/{cell.get('dataset', dataset)}@T{cell.get('temp', '')}",
                "source": "local_repgrid_manifest",
                "artifact": str(artifact.relative_to(ROOT)) if artifact.is_absolute() else str(artifact),
                "dataset_family": cell.get("dataset", dataset),
                "model_or_generator": model,
                "temperature": cell.get("temp", ""),
                "generation_protocol": protocol,
                "independent_questions": cell.get("n_problems", manifest.get("n_samples", "")),
                "k_traces_per_question": manifest.get("k", 1),
                "mean_tokens": cell.get("mean_trace", ""),
                "token_telemetry": "entropy+spilled+top-k" + ("+logsumexp" if capture.get("logsumexp") or is_pb else ""),
                "classification": classification,
                "classification_reason": reason,
                "materialized_local": materialized,
                "label_exposure": "opened development archive",
                "selection_confirmation_role": role,
            })
    return rows


def _localization_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((ROOT / "cache/localization/processbench").glob("pb_*/*.pkl")):
        model = path.parent.name.removeprefix("pb_")
        subset = path.stem.removeprefix("processbench_")
        manifest_path = ROOT / "dataset_cache/repgrid" / path.parent.name / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        info = (manifest.get("cells") or {}).get(subset, {})
        rows.append({
            "record_id": f"processbench/{model}/{subset}",
            "source": "local_materialized_localization_cache",
            "artifact": str(path.relative_to(ROOT)),
            "dataset_family": f"ProcessBench-{subset}",
            "model_or_generator": manifest.get("model", model),
            "temperature": 0.0,
            "generation_protocol": manifest.get("protocol", "teacher-forced fixed official trace"),
            "independent_questions": info.get("n_rows", {"gsm8k": 400}.get(subset, 1000)),
            "k_traces_per_question": 1,
            "mean_tokens": info.get("mean_tokens", ""),
            "token_telemetry": "entropy+spilled+logsumexp+top-k; aligned step spans; final-answer correctness",
            "classification": "causal-prefix-valid",
            "classification_reason": "prefix can be rebuilt from aligned telemetry; step spans are evaluation-only",
            "materialized_local": True,
            "label_exposure": "opened ProcessBench development",
            "selection_confirmation_role": "two qwen3_4b development cells; remaining cells retrospective non-selection/model transfer",
        })
    prm = ROOT / "dataset_cache/four_localization/prmbench_qwen3_8b_telemetry_full/prmbench_telemetry.pkl"
    rows.append({
        "record_id": "prmbench/qwen3_8b/all_classes",
        "source": "local_materialized_localization_cache",
        "artifact": str(prm.relative_to(ROOT)),
        "dataset_family": "PRMBench",
        "model_or_generator": "Qwen/Qwen3-8B",
        "temperature": 0.0,
        "generation_protocol": "teacher-forced fixed official trace; /no_think",
        "independent_questions": 6969,
        "k_traces_per_question": 1,
        "mean_tokens": 370.5259,
        "token_telemetry": "entropy+spilled+logsumexp+top-k; aligned step spans",
        "classification": "localization-only",
        "classification_reason": "step labels support every-step scoring; no final-answer correctness target for the early panel",
        "materialized_local": prm.exists() and not _lfs_pointer(prm),
        "label_exposure": "opened PRMBench development",
        "selection_confirmation_role": "retrospective separate step-level task",
    })
    return rows


def _drive_snapshot_rows() -> list[dict[str, Any]]:
    sizes = {
        0.3: 41140197,
        0.6: 39969066,
        1.0: 53155006,
        1.5: 100094204,
        2.0: 132120778,
    }
    rows = []
    for temp, size in sizes.items():
        rows.append({
            "record_id": f"drive/phase15/math500_qwen7b_T{temp}_run0",
            "source": "read_only_drive_snapshot_2026-08-16",
            "artifact": f"gdrive:hallucination_detection/cache/phase15_temperature/math500_qwen7b_T{temp}_run0.pkl",
            "dataset_family": "MATH-500",
            "model_or_generator": "Qwen/Qwen2.5-Math-7B-Instruct",
            "temperature": temp,
            "generation_protocol": "Phase-15 single-trace temperature sweep; max_new/prompt pinned in historical manifest",
            "independent_questions": 200,
            "k_traces_per_question": 1,
            "mean_tokens": "",
            "size_bytes": size,
            "token_telemetry": "entropy+spilled+top-k; no logsumexp (25/29 historical streams)",
            "classification": "causal-prefix-valid",
            "classification_reason": "valid for CUSUM/sw_var and missing-aware causal replay; not a complete IU28 feature contract",
            "materialized_local": temp == 1.0 and (ROOT / "local_cache/math500_qwen7b_T1.0_run0.pkl").exists(),
            "label_exposure": "opened Phase-15 development",
            "selection_confirmation_role": "temperature heterogeneity/inventory; T1.0 is in the 11-cell early panel",
        })
    # Four additional T=1.0 repeats quantify seed/repeat stability but do not
    # constitute independent dataset/model transfer.
    for run, size in enumerate([52556233, 50960644, 47199866, 50300904], start=1):
        rows.append({
            "record_id": f"drive/phase15/math500_qwen7b_T1.0_run{run}",
            "source": "read_only_drive_snapshot_2026-08-16",
            "artifact": f"gdrive:hallucination_detection/cache/phase15_temperature/math500_qwen7b_T1.0_run{run}.pkl",
            "dataset_family": "MATH-500",
            "model_or_generator": "Qwen/Qwen2.5-Math-7B-Instruct",
            "temperature": 1.0,
            "generation_protocol": "Phase-15 repeat generation",
            "independent_questions": 200,
            "k_traces_per_question": 1,
            "mean_tokens": "",
            "size_bytes": size,
            "token_telemetry": "entropy+spilled+top-k; no logsumexp",
            "classification": "causal-prefix-valid",
            "classification_reason": "valid remote causal telemetry, but large download was not authorized",
            "materialized_local": False,
            "label_exposure": "opened Phase-15 development",
            "selection_confirmation_role": "repeat/seed stability only; shared questions",
        })
    return rows


def _evidence_rows() -> list[dict[str, Any]]:
    return [
        {"experiment": "early_online_existing_data_v1", "data_family": "MATH500 + four ProcessBench families", "models_or_generators": "11 cells; multiple saved generators", "temperature_protocol": "Phase15 T1.0 generation + ProcessBench teacher-forced", "independent_unit": "question; five equal-weight dataset families", "method": "IU28/IU29 and entropy/DeepConf proxies", "primary_metric": "unfinished-prefix AUROC at absolute budgets", "cost": "CPU cache replay", "label_role": "calibration thresholds + evaluation only", "conclusion": "parity at 64/128; no promotion; declarations weak", "independence_note": "generator variants within a family are correlated"},
        {"experiment": "early_online_localization_models_v1", "data_family": "same 11 early cells", "models_or_generators": "same as prior screen", "temperature_protocol": "same", "independent_unit": "question; five families", "method": "causal GL-LIU/global/local/CUSUM/sw_var", "primary_metric": "unfinished-prefix AUROC and held-out declaration", "cost": "CPU cache replay", "label_role": "calibration thresholds + evaluation only", "conclusion": "localizer gives no early jump; CUSUM/sw_var dynamics remain hypothesis", "independence_note": "not independent confirmation of the first screen"},
        {"experiment": "ours_only_localization_v1", "data_family": "ProcessBench four subsets", "models_or_generators": "Qwen3-4B and Qwen3-8B telemetry scorers", "temperature_protocol": "teacher-forced fixed traces", "independent_unit": "original question; four dataset families", "method": "GL-LIU v1", "primary_metric": "ProcessBench F1", "cost": "one teacher-forced pass already cached", "label_role": "two development cells + split-local calibration", "conclusion": "31.36% F1 vs 25.71% Mind the Gap", "independence_note": "4B/8B reuse the same underlying questions"},
        {"experiment": "gl_liu_factorial_v2", "data_family": "same ProcessBench archive", "models_or_generators": "Qwen3-4B/8B", "temperature_protocol": "teacher-forced", "independent_unit": "question; four families", "method": "global/local ordinary, DUFS, temporal graph factorial", "primary_metric": "ProcessBench F1 and component AUROC", "cost": "CPU scoring on cached telemetry", "label_role": "opened retrospective diagnostic", "conclusion": "DUFS increments tiny; broad-28 local pool rejected; ordinary IU is simplicity baseline", "independence_note": "same questions as localization v1"},
        {"experiment": "fixed_application_pipelines_v1", "data_family": "ProcessBench + PRMBench", "models_or_generators": "Qwen3-4B/8B + Qwen3-8B PRMBench", "temperature_protocol": "teacher-forced fixed traces", "independent_unit": "question/response, never token", "method": "trajectory-first ordinary IU-PCR", "primary_metric": "ProcessBench F1; PRMBench step AUROC", "cost": "CPU cached fit/eval", "label_role": "calibration/evaluation only", "conclusion": "PB F1 0.3070; PRMBench AUROC 0.6711", "independence_note": "PRMBench is a separate task, not a confirmation macro"},
        {"experiment": "phase15_temperature", "data_family": "MATH-500", "models_or_generators": "Qwen2.5-Math-7B", "temperature_protocol": "T=0.3/0.6/1.0/1.5/2.0; four extra T1.0 repeats", "independent_unit": "question; repeats share dataset", "method": "historical spectral/temperature analysis", "primary_metric": "answer-error AUROC", "cost": "existing Drive caches; no download in this cycle", "label_role": "opened development", "conclusion": "sw_var is temperature-stable; orientation/fusion can vary with T", "independence_note": "temperature/repeat shifts are not new dataset confirmation"},
        {"experiment": "replication_grid", "data_family": "QA, math, GPQA, RAG", "models_or_generators": "broad multi-model grid", "temperature_protocol": "T=0/0.5/0.6/0.8/1.0/1.5 by native card", "independent_unit": "question within cell", "method": "L-SML/IU/feature audits", "primary_metric": "answer-error AUROC", "cost": "existing manifests; most local files are LFS pointers", "label_role": "opened development/benchmarking", "conclusion": "model/dataset/temperature heterogeneity is substantial", "independence_note": "many entries reuse datasets and are not localization data"},
    ]


def main() -> int:
    cache_rows = _manifest_rows() + _localization_rows() + _drive_snapshot_rows()
    cache_rows.sort(key=lambda row: str(row["record_id"]))
    evidence_rows = _evidence_rows()
    _write_csv(OUT / "CACHE_INVENTORY.csv", cache_rows)
    _write_csv(OUT / "EVIDENCE_INDEPENDENCE.csv", evidence_rows)
    counts: dict[str, int] = {}
    for row in cache_rows:
        key = str(row["classification"])
        counts[key] = counts.get(key, 0) + 1
    _write_json(OUT / "INVENTORY_SUMMARY.json", {
        "protocol": "GLOBAL_LOCAL_ONLINE_IU_V1",
        "generated_utc_date": "2026-08-16",
        "source_mutation": False,
        "drive_snapshot_command": "rclone lsf ... --format pst (read-only)",
        "drive_client_notice": "shared Google Drive client_id is scheduled for retirement during 2026",
        "n_cache_records": len(cache_rows),
        "classification_counts": counts,
        "n_evidence_records": len(evidence_rows),
    })
    print(json.dumps({"cache_records": len(cache_rows), "classification_counts": counts}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
