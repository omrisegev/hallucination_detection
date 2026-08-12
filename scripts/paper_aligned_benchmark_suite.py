#!/usr/bin/env python3
"""Build the paper-aligned hallucination benchmark report suite.

The suite is intentionally protocol-first.  It never averages answer detection,
RAG spans, claims, reasoning steps, or first-error localization.  Dataset
adapters may change the feature *unit*, but all local feature matrices are sent
to the same frozen U-PCR -> IU-PCR -> DUFS-LIU-PCR solver progression.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.inscope_cells import CLUSTER_CELLS, GROUP, INSCOPE  # noqa: E402
from scripts.rag_ec_v1.gasp import (  # noqa: E402
    GASP_FEATURE_NAMES, gasp_threshold_scores,
)
from spectral_utils.paper_benchmark_suite import (  # noqa: E402
    ProtocolSignature, apply_spectral_model, assert_protocol_match,
    bar_chart, binary_metrics, esc, fit_spectral_model, fit_spectral_scores,
    forbid_cross_task_macro, grouped_bootstrap_binary, read_csv, score_hash,
    write_csv, write_json,
)
from spectral_utils.token_feature_views import CORE_TOKEN_VIEWS, token_feature_views  # noqa: E402


VERSION = "paper-aligned-benchmark-suite-v1-2026-08-11"
DEFAULT_OUT = ROOT / "results" / "paper_aligned_benchmark_suite_2026_08_11"
BAD_PRM_IDS = {
    "confidence_confidence_prm_train_p1_303",
    "deception_deception_prm_test_p1_87",
    "step_contradiction_step_contradiction_prm_test_p2_991",
}
METHOD_LABELS = {
    "deployed_upcr": "Deployed U-PCR",
    "iu_pcr": "IU-PCR",
    "dufs_liu_pcr": "DUFS-LIU-PCR",
    "gasp": "GASP-threshold",
    "lettucedetect": "LettuceDetect",
    "qwen_prm": "Qwen2.5-Math-PRM-7B",
    "mind_the_gap": "Mind the Gap",
    "qwen3_judge": "Qwen3-8B judge control",
    "qwen72b_critic": "Qwen2.5-72B critic",
    "gl_liu_v1": "GL-LIU v1 (frozen)",
    "max_entropy": "Maximum token entropy",
    "refchecker_nli": "RefChecker NLI checker",
}


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _trace_summary(row: Mapping[str, Any]) -> tuple[list[float], list[str]]:
    """Small risk-oriented contract available in every localization cache."""

    entropy = np.asarray(row.get("token_entropies", []), dtype=float)
    spilled = np.asarray(row.get("token_spilled_energies", []), dtype=float)
    top = row.get("top_k_logprobs") or {}
    logprobs = np.asarray(top.get("logprobs", []), dtype=float)
    if entropy.size == 0:
        raise ValueError("empty token trace")
    if spilled.size != entropy.size:
        spilled = np.resize(spilled, entropy.size)
    if logprobs.ndim == 2 and len(logprobs):
        width = min(len(logprobs), len(entropy))
        top1_risk = -float(np.mean(logprobs[:width, 0]))
        margin_risk = -float(np.mean(logprobs[:width, 0] - logprobs[:width, 1]))
        probs = np.exp(logprobs[:width] - np.max(logprobs[:width], axis=1, keepdims=True))
        probs /= np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
        top_entropy = float(np.mean(-np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=1)))
    else:
        top1_risk = margin_risk = top_entropy = math.nan
    return [
        float(np.mean(entropy)), float(np.std(entropy)), float(np.quantile(entropy, 0.9)),
        float(np.mean(spilled)), float(np.std(spilled)),
        top1_risk, margin_risk, top_entropy,
    ], [
        "mean_entropy", "entropy_std", "entropy_q90", "mean_target_nll", "target_nll_std",
        "negative_mean_top1_logprob", "negative_top1_margin", "top50_entropy",
    ]


def _method_rows(
    *, protocol_id: str, signature: ProtocolSignature, labels: np.ndarray,
    groups: list[str], score_map: Mapping[str, np.ndarray], role: str = "ours",
    method_roles: Mapping[str, str] | None = None,
    subgroup: str = "all", draws: int = 400,
) -> list[dict[str, Any]]:
    intervals = grouped_bootstrap_binary(labels, score_map, groups, draws=draws)
    rows = []
    for method, score in score_map.items():
        metrics = binary_metrics(labels, score)
        for metric, value in metrics.items():
            low, high = intervals[method][metric]
            rows.append({
                "protocol_id": protocol_id, "dataset": signature.dataset,
                "model": signature.model, "split": signature.split,
                "prediction_unit": signature.prediction_unit, "grader": signature.grader,
                "subgroup": subgroup, "method_key": method,
                "method": METHOD_LABELS.get(method, method),
                "role": (method_roles or {}).get(method, role),
                "metric": metric, "value": value, "ci_low": low, "ci_high": high,
                "n": len(labels), "positive_rate": float(np.mean(labels)),
                "status": "local", "fidelity": "task adapter",
            })
    return rows


def build_detection_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load the frozen 24-cell numbers; no detector is recomputed here."""

    stable = read_csv(ROOT / "results/fixed_orientation_validation/per_cell.csv")
    full = read_csv(ROOT / "results/hard_filter_dufs_liu_24cell/per_cell_metrics.csv")
    competitors = read_csv(ROOT / "results/advisor_inscope/competitors_verified.csv")
    upcr = {
        row["cell"]: row for row in stable
        if row.get("method") == "upcr" and row.get("arm") == "fixed_stable_v1"
    }
    iu = {
        row["cell"]: row for row in full
        if row.get("contract") == "mixed_v2" and row.get("filter") == "full"
        and row.get("solver") == "iu_pcr"
    }
    dufs = {
        row["cell"]: row for row in full
        if row.get("contract") == "mixed_v2" and row.get("filter") == "full"
        and row.get("solver") == "dufs_liu"
    }
    rows = []
    for cell in INSCOPE:
        protocol_id = "internal-transfer" if cell in CLUSTER_CELLS else ""
        reference = iu.get(cell) or dufs.get(cell)
        for key, source in (("deployed_upcr", upcr), ("iu_pcr", iu), ("dufs_liu_pcr", dufs)):
            if cell not in source:
                continue
            record = source[cell]
            rows.append({
                "protocol_id": protocol_id, "cell": cell, "subgroup": cell,
                "dataset": cell, "model": "frozen cell model", "split": "frozen evaluation",
                "prediction_unit": "answer", "grader": "cell-specific frozen grader",
                "method_key": key, "method": METHOD_LABELS[key], "role": "ours",
                "metric": "auroc", "value": _float(record.get("auroc")),
                "ci_low": "", "ci_high": "", "n": (reference or {}).get("n", ""),
                "positive_rate": (reference or {}).get("positive_rate", ""),
                "status": "local", "fidelity": "frozen retrospective development",
                "feature_contract": "fixed_stable_v1" if key == "deployed_upcr" else "mixed-v2 full pool",
                "caveat": "Retrospective development evidence; not independent confirmation.",
            })
    # A cell can support more than one paper page.  Keep one primary page, then
    # cross-link the same local scores to every additional verified reference.
    cell_papers: dict[str, list[str]] = defaultdict(list)
    for row in competitors:
        if row.get("cell") not in CLUSTER_CELLS and row.get("paper_slug"):
            if row["paper_slug"] not in cell_papers[row["cell"]]:
                cell_papers[row["cell"]].append(row["paper_slug"])
    for row in rows:
        if row["protocol_id"] != "internal-transfer":
            row["protocol_id"] = "detection-" + (cell_papers.get(row["cell"]) or ["unverified"])[0]
    extra_local = []
    for row in rows:
        if row.get("role") != "ours":
            continue
        for slug in cell_papers.get(row.get("cell", ""), [])[1:]:
            extra_local.append({**row, "protocol_id": "detection-" + slug})
    rows.extend(extra_local)

    # INSIDE/CoQA is outside the frozen 24-cell full-pool suite, but an older
    # same-protocol U-PCR result exists.  Include it explicitly as a legacy
    # compatibility row so the documented loss is visible instead of absent.
    repgrid = read_csv(ROOT / "results/repgrid/headline_X_vs_Y.csv")
    for record in repgrid:
        if record.get("cell") == "inside_coqa_llama7b" and record.get("method") == "upcr":
            rows.append({
                "protocol_id": "detection-inside-llms-internal-states-retain-the",
                "cell": record["cell"], "subgroup": record["cell"],
                "dataset": record.get("dataset", ""), "model": record.get("model", ""),
                "split": "legacy frozen evaluation", "prediction_unit": "answer",
                "grader": "legacy cell-specific grader", "method_key": "legacy_upcr",
                "method": "Legacy U-PCR compatibility arm", "role": "ours_reference",
                "metric": "auroc", "value": _float(record.get("X")),
                "ci_low": "", "ci_high": "", "n": record.get("n", ""),
                "positive_rate": "", "status": "legacy local reference",
                "fidelity": "repgrid compatibility arm", "feature_contract": record.get("best_subset", ""),
                "caveat": "Outside the 24-cell full-pool suite; shown to retain the documented loss, not as a core-method result.",
            })

    seen_published = set()
    for row in competitors:
        if row.get("cell") not in cell_papers or not row.get("auroc"):
            continue
        identity = (row.get("paper_slug"), row.get("cell"), row.get("method"), row.get("auroc"))
        if identity in seen_published:
            continue
        seen_published.add(identity)
        rows.append({
            "protocol_id": "detection-" + row["paper_slug"], "cell": row["cell"],
            "subgroup": row["cell"], "dataset": row.get("dataset", ""),
            "model": row.get("model", ""), "split": "paper table",
            "prediction_unit": "answer", "grader": "paper-specific",
            "method_key": "published:" + row.get("method", ""), "method": row.get("method", ""),
            "role": "published_ceiling" if row.get("supervision") == "supervised" else "published_peer",
            "metric": "auroc", "value": _float(row["auroc"]), "ci_low": "", "ci_high": "",
            "n": "", "positive_rate": "", "status": "published reference",
            "fidelity": "published paper table", "paper_table": row.get("paper_table", ""),
            "paper_evidence": row.get("paper_evidence", ""),
            "supervision": row.get("supervision", ""), "access": row.get("access", ""),
            "passes": row.get("passes", ""), "caveat": row.get("caveat", ""),
        })
    return rows, competitors


def _fit_token_contract(rows: list[Mapping[str, Any]], max_fit_tokens: int = 60_000):
    per_row, fit_blocks, total = [], [], 0
    rng = np.random.default_rng(0)
    for row in rows:
        views = token_feature_views(dict(row))
        matrix = np.column_stack([views[name] for name in CORE_TOKEN_VIEWS])
        per_row.append(matrix)
    for index in rng.permutation(len(per_row)):
        block = per_row[int(index)]
        take = min(len(block), max_fit_tokens - total)
        if take:
            fit_blocks.append(block[:take]); total += take
        if total >= max_fit_tokens:
            break
    model, diagnostics = fit_spectral_model(
        np.vstack(fit_blocks), feature_names=CORE_TOKEN_VIEWS,
        risk_anchor=np.vstack(fit_blocks)[:, 0],
    )
    return per_row, model, diagnostics


def score_ragtruth_spans(out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache_path = ROOT / "local_cache/ragtruth_ec/test/ragtruth_ec_test.pkl"
    cache = _load_pickle(cache_path)
    rows = [cache[key] for key in sorted(cache) if cache[key]["condition"] == "full"]
    per_row, model, diagnostics = _fit_token_contract(rows)
    scores_by_method = {key: [] for key in model["weights"]}
    labels, groups = [], []
    for row, matrix in zip(rows, per_row):
        score = apply_spectral_model(matrix, model)
        y = np.zeros(len(matrix), dtype=int)
        for span in row.get("span_token_spans", []):
            if span is not None:
                y[max(0, int(span[0])):min(len(y), int(span[1]))] = 1
        labels.extend(y.tolist())
        groups.extend([str(row["source_id"])] * len(y))
        for method in scores_by_method:
            scores_by_method[method].extend(score[method].tolist())
    y = np.asarray(labels, dtype=int)
    score_map = {key: np.asarray(value) for key, value in scores_by_method.items()}
    signature = ProtocolSignature(
        "RAGTruth", "Qwen2.5-1.5B scorer", "test", "scorer token",
        "auroc", "RAGTruth character annotations projected to scorer tokens", len(y),
    )
    result = _method_rows(
        protocol_id="localization-lettucedetect-ragtruth-span", signature=signature,
        labels=y, groups=groups, score_map=score_map, draws=250,
    )
    manifest = json.loads((ROOT / "dataset_cache/four_localization/ragtruth_lettuce_large_span_full/manifest.json").read_text())
    reproduced = _float(manifest.get("observed_example_f1", manifest.get("example_f1", 0.792899)))
    for method, value, status in (
        ("LettuceDetect (published)", manifest.get("published_example_f1", 0.7922), "published reference"),
        ("LettuceDetect (local reproduction)", reproduced, "exact reproduction"),
    ):
        result.append({
            "protocol_id": "localization-lettucedetect-ragtruth-span", "dataset": "RAGTruth",
            "model": "ModernBERT-large token classifier", "split": "test",
            "prediction_unit": "character span", "grader": "RAGTruth span annotations",
            "subgroup": "all", "method_key": "lettucedetect", "method": method,
            "role": "published_ceiling", "metric": "example_f1", "value": _float(value),
            "ci_low": "", "ci_high": "", "n": 2700, "positive_rate": 943 / 2700,
            "status": status, "fidelity": "paper-faithful supervised span classifier",
            "caveat": "Different prediction unit from the scorer-token AUROC; shown as a ceiling, not a head-to-head row.",
        })
    diagnostics.update({"score_hash": score_hash(score_map, [str(i) for i in range(len(y))]),
                        "input_sha256": _sha256(cache_path), "n_responses": len(rows), "n_tokens": len(y)})
    return result, diagnostics


def _slice_for_char_span(span: tuple[int, int], text: str, token_count: int) -> slice:
    """Deterministic fallback mapping for the GASP cache's stored sentence spans.

    The GASP artifact stores sentence character spans but not scorer-token offsets.
    We therefore map cumulative character mass to the fixed answer-token trace and
    disclose this as an adaptation.  This is never labelled an exact reproduction.
    """

    length = max(len(text), 1)
    start = max(0, min(token_count - 1, int(math.floor(span[0] / length * token_count))))
    end = max(start + 1, min(token_count, int(math.ceil(span[1] / length * token_count))))
    return slice(start, end)


def score_gasp_sentences(out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = ROOT / "dataset_cache/four_localization/gasp_ragtruth_exact_qwen15b_full/gasp_exact.pkl"
    cache = _load_pickle(path)
    grouped: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in cache.values():
        grouped[str(row["response_id"])][str(row["condition"])] = row
    feature_names = [
        "full_entropy", "full_target_nll", "negative_noctx_nll_gap",
        "negative_noctx_jsd", "negative_max_loo_nll_drop",
        "negative_max_loo_jsd", "negative_mean_positive_loo_drop",
    ]
    features, gasp_rows, labels, groups, tasks, ids = [], [], [], [], [], []
    for response_id in sorted(grouped):
        conditions = grouped[response_id]
        full, noctx = conditions.get("full"), conditions.get("noctx")
        if full is None or noctx is None:
            continue
        loo = [conditions[key] for key in sorted(conditions) if key.startswith("loo_")]
        T = min(len(full["token_entropies"]), len(noctx["token_entropies"]))
        if T < 2:
            continue
        ent = np.asarray(full["token_entropies"], float)[:T]
        spill = np.asarray(full["token_spilled_energies"], float)[:T]
        no_gap = np.asarray(noctx["token_spilled_energies"], float)[:T] - spill
        no_jsd = np.asarray(noctx.get("token_jsd_vs_full"), float)[:T]
        loo_gap = [np.asarray(row["token_spilled_energies"], float)[:T] - spill for row in loo]
        loo_jsd = [np.asarray(row.get("token_jsd_vs_full"), float)[:T] for row in loo]
        gold = full.get("span_labels", [])
        text = str(full.get("response", ""))
        for sentence_index, raw_span in enumerate(full.get("sentence_spans", [])):
            span = (int(raw_span[0]), int(raw_span[1]))
            token_slice = _slice_for_char_span(span, text, T)
            mean_drop = [float(np.mean(value[token_slice])) for value in loo_gap]
            mean_jsd = [float(np.mean(value[token_slice])) for value in loo_jsd]
            max_drop = max(mean_drop) if mean_drop else 0.0
            max_jsd = max(mean_jsd) if mean_jsd else 0.0
            positive = [value for value in mean_drop if value > 0]
            gap = float(np.mean(no_gap[token_slice]))
            jsd0 = float(np.mean(no_jsd[token_slice]))
            features.append([
                float(np.mean(ent[token_slice])), float(np.mean(spill[token_slice])),
                -gap, -jsd0, -max_drop, -max_jsd,
                -float(np.mean(positive) if positive else 0.0),
            ])
            gasp_rows.append({
                "gasp_gap": gap, "gasp_jsd0": jsd0,
                "gasp_drop": max_drop, "gasp_jsdloo": max_jsd,
            })
            labels.append(int(any(
                max(span[0], int(item["start"])) < min(span[1], int(item["end"]))
                for item in gold
            )))
            groups.append(str(full["source_id"])); tasks.append(str(full["task_type"]))
            ids.append(f"{response_id}:sentence:{sentence_index}")
    matrix = np.asarray(features, dtype=float)
    spectral, diagnostics = fit_spectral_scores(matrix, feature_names=feature_names)
    spectral["gasp"] = gasp_threshold_scores(gasp_rows)
    y = np.asarray(labels, dtype=int)
    signature = ProtocolSignature(
        "RAGTruth balanced GASP cohort", "Qwen2.5-1.5B-Instruct", "400-response sample",
        "sentence", "auroc", "RAGTruth annotations", len(y),
    )
    result = _method_rows(
        protocol_id="localization-gasp-ragtruth-sentence", signature=signature,
        labels=y, groups=groups, score_map=spectral,
        method_roles={"gasp": "protocol_reproduction"}, draws=400,
    )
    for task in sorted(set(tasks)):
        mask = np.asarray([value == task for value in tasks])
        if len(np.unique(y[mask])) != 2:
            continue
        result.extend(_method_rows(
            protocol_id="localization-gasp-ragtruth-sentence", signature=signature,
            labels=y[mask], groups=list(np.asarray(groups)[mask]),
            score_map={key: value[mask] for key, value in spectral.items()},
            method_roles={"gasp": "protocol_reproduction"}, subgroup=task, draws=200,
        ))
    manifest = json.loads(path.with_name("manifest.json").read_text())
    for metric, value in (("auroc", manifest["target_numbers"]["span_auc"]),):
        result.append({
            "protocol_id": "localization-gasp-ragtruth-sentence", "dataset": signature.dataset,
            "model": signature.model, "split": "paper sample", "prediction_unit": "sentence",
            "grader": signature.grader, "subgroup": "published paper",
            "method_key": "gasp:published", "method": "GASP (published)",
            "role": "published_peer", "metric": metric, "value": value,
            "ci_low": "", "ci_high": "", "n": 400, "positive_rate": 0.5,
            "status": "published reference", "fidelity": "paper result",
            "caveat": "The paper did not publish its exact IDs or splitter; do not subtract this value from local rows.",
        })
    diagnostics.update({
        "score_hash": score_hash(spectral, ids), "input_sha256": _sha256(path),
        "n_responses": len(grouped), "n_sentences": len(y),
        "sentence_token_mapping": "character-mass projection; adaptation, not exact token offsets",
        "gasp_features": list(GASP_FEATURE_NAMES),
    })
    return result, diagnostics


def score_refchecker_claims(out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = ROOT / "dataset_cache/four_localization/refchecker_knowhalbench_open_full/refchecker_claim_telemetry.pkl"
    cache = _load_pickle(path)
    grouped: dict[tuple[str, str, str, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in cache.values():
        key = (
            str(row["setting"]), str(row["generator"]),
            str(row["example_id"]), int(row["claim_index"]),
        )
        grouped[key][str(row["condition"])] = row
    features, labels, groups, settings, ids = [], [], [], [], []
    names = None
    for key in sorted(grouped):
        conditions = grouped[key]
        full, noctx = conditions.get("full"), conditions.get("noctx")
        if full is None or noctx is None:
            continue
        base, base_names = _trace_summary(full)
        other, _ = _trace_summary(noctx)
        # A supported claim should lose confidence without evidence.  Negating
        # the change makes weak sensitivity a larger hallucination-risk value.
        contrast = [-(other[index] - base[index]) for index in (0, 2, 3, 5, 6, 7)]
        features.append(base + contrast)
        names = base_names + ["negative_noctx_change:" + base_names[index] for index in (0, 2, 3, 5, 6, 7)]
        # Bootstrap complete source examples: generator-specific claims from
        # the same source remain together within each separately scored setting.
        labels.append(int(full["label_unsupported"])); groups.append(str(full["example_id"]))
        settings.append(str(full["setting"])); ids.append("|".join(map(str, key)))
    matrix = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=int)
    signature = ProtocolSignature(
        "KnowHalBench fixed claims", "Qwen3-8B telemetry", "official fixed claims",
        "claim", "auroc", "human claim labels (unsupported binary collapse)", len(y),
    )
    # The three context settings are distinct tasks in RefChecker.  Do not pool
    # either their unsupervised fit or their evaluation.
    result = []
    setting_diagnostics = {}
    for setting in sorted(set(settings)):
        mask = np.asarray([value == setting for value in settings])
        spectral, fit_diagnostics = fit_spectral_scores(matrix[mask], feature_names=names or [])
        result.extend(_method_rows(
            protocol_id="localization-refchecker-knowhalbench-claim", signature=signature,
            labels=y[mask], groups=list(np.asarray(groups)[mask]),
            score_map=spectral,
            subgroup=setting, draws=150,
        ))
        setting_ids = list(np.asarray(ids)[mask])
        setting_diagnostics[setting] = {
            **fit_diagnostics, "n_claims": int(mask.sum()),
            "score_hash": score_hash(spectral, setting_ids),
        }
    manifest = json.loads(path.with_name("manifest.json").read_text())
    nli = manifest["arms"]["competitor"]["results"]
    for subgroup, source in (("zero_context", nli["zero_context"]),
                             ("noisy_context", nli["noisy_context"]),
                             ("accurate_context", nli["accurate_context"])):
        result.append({
            "protocol_id": "localization-refchecker-knowhalbench-claim",
            "dataset": "KnowHalBench fixed claims", "model": manifest["arms"]["competitor"]["model"],
            "split": "official fixed claims", "prediction_unit": "claim", "grader": "human 3-way labels",
            "subgroup": subgroup, "method_key": "refchecker_nli", "method": METHOD_LABELS["refchecker_nli"],
            "role": "published_peer", "metric": "macro_f1_3way", "value": source["macro_f1"],
            "ci_low": "", "ci_high": "", "n": source["n"], "positive_rate": "",
            "status": "local protocol reproduction", "fidelity": "open official checker",
            "caveat": "Three-way metric; our spectral rows use binary unsupported AUROC/AUPRC and are not directly subtracted.",
        })
    diagnostics = {
        "labels_seen_during_fit": False, "input_sha256": _sha256(path),
        "n_claims": len(y), "n_grouped_claims": len(grouped),
        "group_key_fields": ["setting", "generator", "example_id", "claim_index"],
        "settings_pooled": False, "settings_fitted_separately": True,
        "claim_extraction_evaluated": False, "per_setting": setting_diagnostics,
    }
    return result, diagnostics


def score_prmbench_steps(out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trace_path = ROOT / "dataset_cache/four_localization/prmbench_qwen3_8b_telemetry_full/prmbench_telemetry.pkl"
    prm_path = ROOT / "dataset_cache/four_localization/prmbench_qwen25math7b_full/prmbench_prm.pkl"
    traces = _load_pickle(trace_path); prm = _load_pickle(prm_path)
    competitor = {str(row["idx"]): row for row in prm.values()}
    feature_names = [f"{name}:{agg}" for name in CORE_TOKEN_VIEWS for agg in ("mean", "max")]
    features, labels, groups, categories, ids, competitor_scores = [], [], [], [], [], []
    excluded = []
    for raw_key in sorted(traces):
        row = traces[raw_key]
        idx = str(row["idx"])
        if idx in BAD_PRM_IDS:
            excluded.append(idx); continue
        comp = competitor.get(idx)
        if comp is None or len(comp["labels"]) != len(row["step_token_spans"]):
            excluded.append(idx); continue
        views = token_feature_views(row)
        for step_index, span in enumerate(row["step_token_spans"]):
            if span is None or int(span[1]) <= int(span[0]):
                continue
            start, end = int(span[0]), int(span[1])
            vector = []
            for name in CORE_TOKEN_VIEWS:
                values = np.asarray(views[name], float)[start:end]
                vector.extend([float(np.nanmean(values)), float(np.nanmax(values))])
            features.append(vector)
            # `comp["labels"]` is the PRM's thresholded prediction, not gold.
            # PRMBench's shipped `error_steps` are one-based human error indices.
            labels.append(int((step_index + 1) in {int(value) for value in row["error_steps"]}))
            competitor_scores.append(-float(comp["rewards"][step_index]))
            groups.append(idx); categories.append(str(row["classification"]))
            ids.append(f"{idx}:step:{step_index}")
    matrix = np.asarray(features, dtype=float)
    rng = np.random.default_rng(0)
    fit_population = np.flatnonzero(np.asarray(categories) != "correct")
    fit_index = rng.choice(fit_population, size=min(60_000, len(fit_population)), replace=False)
    model, diagnostics = fit_spectral_model(matrix[fit_index], feature_names=feature_names)
    spectral = apply_spectral_model(matrix, model)
    spectral["qwen_prm"] = np.asarray(competitor_scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    signature = ProtocolSignature(
        "PRMBench Preview", "Qwen3-8B telemetry / Qwen2.5-Math-PRM comparator",
        "official preview rows minus 3 defects", "reasoning step", "auroc",
        "official valid/invalid step labels", len(y),
    )
    result = []
    paper_categories = sorted(set(categories) - {"correct"})
    for category in sorted(set(categories)):
        mask = np.asarray([value == category for value in categories])
        if len(np.unique(y[mask])) != 2:
            continue
        result.extend(_method_rows(
            protocol_id="localization-prmbench-every-step", signature=signature,
            labels=y[mask], groups=list(np.asarray(groups)[mask]),
            score_map={key: value[mask] for key, value in spectral.items()},
            method_roles={"qwen_prm": "published_ceiling"}, subgroup=category, draws=100,
        ))
    error_mask = np.asarray([value in paper_categories for value in categories])
    result.extend(_method_rows(
        protocol_id="localization-prmbench-every-step", signature=signature,
        labels=y[error_mask], groups=list(np.asarray(groups)[error_mask]),
        score_map={key: value[error_mask] for key, value in spectral.items()},
        method_roles={"qwen_prm": "published_ceiling"},
        subgroup="all nine paper classes (constructed control excluded)", draws=250,
    ))
    manifest = json.loads(prm_path.with_name("manifest.json").read_text())
    result.append({
        "protocol_id": "localization-prmbench-every-step", "dataset": "PRMBench Preview",
        "model": manifest["model"], "split": "official preview", "prediction_unit": "reasoning step",
        "grader": "official step labels", "subgroup": "published official metric",
        "method_key": "qwen_prm:official", "method": METHOD_LABELS["qwen_prm"],
        "role": "published_ceiling", "metric": "f1_valid_step", "value": manifest["results"]["total"]["f1"],
        "ci_low": "", "ci_high": "", "n": len(y), "positive_rate": "",
        "status": "exact reproduction", "fidelity": "official evaluator port",
        "caveat": "Thresholded valid-step F1 is shown separately from error-step ranking metrics.",
    })
    diagnostics.update({
        "score_hash": score_hash(spectral, ids), "input_sha256": _sha256(trace_path),
        "competitor_sha256": _sha256(prm_path), "n_steps": len(y),
        "excluded_ids": sorted(set(excluded)), "expected_excluded_ids": sorted(BAD_PRM_IDS),
        "gold_label_source": "telemetry.error_steps (one-based official human error indices)",
        "prediction_labels_field_used_as_gold": False,
        "pooled_headline_excludes": ["correct"],
        "multi_solutions_note": "Part of the nine paper classes; it has no annotated error steps and therefore no standalone binary AUROC.",
        "step_adapter": "mean and max of five token-resolved views; not the 30-feature response contract",
    })
    return result, diagnostics


def score_processbench_first_error(out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Re-evaluate the three named heads on the same Qwen3-8B fixed traces."""

    from scripts.gl_liu_v1.run import evaluate_cell, mindgap_control
    from scripts.gl_liu_v1.two_stage_localization import evaluate_two_stage

    subsets = ("gsm8k", "math", "olympiadbench", "omnimath")
    chosen = {
        "deployed_upcr": ("answer_deployed_upcr_stable", "token_upcr_core"),
        "iu_pcr": ("answer_iu_mixed", "token_iu"),
        "dufs_liu_pcr": ("answer_dufs_liu_mixed", "token_dufs_liu_l0p1"),
    }
    per_subset: dict[str, dict[str, Mapping[str, Any]]] = {}
    diagnostics = {"labels_seen_during_score_fit": False, "subsets": {}}
    for subset in subsets:
        cell = evaluate_cell(
            "qwen3_8b", subset,
            ROOT / f"cache/localization/processbench/pb_qwen3_8b/processbench_{subset}.pkl",
        )
        missing = [
            pair for pair in chosen.values()
            if pair[0] not in cell["detectors"] or pair[1] not in cell["locators"]
        ]
        if missing:
            raise RuntimeError(f"ProcessBench {subset}: missing systems {missing}")
        systems = {
            method: evaluate_two_stage(
                cell["detectors"][pair[0]], cell["locators"][pair[1]], cell["labels"],
            )
            for method, pair in chosen.items()
        }
        paper_detector, paper_locator = mindgap_control(cell["rows"])
        systems["mind_the_gap"] = evaluate_two_stage(
            paper_detector, paper_locator, cell["labels"],
        )
        per_subset[subset] = systems
        diagnostics["subsets"][subset] = cell["diag"]

    prm_manifest = json.loads((ROOT / "dataset_cache/four_localization/pb_prm_qwen25math7b_full/manifest.json").read_text())
    judge_manifest = json.loads((ROOT / "dataset_cache/four_localization/pb_uprm_baseline_qwen3_8b_full/manifest.json").read_text())
    critic_path = ROOT / "dataset_cache/four_localization/pb_critic_qwen72b_full/manifest.json"
    critic_manifest = json.loads(critic_path.read_text())
    frozen_rows = read_csv(ROOT / "results/ours_only_localization_v1/final_systems_per_cell.csv")
    frozen = {
        row["subset"]: row for row in frozen_rows
        if row.get("model") == "qwen3_8b" and row.get("system") == "ours_only"
    }
    rows = []
    for subset in subsets:
        for method, record in per_subset[subset].items():
            rows.append({
                "protocol_id": "localization-processbench-first-error", "dataset": "ProcessBench",
                "model": "fixed ProcessBench reasoning traces", "split": "official subset",
                "prediction_unit": "first erroneous step / no-error", "grader": "ProcessBench labels",
                "subgroup": subset, "method_key": method, "method": METHOD_LABELS[method],
                "role": "published_peer" if method == "mind_the_gap" else "ours_exploratory",
                "metric": "f1", "value": record["f1"], "ci_low": "", "ci_high": "",
                "uncertainty_sd": record.get("f1_sd", ""), "n": record["n"],
                "positive_rate": "", "status": "local protocol reproduction",
                "fidelity": "same fixed trace and official metric",
                "caveat": "Exploratory matched pairing, not the frozen GL-LIU v1 system. A calibration half selects the operating threshold.",
            })
        frozen_row = frozen[subset]
        rows.append({
            "protocol_id": "localization-processbench-first-error", "dataset": "ProcessBench",
            "model": "fixed ProcessBench reasoning traces", "split": "official subset",
            "prediction_unit": "first erroneous step / no-error", "grader": "ProcessBench labels",
            "subgroup": subset, "method_key": "gl_liu_v1", "method": METHOD_LABELS["gl_liu_v1"],
            "role": "ours_frozen", "metric": "f1", "value": _float(frozen_row["f1"]),
            "ci_low": "", "ci_high": "", "uncertainty_sd": _float(frozen_row["f1_sd"]),
            "n": int(frozen_row["n"]), "positive_rate": "", "status": "frozen confirmation system",
            "fidelity": "preselected GL-LIU v1 pairing",
            "caveat": "Detector and locator were selected only on the declared Qwen3-4B development cells.",
        })
        for key, manifest, role in (
            ("qwen_prm", prm_manifest, "published_ceiling"),
            ("qwen3_judge", judge_manifest, "control"),
        ):
            cell = manifest["cells"][subset]
            rows.append({
                "protocol_id": "localization-processbench-first-error", "dataset": "ProcessBench",
                "model": "fixed ProcessBench reasoning traces", "split": "official subset",
                "prediction_unit": "first erroneous step / no-error", "grader": "ProcessBench labels",
                "subgroup": subset, "method_key": key, "method": METHOD_LABELS[key], "role": role,
                "metric": "f1", "value": float(cell["f1"]) / 100.0,
                "ci_low": "", "ci_high": "", "n": cell.get("n_rows", per_subset[subset]["mind_the_gap"]["n"]),
                "positive_rate": "", "status": "exact local competitor run",
                "fidelity": "official fixed traces and metric",
                "caveat": "Supervised PRM is a ceiling; the Qwen3 judge is a control, not uPRM.",
            })
        critic = critic_manifest["cells"][subset]
        rows.append({
            "protocol_id": "localization-processbench-first-error", "dataset": "ProcessBench",
            "model": "fixed ProcessBench reasoning traces", "split": "official subset",
            "prediction_unit": "first erroneous step / no-error", "grader": "ProcessBench labels",
            "subgroup": subset, "method_key": "qwen72b_critic", "method": METHOD_LABELS["qwen72b_critic"],
            "role": "published_peer", "metric": "f1", "value": float(critic["f1"]) / 100.0,
            "ci_low": "", "ci_high": "", "n": per_subset[subset]["mind_the_gap"]["n"],
            "positive_rate": "", "status": "local protocol reproduction",
            "fidelity": "ProcessBench critic protocol; different critic model",
            "caveat": "Uses ProcessBench's critic prompt with Qwen2.5-72B instead of QwQ-32B-Preview.",
        })
    # The macro is a descriptive mean over the four official subsets, not a suite-wide macro.
    for method in (*chosen, "mind_the_gap"):
        values = [per_subset[subset][method]["f1"] for subset in subsets]
        rows.append({
            **{key: rows[0][key] for key in ("protocol_id", "dataset", "model", "split", "prediction_unit", "grader")},
            "subgroup": "four-subset macro", "method_key": method, "method": METHOD_LABELS[method],
            "role": "published_peer" if method == "mind_the_gap" else "ours_exploratory",
            "metric": "f1", "value": float(np.mean(values)), "ci_low": "", "ci_high": "",
            "n": 3400, "positive_rate": 2221 / 3400, "status": "local protocol reproduction",
            "fidelity": "unweighted official-subset macro", "caveat": "No cross-task averaging.",
        })
    for key, manifest, role in (("qwen_prm", prm_manifest, "published_ceiling"),
                                ("qwen3_judge", judge_manifest, "control")):
        values = [float(manifest["cells"][subset]["f1"]) / 100.0 for subset in subsets]
        rows.append({
            **{field: rows[0][field] for field in ("protocol_id", "dataset", "model", "split", "prediction_unit", "grader")},
            "subgroup": "four-subset macro", "method_key": key, "method": METHOD_LABELS[key],
            "role": role, "metric": "f1", "value": float(np.mean(values)), "ci_low": "", "ci_high": "",
            "n": 3400, "positive_rate": 2221 / 3400, "status": "exact local competitor run",
            "fidelity": "unweighted official-subset macro", "caveat": "All four official subsets are present.",
        })
    frozen_values = [_float(frozen[subset]["f1"]) for subset in subsets]
    critic_values = [float(critic_manifest["cells"][subset]["f1"]) / 100.0 for subset in subsets]
    for key, values, role, fidelity, caveat in (
        ("gl_liu_v1", frozen_values, "ours_frozen", "preselected GL-LIU v1 official-subset macro",
         "Frozen detector/locator selection; thresholds use split-local calibration labels."),
        ("qwen72b_critic", critic_values, "published_peer", "ProcessBench critic-protocol macro",
         "Different critic model from the paper; all four subsets complete."),
    ):
        rows.append({
            **{field: rows[0][field] for field in ("protocol_id", "dataset", "model", "split", "prediction_unit", "grader")},
            "subgroup": "four-subset macro", "method_key": key, "method": METHOD_LABELS[key],
            "role": role, "metric": "f1", "value": float(np.mean(values)),
            "ci_low": "", "ci_high": "", "n": 3400, "positive_rate": 2221 / 3400,
            "status": "local protocol reproduction", "fidelity": fidelity, "caveat": caveat,
        })

    # Independent-family confirmation: the frozen GL-LIU system is essentially
    # tied with a transparent max-token-entropy detector.  This prevents an
    # advisor-facing overclaim based only on the in-family Qwen3 panel.
    external = read_csv(ROOT / "results/gl_liu_external_v1/llama31_8b/external_systems_per_cell.csv")
    for record in external:
        if record.get("system") not in {"gl_liu_v1_frozen", "candidate_detector__baseline_max_entropy"}:
            continue
        key = "gl_liu_v1" if record["system"] == "gl_liu_v1_frozen" else "max_entropy"
        rows.append({
            "protocol_id": "localization-processbench-first-error", "dataset": "ProcessBench",
            "model": "Llama-3.1-8B external-family fixed traces", "split": "official subset",
            "prediction_unit": "first erroneous step / no-error", "grader": "ProcessBench labels",
            "subgroup": "external-family: " + record["subset"], "method_key": key,
            "method": METHOD_LABELS[key], "role": "ours_frozen" if key == "gl_liu_v1" else "transparent_baseline",
            "metric": "f1", "value": _float(record["f1"]), "ci_low": "", "ci_high": "",
            "uncertainty_sd": _float(record.get("f1_sd")), "n": int(record["n"]),
            "positive_rate": "", "status": "independent-family confirmation",
            "fidelity": "same fixed trace and official metric",
            "caveat": "The 0.21-point macro margin between GL-LIU and max entropy is noise-level and changes sign by subset.",
        })
    external_macro = read_csv(ROOT / "results/gl_liu_external_v1/llama31_8b/external_macro_f1.csv")
    macro_by_system = {row["system"]: _float(row["macro_f1"]) for row in external_macro}
    for key, system, role in (("gl_liu_v1", "gl_liu_v1_frozen", "ours_frozen"),
                              ("max_entropy", "candidate_detector__baseline_max_entropy", "transparent_baseline")):
        rows.append({
            "protocol_id": "localization-processbench-first-error", "dataset": "ProcessBench",
            "model": "Llama-3.1-8B external-family fixed traces", "split": "official four-subset macro",
            "prediction_unit": "first erroneous step / no-error", "grader": "ProcessBench labels",
            "subgroup": "external-family: four-subset macro", "method_key": key,
            "method": METHOD_LABELS[key], "role": role, "metric": "f1",
            "value": macro_by_system[system], "ci_low": "", "ci_high": "", "n": 3400,
            "positive_rate": "", "status": "independent-family confirmation",
            "fidelity": "unweighted official-subset macro",
            "caveat": "GL-LIU 0.3171 versus max entropy 0.3150: noise-level difference, not a confirmed win.",
        })
    diagnostics.update({
        "core_method_systems": chosen, "threshold_splits": 100,
        "frozen_gl_liu_system": {"detector": "answer_dufs_liu_mixed", "locator": "token_temporal_liu_l0p3"},
        "critic_included": True, "critic_manifest_sha256": _sha256(critic_path),
        "external_family_controls": ["gl_liu_v1_frozen", "candidate_detector__baseline_max_entropy"],
    })
    return rows, diagnostics


def score_semgrad_detection(out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, diagnostics = [], {}
    for dataset in ("sciq", "truthfulqa"):
        path = ROOT / f"local_cache/semgrad_bem_regraded/raw_semgrad_{dataset}_T0.0_bem.pkl"
        cache = _load_pickle(path)
        samples = []
        for key in sorted(cache):
            record = cache[key]
            if not record.get("candidates"):
                continue
            candidate = record["candidates"][0]
            vector, names = _trace_summary(candidate)
            samples.append((str(key), vector, int(not candidate["bem_correct"])))
        matrix = np.asarray([item[1] for item in samples], dtype=float)
        spectral, fit_diag = fit_spectral_scores(matrix, feature_names=names)
        labels = np.asarray([item[2] for item in samples], dtype=int)
        signature = ProtocolSignature(
            dataset.upper(), "Qwen3-8B", "full frozen run", "answer", "auroc",
            "BEM answer-equivalence grader", len(labels),
        )
        rows.extend(_method_rows(
            protocol_id=f"detection-semgrad-{dataset}", signature=signature,
            labels=labels, groups=[item[0] for item in samples], score_map=spectral,
            draws=300,
        ))
        diagnostics[dataset] = {
            **fit_diag, "score_hash": score_hash(spectral, [item[0] for item in samples]),
            "input_sha256": _sha256(path), "n": len(samples),
            "grader_note": "BEM matches the original SemGrad automatic answer-equivalence stage; manual audit remains a limitation.",
        }
    return rows, diagnostics


def score_ragtruth_detection(out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load the frozen Step-239 RAG response-detection result.

    These rows were already score-hashed before labels were opened in the
    original experiment.  This suite does not refit or retune any arm.
    """

    path = ROOT / "results/rag_ec_v1/full_test_split_result.json"
    artifact = json.loads(path.read_text())
    ci = artifact["grouped_bootstrap_auroc_ci"]
    rows = []
    labels = {
        "likelihood_drop": "Likelihood drop",
        "gasp_reproduction": "GASP reproduction",
        "fusion_isolation_naive_avg": "Evidence-contrast naive average",
        "ec_upcr": "Evidence-Contrast U-PCR",
        "full_context_only_dufs_liu": "Full-context-only DUFS-LIU",
        "ec_dufs_liu_temporal": "Evidence-Contrast DUFS-LIU (temporal graph)",
        "ec_dufs_liu_evidence_graph": "Evidence-Contrast DUFS-LIU (evidence graph)",
    }
    for record in artifact["detector_rows"]:
        arm = record["arm"]
        interval = ci[arm]
        rows.append({
            "protocol_id": "detection-ragtruth-evidence-contrast", "dataset": "RAGTruth",
            "model": "Qwen2.5-1.5B-Instruct scorer", "split": "full test split",
            "prediction_unit": "answer", "grader": "RAGTruth response label",
            "subgroup": "all responses", "method_key": arm, "method": labels.get(arm, arm),
            "role": "ours" if arm.startswith("ec_") else "transparent_baseline",
            "metric": "auroc", "value": record["auroc"], "ci_low": interval["ci_lo"],
            "ci_high": interval["ci_hi"], "n": record["n_valid"], "positive_rate": "",
            "status": "frozen exploratory result", "fidelity": "hashed before labels",
            "caveat": "RAGTruth labels were previously opened; no arm was tuned inside this reporting suite.",
        })
    delta = artifact["paired_diff_vs_fusion_isolation_naive_avg"]["ec_dufs_liu_evidence_graph"]
    diagnostics = {
        "input_sha256": _sha256(path), "score_hashes_before_labels": artifact["score_hashes_before_labels"]["response"],
        "n_responses": artifact["n_responses"], "n_source_ids": artifact["n_source_ids"],
        "bootstrap_group": "source_id", "novelty_test": delta,
        "novelty_confirmed": bool(delta["ci_lo"] > 0),
        "scientific_conclusion": "Evidence intervention design is supported; fusion gain over naive averaging is not confirmed.",
    }
    return rows, diagnostics


PAPER_TITLES = {
    "epr": "EPR",
    "beyond-next-token-probabilities-learnable-fast-detection-of": "LOS-Net: Beyond Next-Token Probabilities",
    "zero-source-llm-hallucination-detection-with-human-like-crit": "HCPD / TSV / zero-source detection",
    "automatic-layer-selection-for-hallucination-detection": "Automatic Layer Selection / Semantic Entropy",
    "semantic-energy-detecting-llm-hallucination-beyond-entropy": "Semantic Energy",
    "hallucination-detection-in-llms-using-spectral-features-of-a": "LapEigvals and AttentionScore",
    "enhancing-hallucination-detection-through-noise-injection": "Noise Injection",
    "harnessing-reasoning-trajectories-for-hallucination-detectio": "ARS: reasoning-trajectory detection",
    "hallucination-detection-via-internal-states-and-structured-r": "Internal States + Reasoning Consistency",
    "harp-hallucination-detection-via-reasoning-subspace-projecti": "HARP",
    "inside-llms-internal-states-retain-the": "INSIDE",
}


LOCALIZATION_META = {
    "localization-lettucedetect-ragtruth-span": {
        "title": "RAGTruth character/token-span localization",
        "paper": "LettuceDetect: A Hallucination Detection Framework for RAG Applications",
        "benchmark_revision": "RAGTruth test; locally stored 2,700 responses",
        "prompt": "Fixed published response; one teacher-forced scorer pass for our methods",
        "decoding": "No new answer generation",
        "bootstrap_group": "source_id",
        "readiness": "READY_WITH_LIMITATIONS",
        "limitations": [
            "Our methods are evaluated on scorer tokens; LettuceDetect is evaluated on character spans.",
            "The supervised example-F1 ceiling is therefore shown separately, not subtracted from token AUROC.",
            "RAGTruth labels were opened in earlier work; this is exploratory.",
        ],
    },
    "localization-gasp-ragtruth-sentence": {
        "title": "RAGTruth sentence localization with evidence perturbation",
        "paper": "GASP: Grounding-Aware Sensitivity by Perturbation",
        "benchmark_revision": "Own balanced 400-response protocol reproduction",
        "prompt": "Fixed response rescored under full, no-context, and leave-one-chunk-out evidence",
        "decoding": "Teacher-forced rescoring; no new generation",
        "bootstrap_group": "source_id",
        "readiness": "READY_WITH_LIMITATIONS",
        "limitations": [
            "The paper did not release its exact 400 IDs or sentence splitter.",
            "Sentence-to-token mapping uses deterministic character-mass projection because offsets were not stored.",
            "Published and local GASP numbers are reference and reproduction rows, not a paired subtraction.",
        ],
    },
    "localization-refchecker-knowhalbench-claim": {
        "title": "KnowHalBench fixed-claim checking",
        "paper": "Knowledge-Centric Hallucination Detection / RefChecker",
        "benchmark_revision": "Official fixed Claude-2 claim triplets",
        "prompt": "Each fixed claim is scored with evidence and without evidence",
        "decoding": "Teacher-forced claim scoring; no response generation",
        "bootstrap_group": "example_id",
        "readiness": "READY_WITH_LIMITATIONS",
        "limitations": [
            "This page evaluates claim checking, not claim extraction.",
            "RefChecker reports three-way macro-F1; our spectral methods report binary unsupported AUROC/AUPRC.",
        ],
    },
    "localization-prmbench-every-step": {
        "title": "PRMBench every-step error scoring",
        "paper": "PRMBench and Qwen2.5-Math-PRM",
        "benchmark_revision": "PRMBench Preview; 6,966 rows after registered exclusions",
        "prompt": "Fixed official reasoning trace; teacher-forced Qwen3-8B telemetry",
        "decoding": "No new reasoning generation",
        "bootstrap_group": "PRMBench row ID",
        "readiness": "READY_WITH_LIMITATIONS",
        "limitations": [
            "Three named alignment-defect IDs are excluded from every method.",
            "Qwen PRM is trained with process supervision and is a ceiling, not a label-free peer.",
            "The headline excludes only the constructed correct controls. The multi_solutions class remains part of the nine paper classes but has no standalone binary AUROC because it contains no annotated error step.",
            "The adapter uses mean/max summaries of five token views; it is not the 30-feature response contract.",
            "71.0% of steps are shorter than 32 tokens (median 24), so many long-trace spectral features are unavailable.",
        ],
    },
    "localization-processbench-first-error": {
        "title": "ProcessBench first-error localization",
        "paper": "Mind the Gap / ProcessBench protocol",
        "benchmark_revision": "All four official subsets; Qwen3-8B telemetry",
        "prompt": "Fixed reasoning trace; no extra answer generation for spectral methods",
        "decoding": "Threshold calibrated on one half, evaluated on the other; 100 repeated splits",
        "bootstrap_group": "ProcessBench row within official subset",
        "readiness": "READY",
        "limitations": [
            "Qwen2.5-Math-PRM uses human process supervision and is a ceiling.",
            "The Qwen3 judge row is a control, not the uPRM algorithm.",
            "The Qwen2.5-72B row reproduces the ProcessBench critic protocol with a different critic model.",
            "GL-LIU uses labels on declared development cells for component selection and on each calibration half for its threshold.",
            "On the independent Llama-3.1-8B family, GL-LIU is essentially tied with maximum token entropy.",
        ],
    },
}


RAG_DETECTION_META = {
    "title": "RAGTruth response detection with evidence contrast",
    "paper": "RAGTruth / GASP protocol context",
    "benchmark_revision": "RAGTruth full test; frozen Step-239 result",
    "prompt": "Fixed answer rescored under evidence interventions",
    "decoding": "Teacher-forced rescoring; no answer regeneration",
    "bootstrap_group": "source_id",
    "readiness": "EXPLORATORY",
    "limitations": [
        "RAGTruth labels were opened in earlier development.",
        "The evidence-graph gain over naive averaging is +2.51 points with a 95% interval crossing zero.",
        "The intervention design is supported; the additional fusion mechanism is not confirmed.",
    ],
}


def build_registry(rows: list[dict[str, Any]], competitors: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["protocol_id"])].append(row)
    competitor_by_slug: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in competitors:
        competitor_by_slug[row.get("paper_slug", "")].append(row)
    registry = []
    for protocol_id in sorted(grouped):
        protocol_rows = grouped[protocol_id]
        first = protocol_rows[0]
        if protocol_id.startswith("detection-semgrad-"):
            dataset = protocol_id.rsplit("-", 1)[-1]
            entry = {
                "protocol_id": protocol_id, "task_family": "answer hallucination detection",
                "title": f"SemGrad {dataset.upper()} detection", "paper": "SemGrad evaluation protocol",
                "benchmark_revision": "full cluster run", "dataset": dataset.upper(), "model": "Qwen3-8B",
                "split": "full frozen run", "sample_count": max(int(_float(row.get("n"), 0)) for row in protocol_rows),
                "prompt": "dataset question; one generated answer", "decoding": "temperature 0",
                "grader": "BEM answer-equivalence model", "prediction_unit": "answer",
                "metrics": ["AUROC", "AUPRC"], "bootstrap_group": "question ID",
                "published_competitors": [], "published_tables": [],
                "local_reproduction_status": "paper-faithful automatic grading stage",
                "fidelity_level": "ready with manual-audit limitation",
                "feature_contract": "common 8-feature trace adapter to the frozen solvers",
                "readiness": "BACKGROUND_ONLY", "limitations": ["BEM disagreements still merit a stratified human audit."],
            }
        elif protocol_id == "detection-ragtruth-evidence-contrast":
            entry = {
                "protocol_id": protocol_id, "task_family": "RAG answer hallucination detection",
                **RAG_DETECTION_META, "dataset": "RAGTruth", "model": "Qwen2.5-1.5B-Instruct scorer",
                "split": "full test split", "sample_count": max(int(_float(row.get("n"), 0)) for row in protocol_rows),
                "grader": "RAGTruth response labels", "prediction_unit": "answer",
                "metrics": ["AUROC"], "published_competitors": [], "published_tables": [],
                "local_reproduction_status": "frozen exploratory evaluation",
                "fidelity_level": "scores hashed before labels; labels previously opened in this research program",
                "feature_contract": "evidence-contrast response arms from Step 239",
            }
        elif protocol_id.startswith("detection-"):
            slug = protocol_id.removeprefix("detection-")
            refs = competitor_by_slug.get(slug, [])
            local_cells = sorted({row.get("cell", "") for row in protocol_rows if row.get("role") == "ours"})
            entry = {
                "protocol_id": protocol_id, "task_family": "answer hallucination detection",
                "title": PAPER_TITLES.get(slug, slug.replace("-", " ").title()),
                "paper": PAPER_TITLES.get(slug, slug), "benchmark_revision": "frozen protocol cells",
                "dataset": sorted({row.get("dataset", "") for row in refs}),
                "model": sorted({row.get("model", "") for row in refs}),
                "split": "paper/cell-specific frozen split", "sample_count": sum(int(_float(row.get("n"), 0)) for row in protocol_rows if row.get("method_key") == "iu_pcr"),
                "prompt": "Frozen in the cell preset", "decoding": "Frozen in the cell preset",
                "grader": "Cell-specific frozen correctness grader", "prediction_unit": "answer",
                "metrics": ["AUROC"], "bootstrap_group": "not re-bootstrapped in this report",
                "published_competitors": sorted({row.get("method", "") for row in refs}),
                "published_tables": sorted({row.get("paper_table", "") for row in refs if row.get("paper_table")}),
                "local_reproduction_status": "retrospective frozen-cell evaluation",
                "fidelity_level": "published reference plus same-cell local score; no new reproduction",
                "feature_contract": "Deployed U-PCR=fixed_stable_v1; IU/DUFS-LIU-PCR=full-pool mixed-v2",
                "readiness": "READY_WITH_LIMITATIONS", "applicable_cells": local_cells,
                "limitations": [
                    "These cells were used during development and are not independent confirmation.",
                    "Published values are shown as paper references; a delta is not computed where split/grader details are incomplete.",
                    "This suite intentionally uses the frozen full-pool core methods. The older repgrid uses per-cell best-subset compatibility arms and can therefore report different values; it is not interchangeable with this table.",
                ],
            }
        elif protocol_id == "internal-transfer":
            entry = {
                "protocol_id": protocol_id, "task_family": "answer hallucination detection",
                "title": "Internal transfer cells without a verified paper comparator",
                "paper": "None", "benchmark_revision": "six cluster-era cells",
                "dataset": sorted({row.get("cell", "") for row in protocol_rows}),
                "model": "cell-specific", "split": "frozen", "sample_count": "",
                "prompt": "frozen", "decoding": "frozen", "grader": "cell-specific",
                "prediction_unit": "answer", "metrics": ["AUROC"], "bootstrap_group": "none",
                "published_competitors": [], "published_tables": [],
                "local_reproduction_status": "internal transfer only", "fidelity_level": "no paper comparison",
                "feature_contract": "frozen core contracts", "readiness": "INTERNAL_APPENDIX",
                "limitations": ["No verified published comparator exists for these exact cells."],
            }
        else:
            meta = LOCALIZATION_META[protocol_id]
            entry = {
                "protocol_id": protocol_id, "task_family": "hallucination localization",
                **meta, "dataset": sorted({row.get("dataset", "") for row in protocol_rows}),
                "model": sorted({row.get("model", "") for row in protocol_rows}),
                "split": sorted({row.get("split", "") for row in protocol_rows}),
                "sample_count": max(int(_float(row.get("n"), 0)) for row in protocol_rows),
                "grader": sorted({row.get("grader", "") for row in protocol_rows}),
                "prediction_unit": sorted({row.get("prediction_unit", "") for row in protocol_rows}),
                "metrics": sorted({row.get("metric", "") for row in protocol_rows}),
                "published_competitors": sorted({row.get("method", "") for row in protocol_rows if "published" in row.get("role", "") or row.get("role") == "published_peer"}),
                "published_tables": [], "local_reproduction_status": "see fidelity panel",
                "fidelity_level": "exact reproduction, protocol reproduction, and adaptation are separated",
                "feature_contract": "task-local risk matrix; same frozen three solvers",
            }
        stored_passes: Any = 1
        if protocol_id in {"detection-ragtruth-evidence-contrast", "localization-gasp-ragtruth-sentence"}:
            stored_passes = "2 + number of LOO evidence chunks"
        elif protocol_id == "localization-refchecker-knowhalbench-claim":
            stored_passes = 2
        entry["method_access"] = {
            "Deployed U-PCR": {"labels_for_fit": "no", "passes": stored_passes, "access": "gray-box token probabilities"},
            "IU-PCR": {"labels_for_fit": "no", "passes": stored_passes, "access": "gray-box token probabilities"},
            "DUFS-LIU-PCR": {"labels_for_fit": "no", "passes": stored_passes, "access": "gray-box token probabilities; graph fit"},
        }
        registry.append(entry)
    return registry


CSS = """
:root{--ink:#182235;--muted:#637083;--paper:#fff;--bg:#f2f5f8;--blue:#2563eb;--teal:#0f766e;--amber:#b45309;--line:#d9e1ea}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 Inter,ui-sans-serif,system-ui,-apple-system,sans-serif}
header{background:#101827;color:white;padding:24px 5vw}header a{color:#c7d8ff;text-decoration:none;margin-right:18px}main{max-width:1240px;margin:auto;padding:28px 5vw 60px}
h1{font-size:34px;line-height:1.15;margin:8px 0}h2{margin-top:34px}.lede{color:#d4deec;max-width:900px}.card{background:var(--paper);border:1px solid var(--line);border-radius:14px;padding:20px;margin:16px 0;box-shadow:0 3px 14px #12213a0a}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:14px}.pill{display:inline-block;padding:3px 9px;border-radius:99px;background:#e7eefc;color:#1d4ed8;font-size:12px;font-weight:700}.muted{color:var(--muted)}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:9px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{background:#f7f9fb;position:sticky;top:0}.table-wrap{overflow:auto;max-height:680px}
.chart{width:100%;min-height:140px;background:#fbfcfe;border:1px solid var(--line);border-radius:10px}.chart text{font-size:12px;fill:var(--ink)}.chart .chart-title{font-size:15px;font-weight:700}.bar-ours{fill:var(--blue)}.bar-published{fill:var(--teal)}.bar-ceiling{fill:var(--amber)}
.ok{color:#047857;font-weight:700}.warn{color:#b45309;font-weight:700}.navgrid a{display:block;background:white;border:1px solid var(--line);border-radius:12px;padding:15px;text-decoration:none;color:var(--ink)}code{background:#edf1f5;padding:2px 5px;border-radius:5px}.empty{padding:20px;color:var(--muted)}
"""


def _page(title: str, body: str, registry: list[dict[str, Any]]) -> str:
    nav = " ".join(f"<a href='{esc(item['protocol_id'])}.html'>{esc(item['title'])}</a>" for item in registry[:5])
    return f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>{esc(title)}</title><style>{CSS}</style></head><body><header><nav><a href='index.html'>Suite index</a>{nav}</nav><h1>{esc(title)}</h1><p class='lede'>One published protocol per page. Incompatible metrics are never averaged.</p></header><main>{body}</main></body></html>"


def _fmt(value: Any) -> str:
    if value in (None, ""):
        return "—"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return esc(value)


def _fmt_count(value: Any) -> str:
    if value in (None, ""):
        return "—"
    try:
        number = float(value)
        return str(int(number)) if number.is_integer() else f"{number:.4f}"
    except (TypeError, ValueError):
        return esc(value)


def render_protocol(entry: Mapping[str, Any], rows: list[dict[str, Any]], registry: list[dict[str, Any]]) -> str:
    protocol_id = str(entry["protocol_id"])
    rows = [row for row in rows if row["protocol_id"] == protocol_id]
    subgroups = sorted({str(row.get("subgroup", "all")) for row in rows})
    metrics = sorted({str(row.get("metric", "")) for row in rows})
    charts = []
    for subgroup in subgroups:
        selected = [row for row in rows if str(row.get("subgroup", "all")) == subgroup]
        for metric in metrics:
            same = [row for row in selected if row.get("metric") == metric]
            # A chart is a visual head-to-head.  Partition by the complete
            # protocol signature so a published reference with a different
            # model/split/grader can never look like a paired comparison.
            compatible: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
            for row in same:
                signature = ProtocolSignature(
                    str(row.get("dataset", "")), str(row.get("model", "")),
                    str(row.get("split", "")), str(row.get("prediction_unit", "")),
                    str(row.get("metric", "")), str(row.get("grader", "")),
                )
                compatible[signature.compatibility_key()].append(row)
            for index, group_rows in enumerate(compatible.values(), 1):
                signatures = [ProtocolSignature(
                    str(row.get("dataset", "")), str(row.get("model", "")),
                    str(row.get("split", "")), str(row.get("prediction_unit", "")),
                    str(row.get("metric", "")), str(row.get("grader", "")),
                ) for row in group_rows]
                for signature in signatures[1:]:
                    assert_protocol_match(signatures[0], signature)
                label = "matched comparison" if len(group_rows) > 1 else "reference row"
                suffix = f" · {label}" + (f" {index}" if len(compatible) > 1 else "")
                charts.append(
                    f"<div class='card'><h3>{esc(subgroup)} · {esc(metric)}{esc(suffix)}</h3>"
                    f"{bar_chart(group_rows, metric=metric, title=f'{subgroup}: {metric} ({label})')}</div>"
                )
    table_rows = "".join(
        "<tr>" + "".join(f"<td>{formatter(row.get(key))}</td>" for key, formatter in (
            ("subgroup", _fmt), ("method", _fmt), ("role", _fmt), ("metric", _fmt),
            ("value", _fmt), ("ci_low", _fmt), ("ci_high", _fmt), ("n", _fmt_count),
            ("positive_rate", _fmt), ("fidelity", _fmt), ("caveat", _fmt),
        )) + "</tr>" for row in rows
    )
    access_rows = "".join(
        f"<tr><td>{esc(method)}</td><td>{esc(data['labels_for_fit'])}</td><td>{esc(data['passes'])}</td><td>{esc(data['access'])}</td></tr>"
        for method, data in entry["method_access"].items()
    )
    limitations = "".join(f"<li>{esc(item)}</li>" for item in entry.get("limitations", []))
    local_roles = {"ours", "ours_frozen"}
    headline_subgroups = {
        "all", "all responses", "four-subset macro",
        "all nine paper classes (constructed control excluded)",
    }
    local = [
        row for row in rows
        if row.get("role") in local_roles and row.get("subgroup") in headline_subgroups
    ]
    conclusion = "No common local headline metric is available yet."
    if local:
        by_metric = defaultdict(list)
        for row in local: by_metric[row["metric"]].append(row)
        metric = sorted(by_metric)[0]
        best = max(by_metric[metric], key=lambda row: _float(row.get("value"), -math.inf))
        conclusion = f"Within the local {metric} rows, {best['method']} is highest at {_fmt(best['value'])}. This is protocol-scoped and is not a suite-wide ranking."
    body = f"""
    <section class='card'><span class='pill'>{esc(entry['readiness'])}</span><h2>Protocol card</h2>
    <div class='grid'><div><b>Paper</b><br>{esc(entry['paper'])}</div><div><b>Benchmark revision</b><br>{esc(entry['benchmark_revision'])}</div><div><b>Dataset</b><br>{esc(entry['dataset'])}</div><div><b>Model</b><br>{esc(entry['model'])}</div><div><b>Prediction unit</b><br>{esc(entry['prediction_unit'])}</div><div><b>Grader</b><br>{esc(entry['grader'])}</div><div><b>Feature contract</b><br>{esc(entry['feature_contract'])}</div><div><b>Bootstrap group</b><br>{esc(entry['bootstrap_group'])}</div></div></section>
    <section class='card'><h2>What the three methods change</h2><p><b>Deployed U-PCR</b> solves the original pairwise covariance equations and may exclude weak features. <b>IU-PCR</b> keeps the full pool and uses two covariance components. <b>DUFS-LIU-PCR</b> first learns label-free continuous feature gates, builds a k=7 sample graph, and adds the frozen Laplacian penalty λ=0.1. The task adapter changes the prediction unit; the solvers do not change.</p></section>
    <h2>Matched views</h2>{''.join(charts)}
    <section class='card'><h2>Cost and access</h2><table><thead><tr><th>Method</th><th>Labels for score fit</th><th>Model passes</th><th>Access</th></tr></thead><tbody>{access_rows}</tbody></table></section>
    <section class='card'><h2>Every machine-readable result row</h2><div class='table-wrap'><table><thead><tr><th>Cell/subgroup</th><th>Method</th><th>Role</th><th>Metric</th><th>Value</th><th>CI low</th><th>CI high</th><th>n</th><th>Balance</th><th>Fidelity</th><th>Caveat</th></tr></thead><tbody>{table_rows}</tbody></table></div></section>
    <section class='card'><h2>Fidelity and limitations</h2><p><b>Local status:</b> {esc(entry['local_reproduction_status'])}. <b>Fidelity:</b> {esc(entry['fidelity_level'])}.</p><ul>{limitations}</ul></section>
    <section class='card'><h2>Protocol-scoped conclusion</h2><p>{esc(conclusion)}</p></section>
    """
    return _page(str(entry["title"]), body, registry)


def render_index(registry: list[dict[str, Any]], rows: list[dict[str, Any]]) -> str:
    table = "".join(
        f"<tr><td><a href='{esc(entry['protocol_id'])}.html'>{esc(entry['title'])}</a></td><td>{esc(entry['task_family'])}</td><td>{esc(entry['prediction_unit'])}</td><td>{esc(entry['metrics'])}</td><td>{esc(entry['readiness'])}</td><td>{esc(entry['sample_count'])}</td></tr>"
        for entry in registry
    )
    body = f"""
    <section class='card'><h2>How to read this suite</h2><p>Each row is one paper-aligned protocol. AUROC for answer detection is not averaged with span F1, claim AUROC, step AUROC, or first-error F1. Published references, exact reproductions, protocol reproductions, and adaptations are marked separately.</p><p><b>Core progression:</b> Deployed U-PCR → IU-PCR → DUFS-LIU-PCR. “IO-PCR” is normalized to IU-PCR.</p></section>
    <section class='card'><h2>Protocol index</h2><div class='table-wrap'><table><thead><tr><th>Protocol</th><th>Task family</th><th>Prediction unit</th><th>Metrics</th><th>Readiness</th><th>n</th></tr></thead><tbody>{table}</tbody></table></div></section>
    <section class='card'><h2>Claim boundary</h2><ul><li>The frozen 24 cells are retrospective development evidence.</li><li>RAGTruth pages are exploratory because labels were already opened.</li><li>HLE is intentionally absent until the paper's GPT-4o grader is reproduced.</li><li>The complete Qwen2.5-72B critic is a ProcessBench protocol reproduction with a different critic model.</li></ul></section>
    """
    return _page("Paper-aligned hallucination benchmark suite", body, registry)


def score_all(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = out / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    partial_path = out / "benchmark_scores.partial.csv"
    progress_path = out / "score_progress.json"
    detection, competitors = build_detection_rows()
    if partial_path.exists() and progress_path.exists():
        all_rows = read_csv(partial_path)
        progress = json.loads(progress_path.read_text())
        diagnostics = {}
        for name in progress.get("completed", []):
            diag_path = diagnostics_dir / f"{name}.json"
            if diag_path.exists():
                diagnostics[name] = json.loads(diag_path.read_text())
    else:
        all_rows, diagnostics = list(detection), {}
        progress = {"version": VERSION, "completed": ["frozen_detection"], "current": None}
        write_csv(partial_path, all_rows)

    stages = [
        ("semgrad", score_semgrad_detection),
        ("ragtruth_detection", score_ragtruth_detection),
        ("ragtruth_spans", score_ragtruth_spans),
        ("gasp_sentences", score_gasp_sentences),
        ("refchecker_claims", score_refchecker_claims),
        ("prmbench_steps", score_prmbench_steps),
        ("processbench_first_error", score_processbench_first_error),
    ]
    write_json(out / "score_progress.json", progress)
    for name, function in stages:
        if name in progress.get("completed", []):
            print(f"SKIP {name} (checkpoint complete)", flush=True)
            continue
        print(f"SCORE {name}", flush=True)
        progress["current"] = name
        write_json(out / "score_progress.json", progress)
        rows, stage_diag = function(out)
        all_rows.extend(rows)
        diagnostics[name] = stage_diag
        write_json(diagnostics_dir / f"{name}.json", stage_diag)
        write_csv(out / "benchmark_scores.partial.csv", all_rows)
        progress["completed"].append(name); progress["current"] = None
        write_json(out / "score_progress.json", progress)

    # Acceptance checks that operate on the combined artifact.
    critic_rows = [row for row in all_rows if row.get("method_key") == "qwen72b_critic"]
    if len(critic_rows) != 5:
        raise AssertionError("complete ProcessBench critic must have four subsets plus one macro")
    forbid_cross_task_macro(all_rows)
    prm_diag = diagnostics["prmbench_steps"]
    if sorted(prm_diag["excluded_ids"]) != sorted(BAD_PRM_IDS):
        raise AssertionError("PRMBench exclusions are not exactly the three registered IDs")
    write_csv(out / "benchmark_scores.csv", all_rows)
    registry = build_registry(all_rows, competitors)
    write_json(out / "protocol_registry.json", registry)
    (out / "benchmark_scores.partial.csv").unlink(missing_ok=True)
    progress["status"] = "complete"; write_json(out / "score_progress.json", progress)


def report_all(out: Path) -> None:
    score_path = out / "benchmark_scores.csv"
    registry_path = out / "protocol_registry.json"
    if not score_path.exists() or not registry_path.exists():
        raise FileNotFoundError("run the score command before report")
    rows = read_csv(score_path)
    registry = json.loads(registry_path.read_text())
    forbid_cross_task_macro(rows)
    out.joinpath("index.html").write_text(render_index(registry, rows), encoding="utf-8")
    for entry in registry:
        out.joinpath(f"{entry['protocol_id']}.html").write_text(
            render_protocol(entry, rows, registry), encoding="utf-8"
        )
    files = [out / "index.html", out / "benchmark_scores.csv", out / "protocol_registry.json", out / "score_progress.json"]
    files.extend(out / f"{entry['protocol_id']}.html" for entry in registry)
    files.extend(sorted((out / "diagnostics").glob("*.json")))
    if (out / "REVIEW_GUIDE.md").exists():
        files.append(out / "REVIEW_GUIDE.md")
    manifest = {
        "version": VERSION, "generated_from_machine_readable_rows": True,
        "n_protocols": len(registry), "n_score_rows": len(rows),
        "cross_task_macro": False, "algorithms_modified": False,
        "hles_included": False, "complete_processbench_critic_included": True,
        "method_aliases": {"IO-PCR": "IU-PCR"},
        "files": {str(path.relative_to(ROOT)): _sha256(path) for path in files},
    }
    write_json(out / "suite_manifest.json", manifest)
    print(out / "index.html")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("score", "report", "all"), nargs="?", default="all")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    if args.command in {"score", "all"}:
        score_all(out)
    if args.command in {"report", "all"}:
        report_all(out)


if __name__ == "__main__":
    main()
