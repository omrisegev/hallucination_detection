#!/usr/bin/env python3
"""Pure known-answer tests for the ProcessBench latent-state adapter."""

from __future__ import annotations

import inspect
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.processbench_latent_state_v1 import run as experiment  # noqa: E402
from scripts.processbench_latent_state_v1 import report as report_builder  # noqa: E402


def fake_row(row_id="x", length=8, label=-1):
    top = np.tile(np.linspace(-0.1, -5.0, 50), (length, 1))
    return {
        "id": row_id,
        "problem": "must not enter fit",
        "steps": ["a", "b"],
        "label": label,
        "final_answer_correct": label == -1,
        "step_token_spans": [(0, length // 2), (length // 2, length)],
        "align_diag": {"problems": []},
        "gen_token_ids": list(range(length)),
        "token_entropies": np.linspace(0.1, 1.0, length),
        "token_spilled_energies": np.linspace(1.0, 0.1, length),
        "token_logsumexp": np.linspace(2.0, 2.5, length),
        "top_k_logprobs": {
            "ids": np.tile(np.arange(50), (length, 1)),
            "logprobs": top,
        },
    }


def test_telemetry_view_removes_every_evaluation_field():
    raw = [(0, fake_row("a", label=1)), (1, fake_row("b", label=-1))]
    telemetry, row_ids = experiment.telemetry_view(raw)
    assert row_ids == ["a", "b"]
    assert all(set(row) == set(experiment.TELEMETRY_KEYS) for row in telemetry)
    assert all(experiment.FORBIDDEN_FIT_KEYS.isdisjoint(row) for row in telemetry)


def test_curve_flattening_round_trip():
    curves = [np.array([1.0, 2.0]), np.array([3.0]), np.array([4.0, 5.0, 6.0])]
    flat, offsets = experiment.flatten_curves(curves)
    rebuilt = experiment.unflatten_curves(flat, offsets)
    assert np.array_equal(offsets, [0, 2, 3, 6])
    assert all(np.array_equal(left, right) for left, right in zip(curves, rebuilt))


def fake_frozen_arrays():
    row_ids = np.asarray(["a", "b"])
    offsets = np.asarray([0, 3, 7], dtype=np.int64)
    arrays = {
        "row_ids": row_ids,
        "offsets": offsets,
        "global_mixed_v2_dufs": np.asarray([0.1, 0.8]),
        "mindgap_detector": np.asarray([0.2, 0.7]),
        "mindgap_evidence": np.linspace(-1, 1, 7),
    }
    for index, method in enumerate(experiment.LOCAL_METHODS):
        arrays[method] = np.linspace(index, index + 1, 7)
        arrays[method + "__token_locator"] = np.asarray([1, 2], dtype=np.int64)
    hashes = {
        "row_ids": experiment._hash_text(row_ids),
        "offsets": experiment._hash_int(offsets),
        "global_mixed_v2_dufs": experiment._hash_float(arrays["global_mixed_v2_dufs"]),
        "mindgap_detector": experiment._hash_float(arrays["mindgap_detector"]),
        "mindgap_evidence": experiment._hash_float(arrays["mindgap_evidence"]),
    }
    for method in experiment.LOCAL_METHODS:
        hashes[method] = experiment._hash_float(arrays[method])
        hashes[method + "__token_locator"] = experiment._hash_int(
            arrays[method + "__token_locator"]
        )
    return arrays, hashes


def test_frozen_hash_verification_catches_tampering():
    arrays, hashes = fake_frozen_arrays()
    with tempfile.TemporaryDirectory() as folder:
        scores, diagnostic = experiment.write_frozen_cell(folder, "qwen3_4b", "gsm8k", arrays, {
            "score_hashes_before_evaluation": hashes,
        })
        manifest_entry = {
            "model": "qwen3_4b",
            "subset": "gsm8k",
            "scores": str(scores.relative_to(folder)),
            "scores_file_sha256": experiment.sha256_file(scores),
            "diagnostics": str(diagnostic.relative_to(folder)),
            "diagnostics_file_sha256": experiment.sha256_file(diagnostic),
            "score_hashes": hashes,
        }
        loaded, _ = experiment.load_and_verify_frozen(
            folder, "qwen3_4b", "gsm8k", manifest_entry
        )
        assert np.array_equal(loaded["local_iu_core"], arrays["local_iu_core"])

        score_path = Path(folder) / "label_free_scores" / "qwen3_4b__gsm8k.npz"
        with np.load(score_path, allow_pickle=False) as archive:
            changed = {key: archive[key] for key in archive.files}
        changed["local_iu_core"] = changed["local_iu_core"].copy()
        changed["local_iu_core"][0] += 1.0
        np.savez_compressed(score_path, **changed)
        try:
            # Re-bless the changed array inside the mutable diagnostic. The
            # independent manifest must still reject the changed NPZ file.
            updated_hashes = dict(hashes)
            updated_hashes["local_iu_core"] = experiment._hash_float(
                changed["local_iu_core"]
            )
            experiment._write_json(diagnostic, {
                "score_hashes_before_evaluation": updated_hashes,
            })
            experiment.load_and_verify_frozen(
                folder, "qwen3_4b", "gsm8k", manifest_entry
            )
        except RuntimeError as error:
            assert "file hash changed" in str(error)
        else:
            raise AssertionError("tampered score artifact passed verification")


def test_evaluation_payload_is_the_only_label_opening_boundary():
    raw = [(0, fake_row("a", label=1)), (1, fake_row("b", label=-1))]
    labels, spans, row_ids = experiment._open_evaluation_payload(raw, ["a", "b"])
    assert np.array_equal(labels, [1, -1])
    assert spans[0] == [(0, 4), (4, 8)]
    assert row_ids == ["a", "b"]
    signature = inspect.signature(experiment.fit_cell).parameters
    assert "raw_rows" not in signature
    assert "labels" not in signature and "step_token_spans" not in signature
    telemetry, row_ids = experiment.telemetry_view(raw)
    contaminated = [dict(telemetry[0], label=1), telemetry[1]]
    try:
        experiment.fit_cell("qwen3_4b", "gsm8k", contaminated, row_ids, ".")
    except RuntimeError as error:
        assert "non-canonical telemetry" in str(error)
    else:
        raise AssertionError("fit_cell accepted a label-bearing row")


def test_split_replay_matches_canonical_protocol():
    labels = np.asarray([-1, -1, -1, 0, 0, 1, 1, 1] * 6, dtype=int)
    risk = np.linspace(-1.0, 1.0, len(labels))
    locator = np.where(labels == -1, 0, labels)
    canonical = experiment.evaluate_two_stage(
        risk, locator, labels,
        n_splits=experiment.N_SPLITS,
        seed=experiment.SPLIT_SEED,
    )
    rows = experiment._split_metrics(risk, locator, labels, "known")
    for key in ("f1", "acc_erroneous", "acc_correct", "sla", "sla_tol1"):
        assert np.isclose(np.mean([row[key] for row in rows]), canonical[key], atol=1e-14)


def test_report_is_generated_from_machine_readable_artifacts():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        evaluation = root / "evaluation"
        experiment._write_json(evaluation / "EVALUATION_MANIFEST.json", {"ok": True})
        systems = []
        for model, subset in sorted(report_builder.EXPECTED_CELLS):
            split = "development" if (model, subset) in experiment.DEV else "nonselection"
            for index, method in enumerate(report_builder.ORDER):
                systems.append({
                    "model": model, "subset": subset, "split": split,
                    "system": method, "f1": 0.25 + 0.005 * index,
                    "acc_erroneous": 0.2, "acc_correct": 0.5,
                    "sla": 0.2, "sla_tol1": 0.5,
                })
        components = []
        for model, subset in sorted(report_builder.EXPECTED_CELLS):
            split = "development" if (model, subset) in experiment.DEV else "nonselection"
            for index, method in enumerate(report_builder.LOCAL_ORDER):
                components.append({
                    "model": model, "subset": subset, "split": split,
                    "candidate": method, "exact": 0.2 + 0.005 * index,
                    "tol1": 0.5, "mean_signed_step_error": 0.1,
                    "mean_normalized_token_distance": 0.2,
                })
        split_rows = []
        for model, subset in sorted(report_builder.EXPECTED_CELLS):
            for split_index in range(100):
                for method in report_builder.ORDER:
                    split_rows.append({
                        "model": model, "subset": subset,
                        "split_index": split_index, "method": method,
                        "f1": 0.31 if method == "global_dufs__hmm_reversible" else 0.30,
                    })
        predictions = []
        for model, subset in sorted(report_builder.EXPECTED_CELLS):
            for method in report_builder.LOCAL_ORDER:
                for row_index in range(2):
                    token = "" if method == "mindgap_locator" else 10 + row_index
                    predictions.append({
                        "model": model, "subset": subset, "row_id": row_index,
                        "candidate": method, "gold_step": 1,
                        "predicted_step": row_index % 3,
                        "predicted_token": token, "trace_tokens": 80 + row_index,
                    })
        aligned = []
        for model, subset in sorted(report_builder.EXPECTED_CELLS):
            for method in (
                "local_hmm_reversible_core_iu",
                "local_hmm_absorbing_core_iu",
            ):
                for offset in range(-50, 51):
                    aligned.append({
                        "model": model, "subset": subset, "candidate": method,
                        "curve_kind": "posterior_state_entry_probability",
                        "relative_token": offset, "mean_entry_probability": 0.1,
                        "sum_entry_probability": 1.0, "n": 10,
                    })
        experiment._write_csv(evaluation / "systems_per_cell.csv", systems)
        experiment._write_csv(evaluation / "components_per_cell.csv", components)
        experiment._write_csv(evaluation / "split_metrics.csv", split_rows)
        experiment._write_csv(evaluation / "localization_rows.csv", predictions)
        experiment._write_csv(evaluation / "error_aligned_entry.csv", aligned)
        for model, subset in sorted(report_builder.EXPECTED_CELLS):
            diag = {"model": model, "subset": subset, "hmm": {}}
            for kind in ("reversible", "absorbing"):
                diag["hmm"][kind] = {
                    "fallback": False,
                    "selected": {
                        "separation": 1.2,
                        "transition": [[0.9, 0.1], [0.2, 0.8]],
                        "occupancy": [0.7, 0.3], "means": [-0.5, 0.8],
                        "variance": 0.4,
                    },
                    "fit": {
                        "mean_pair_exact_argmax_agreement": 0.8,
                        "mean_pair_normalized_argmax_displacement": 0.05,
                    },
                    "apply": {
                        "mean_peak_entry_probability": 0.5,
                        "mean_normalized_entry_entropy": 0.4,
                        "fraction_without_entry_above_0p10": 0.1,
                        "mean_normalized_entry_position": 0.5,
                    },
                }
            experiment._write_json(
                root / "label_free_diagnostics" / f"{model}__{subset}.json", diag
            )
        original_verify = report_builder.verify_artifacts
        report_builder.verify_artifacts = lambda _: None
        try:
            report_builder.build_report(root)
        finally:
            report_builder.verify_artifacts = original_verify
        assert (root / "REPORT.md").exists()
        assert (root / "REPORT.html").exists()
        assert (root / "figures" / "hmm_diagnostics.png").exists()
        html_text = (root / "REPORT.html").read_text()
        assert "<table>" in html_text and "<pre>" not in html_text


def main():
    tests = [
        test_telemetry_view_removes_every_evaluation_field,
        test_curve_flattening_round_trip,
        test_frozen_hash_verification_catches_tampering,
        test_evaluation_payload_is_the_only_label_opening_boundary,
        test_split_replay_matches_canonical_protocol,
        test_report_is_generated_from_machine_readable_artifacts,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"processbench_latent_state_v1: {len(tests)} tests passed")


if __name__ == "__main__":
    main()
