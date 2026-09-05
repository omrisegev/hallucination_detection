import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from spectral_utils.joint_lsml_integrity import (
    IntegrityError, RECORD_NAME, contained_file, create_late_record,
    expected_run_files, file_sha256, relative_manifest, verify_prelabel_record,
)
from spectral_utils.trajectory_reducer import center_within_answers


def fixture_run(tmp_path):
    root = tmp_path / "results"
    producer = tmp_path / "producer"
    audit = producer / "scripts/joint_lsml_optimization_v2/audit_prelabel.py"
    audit.parent.mkdir(parents=True)
    audit.write_text("# fixture audit source\n", encoding="utf-8")
    for name in expected_run_files(["cell"], 1, 2, True):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    (root / "AUDIT_PRELABEL_RECEIPT.json").write_text(json.dumps({
        "status": "PASS", "checks": [{"passed": True}],
        "audit_source_sha256": file_sha256(audit),
    }), encoding="utf-8")
    amendment = tmp_path / "amendment.md"
    amendment.write_text("Late record, not proof of an original launch freeze.\n", encoding="utf-8")
    return root, producer, amendment


def freeze(root, producer, amendment):
    return create_late_record(root, ["cell"], producer_root=producer, amendment=amendment,
                              n_outer=1, n_inner=2)


def verify(root):
    return verify_prelabel_record(root, ["cell"], n_outer=1, n_inner=2)


def test_every_repeated_inner_basename_is_bound_and_checked(tmp_path):
    root, producer, amendment = fixture_run(tmp_path)
    record = freeze(root, producer, amendment)
    assert verify(root)["files"] == record["files"]
    first = "structure/cell/outer0/inner0/scores_inner.npz"
    second = "structure/cell/outer0/inner1/scores_inner.npz"
    assert record["files"][first] != record["files"][second]
    (root / first).write_bytes(b"changed earliest inner fold")
    with pytest.raises(IntegrityError, match="inner0"):
        verify(root)


def test_incomplete_r1_cannot_receive_a_record(tmp_path):
    root, producer, amendment = fixture_run(tmp_path)
    (root / "structure/cell/outer0/moduleb_grid.npz").unlink()
    with pytest.raises(IntegrityError, match="missing"):
        freeze(root, producer, amendment)
    assert not (root / RECORD_NAME).exists()


def test_record_cannot_silently_omit_an_inner_fold(tmp_path):
    root, producer, amendment = fixture_run(tmp_path)
    record = freeze(root, producer, amendment)
    del record["files"]["structure/cell/outer0/inner0/meta_inner.json"]
    (root / RECORD_NAME).write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(IntegrityError, match="exact required"):
        verify(root)


def test_failed_audit_is_not_frozen(tmp_path):
    root, producer, amendment = fixture_run(tmp_path)
    (root / "AUDIT_PRELABEL_RECEIPT.json").write_text('{"status":"FAIL"}', encoding="utf-8")
    with pytest.raises(IntegrityError, match="audit receipt"):
        freeze(root, producer, amendment)
    assert not (root / RECORD_NAME).exists()


def test_existing_evaluation_and_record_are_not_overwritten(tmp_path):
    root, producer, amendment = fixture_run(tmp_path)
    freeze(root, producer, amendment)
    before = (root / RECORD_NAME).read_bytes()
    with pytest.raises(IntegrityError, match="never overwrite"):
        freeze(root, producer, amendment)
    assert before == (root / RECORD_NAME).read_bytes()


def test_started_evaluation_blocks_a_late_prelabel_record(tmp_path):
    root, producer, amendment = fixture_run(tmp_path)
    (root / "evaluation").mkdir()
    with pytest.raises(IntegrityError, match="evaluation directory"):
        freeze(root, producer, amendment)


def test_manifest_rejects_path_escape_and_preserves_duplicate_names(tmp_path):
    for sub in ("inner0", "inner1"):
        (tmp_path / sub).mkdir()
        (tmp_path / sub / "scores.npz").write_bytes(sub.encode())
    assert set(relative_manifest(tmp_path)) == {"inner0/scores.npz", "inner1/scores.npz"}
    with pytest.raises(IntegrityError, match="unsafe"):
        contained_file(tmp_path, "../elsewhere")


def test_within_answer_centering_removes_answer_offsets_not_step_differences():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(12, 4))
    owners = np.repeat([0, 1, 2], 4)
    shifted = x + np.asarray([[100, 200, -5, 8], [-20, 90, 14, 0], [30, 60, 7, 1]])[owners]
    actual = center_within_answers(shifted, owners)
    np.testing.assert_allclose(actual, center_within_answers(x, owners), atol=1e-12)
    for owner in np.unique(owners):
        np.testing.assert_allclose(actual[owners == owner].mean(axis=0), 0, atol=1e-12)
    assert np.linalg.norm(actual) > 0


def test_evaluator_refuses_labels_before_any_preflight(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts/joint_lsml_optimization_v2/evaluate_v2.py"
    spec = importlib.util.spec_from_file_location("guarded_v2_evaluator", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "OUT", tmp_path)
    with pytest.raises(IntegrityError, match="barred"):
        module._labels("cell")
    with pytest.raises(IntegrityError, match="must not load labels"):
        module.main([])
    assert not (tmp_path / "evaluation").exists()


def test_additive_centering_uses_only_training_answers_and_preserves_saved_scores(tmp_path, monkeypatch):
    from scripts.joint_lsml_optimization_v2 import repair_centered_diagnostic as repair

    root = tmp_path / "results"
    cell = "pb_gsm8k_q4"
    folder = root / "structure" / cell / "outer0"
    folder.mkdir(parents=True)
    (root / "folds").mkdir()
    (root / "cells").mkdir()
    (folder / "COMPLETE.json").write_text("{}")
    (root / "folds/folds.json").write_text(json.dumps({
        "processbench": {"outer": {"a": 1, "b": 1, "held": 0}},
    }))
    np.savez(root / "cells" / f"{cell}.npz", group_ids=["a", "b", "held"])
    owners = np.repeat([0, 1, 2], 60)
    values = np.random.default_rng(4).normal(size=(180, 10)).astype(np.float32)
    values += (1000 * owners[:, None]).astype(np.float32)
    path = folder / "moduleb.npz"
    np.savez(path, orderstats=values, lengths=np.full(180, 10), step_rows=owners,
             b1_scores=np.arange(180))
    before = path.read_bytes()
    observed = {}

    def fit(matrix, lengths):
        observed["matrix"] = matrix.copy()
        return np.ones(10), {"n_fit_steps": len(matrix)}

    monkeypatch.setattr(repair, "fit_orderstat_weights", fit)
    result = repair.repair_fold(root, cell, 0)
    assert result["status"] == "FITTED"
    assert observed["matrix"].shape == (120, 10)
    np.testing.assert_allclose(observed["matrix"][:60].mean(0), 0, atol=1e-12)
    np.testing.assert_allclose(observed["matrix"][60:].mean(0), 0, atol=1e-12)
    assert path.read_bytes() == before
    with pytest.raises(RuntimeError, match="existing correction"):
        repair.repair_fold(root, cell, 0)
