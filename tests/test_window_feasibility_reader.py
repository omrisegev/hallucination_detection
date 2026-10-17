import json
import zipfile

import numpy as np
import pytest

from scripts.per_answer_localization.feasibility import (
    cell_geometry, selected_traces, valid_checkpoint,
)


def test_streaming_reads_only_requested_telemetry_and_offsets(tmp_path, monkeypatch):
    raw = np.arange(384, dtype=np.float32).reshape(128, 3)
    path = tmp_path / "cell.npz"
    np.savez_compressed(path, raw=raw, token_offsets=[0, 32, 96, 128],
                        forbidden_labels=np.array([object()], dtype=object))
    original = zipfile.ZipFile.open
    opened = []

    def guarded(self, name, *args, **kwargs):
        actual = name.filename if isinstance(name, zipfile.ZipInfo) else name
        assert actual in {"raw.npy", "token_offsets.npy"}
        opened.append(actual)
        return original(self, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "open", guarded)
    offsets = cell_geometry(path)
    traces = list(selected_traces(path, offsets, [2, 0], ["a", "b", "c"]))
    assert [row for row, _ in traces] == [0, 2]
    np.testing.assert_array_equal(traces[0][1], raw[:32])
    np.testing.assert_array_equal(traces[1][1], raw[96:])
    assert set(opened) == {"raw.npy", "token_offsets.npy"}


def test_resume_rejects_wrong_configuration_or_answer(tmp_path):
    path = tmp_path / "checkpoint.json"
    assert not valid_checkpoint(path, "hash", "cell", 0)
    payload = {"config_sha256": "hash", "cell": "cell", "row": 0,
               "labels_accessed": False, "widths": []}
    path.write_text(json.dumps(payload))
    assert valid_checkpoint(path, "hash", "cell", 0)
    with pytest.raises(RuntimeError, match="checkpoint"):
        valid_checkpoint(path, "another_hash", "cell", 0)
    with pytest.raises(RuntimeError, match="checkpoint"):
        valid_checkpoint(path, "hash", "cell", 1)
