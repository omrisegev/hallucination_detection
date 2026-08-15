from __future__ import annotations

import hashlib
import unittest

from spectral_utils.a6_s0b_input import (
    ALL_PATHS,
    FrozenFile,
    REPOSITORY,
    REVISION,
    SELECTED_FILES,
    canonical_json_bytes,
    git_blob_sha1,
    validate_official_tree,
    verify_selected_bytes,
)


UNSELECTED = {
    ".gitattributes": ("c7d9f3332a950355d5a77d85000f05e6f45435ea", 1477, None),
    "README.md": ("e5b8ac8b1d4fe74a360491bc01d160727a33c73d", 13691, None),
    "pytorch_model.bin": (
        "2555075e090b5ea81bccae072f8c38d2926baf56", 911_449_213,
        "8b8f211c55958bf39c52d8a5db2b208481beddc09673d614d2d9059a882d3e11",
    ),
}


def _official_tree() -> dict:
    rows = []
    selected = {item.path: item for item in SELECTED_FILES}
    for path in ALL_PATHS:
        if path in selected:
            item = selected[path]
            blob_id, size, lfs_sha = item.git_blob_sha1, item.size, item.lfs_sha256
        else:
            blob_id, size, lfs_sha = UNSELECTED[path]
        row = {"rfilename": path, "blobId": blob_id, "size": size}
        if lfs_sha is not None:
            row["lfs"] = {"sha256": lfs_sha, "size": size, "pointerSize": 134}
        rows.append(row)
    return {"id": REPOSITORY, "sha": REVISION, "gated": False, "siblings": rows}


class TestA6S0bInput(unittest.TestCase):
    def test_exact_official_projection(self) -> None:
        value = _official_tree()
        projection = validate_official_tree(canonical_json_bytes(value))
        self.assertEqual(projection["id"], REPOSITORY)
        self.assertEqual(projection["sha"], REVISION)
        self.assertEqual(len(projection["siblings"]), 8)

    def test_tree_tampering_closes(self) -> None:
        value = _official_tree()
        value["siblings"][2]["blobId"] = "0" * 40
        with self.assertRaisesRegex(RuntimeError, "official Pythia object changed"):
            validate_official_tree(canonical_json_bytes(value))
        value = _official_tree()
        value["siblings"].append(dict(value["siblings"][0]))
        with self.assertRaisesRegex(RuntimeError, "official path invalid"):
            validate_official_tree(canonical_json_bytes(value))

    def test_git_and_lfs_payload_verification(self) -> None:
        payload = b"authenticated-small-object"
        git_spec = FrozenFile(
            "small", len(payload), git_blob_sha1(payload), None,
        )
        self.assertEqual(verify_selected_bytes(git_spec, payload)["size"], len(payload))
        lfs_spec = FrozenFile(
            "large", len(payload),
            git_blob_sha1((
                "version https://git-lfs.github.com/spec/v1\n"
                f"oid sha256:{hashlib.sha256(payload).hexdigest()}\n"
                f"size {len(payload)}\n"
            ).encode("ascii")),
            hashlib.sha256(payload).hexdigest(),
        )
        self.assertEqual(verify_selected_bytes(lfs_spec, payload)["sha256"], lfs_spec.lfs_sha256)
        with self.assertRaisesRegex(RuntimeError, "payload size mismatch"):
            verify_selected_bytes(lfs_spec, payload + b"x")


if __name__ == "__main__":
    unittest.main()
