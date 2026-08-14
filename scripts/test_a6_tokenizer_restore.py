from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from spectral_utils import a6_tokenizer_restore as restore


class A6TokenizerRestoreTests(unittest.TestCase):
    def _official(self, spec: restore.RoleSpec) -> bytes:
        selected = {row.path: row for row in spec.selected}
        siblings = []
        for path in spec.all_paths:
            expected = selected.get(path)
            row = {
                "rfilename": path,
                "blobId": expected.blob_id if expected else "0" * 40,
                "size": expected.size if expected else 1,
            }
            if expected and expected.lfs_sha256:
                row["lfs"] = {
                    "sha256": expected.lfs_sha256, "size": expected.size,
                    "pointerSize": 128,
                }
            siblings.append(row)
        return json.dumps({
            "id": spec.repository, "sha": spec.revision,
            "gated": spec.gated, "siblings": siblings,
        }).encode("utf-8")

    def test_all_three_official_trees_and_allowlists_are_self_consistent(self) -> None:
        for role in restore.ROLE_ORDER:
            spec = restore.ROLE_SPECS[role]
            projection = restore.validate_official_tree(spec, self._official(spec))
            self.assertEqual(projection["id"], spec.repository)
            self.assertEqual(
                restore.selected_allowlist(spec.all_paths),
                tuple(sorted((row.path for row in spec.selected), key=str.encode)),
            )

    def test_official_tree_tampering_fails_closed(self) -> None:
        spec = restore.ROLE_SPECS["qwen3-4b"]
        value = json.loads(self._official(spec))
        value["siblings"][0]["rfilename"] = "unexpected.txt"
        with self.assertRaisesRegex(RuntimeError, "tree changed"):
            restore.validate_official_tree(spec, json.dumps(value).encode())

    def test_git_and_lfs_payload_authentication(self) -> None:
        payload = b"authenticated bytes"
        direct = restore.SelectedFile(
            "config.json", restore.git_blob_sha1(payload), len(payload),
            restore.sha256_bytes(payload), restore.git_blob_sha1(payload),
        )
        restore.verify_selected_bytes(direct, payload)
        with self.assertRaisesRegex(RuntimeError, "payload hash"):
            restore.verify_selected_bytes(direct, payload + b"x")
        pointer = (
            "version https://git-lfs.github.com/spec/v1\n"
            f"oid sha256:{restore.sha256_bytes(payload)}\nsize {len(payload)}\n"
        ).encode("ascii")
        lfs = restore.SelectedFile(
            "tokenizer.json", restore.git_blob_sha1(pointer), len(payload),
            restore.sha256_bytes(payload), restore.git_blob_sha1(pointer),
            lfs_sha256=restore.sha256_bytes(payload),
        )
        restore.verify_selected_bytes(lfs, payload)

    def test_full_restore_is_all_three_atomic_and_replayable(self) -> None:
        shared_names = (
            "generation_config.json", "merges.txt", "tokenizer.json",
            "tokenizer_config.json", "vocab.json",
        )
        payloads = {name: ("shared:" + name).encode() for name in shared_names}
        payloads["q4-config"] = b"q" * 726
        payloads["q8-config"] = b"q8 config"
        payloads["llama-config"] = b"llama config"

        def direct(path: str, source: str, payload: bytes, *, url=False):
            return restore.SelectedFile(
                path, restore.git_blob_sha1(payload), len(payload),
                restore.sha256_bytes(payload), restore.git_blob_sha1(payload),
                source_url=source if url else None,
                source_remote=None if url else source,
            )

        real_q4 = restore.ROLE_SPECS["qwen3-4b"]
        specs = {
            "qwen3-4b": restore.RoleSpec(
                "qwen3-4b", real_q4.repository, real_q4.revision, False,
                ("config.json", *shared_names),
                (
                    direct("config.json", "https://test/q4-config", payloads["q4-config"], url=True),
                    *(direct(name, "remote:" + name, payloads[name]) for name in shared_names),
                ),
            ),
            "qwen3-8b": restore.RoleSpec(
                "qwen3-8b", "Qwen/Qwen3-8B", "b" * 40, False,
                ("config.json", *shared_names),
                (
                    direct("config.json", "remote:q8-config", payloads["q8-config"]),
                    *(direct(name, "remote:" + name, payloads[name]) for name in shared_names),
                ),
            ),
            "llama31-8b": restore.RoleSpec(
                "llama31-8b", "meta-llama/Llama-3.1-8B-Instruct", "c" * 40,
                "manual", ("config.json",),
                (direct("config.json", "remote:llama-config", payloads["llama-config"]),),
            ),
        }

        def fake_http(url: str, *, accept: str):
            for spec in specs.values():
                if url == restore._official_api_url(spec):
                    return self._official(spec), {"content-type": "application/json"}
            self.assertEqual(url, "https://test/q4-config")
            return payloads["q4-config"], {
                "etag": '"e49eccdc32f36da9c09cfa0e737084f9e0105e5e"',
                "x-repo-commit": real_q4.revision,
                "content-type": "text/plain", "content-length": "726",
            }

        remote_calls = []

        def fake_remote(remote: str, *, rclone: str):
            remote_calls.append(remote)
            return payloads[remote.removeprefix("remote:")]

        with tempfile.TemporaryDirectory() as temporary, patch.object(
            restore, "ROLE_SPECS", specs,
        ), patch.object(restore, "https_get", side_effect=fake_http), patch.object(
            restore, "rclone_cat", side_effect=fake_remote,
        ):
            out = Path(temporary) / "restored"
            manifest = restore.restore_all_three(out)
            self.assertEqual(
                restore.load_and_verify_restore(out), manifest,
            )
            self.assertFalse((Path(temporary) / ".restored.staging").exists())
            self.assertFalse((Path(temporary) / ".restored.publish.lock").exists())
            stage = Path(temporary) / ".restored.staging"
            out.rename(stage)
            (stage / "CACHE_RESTORE_PROVENANCE.json").unlink()
            (stage / "materialized" / "llama31-8b" / "config.json").unlink()
            remote_calls.clear()
            resumed = restore.restore_all_three(out)
            self.assertEqual(resumed["materialized_sha256"], manifest["materialized_sha256"])
            self.assertEqual(remote_calls, ["remote:llama-config"])
            out.rename(stage)
            remote_calls.clear()
            completed_resume = restore.restore_all_three(out)
            self.assertEqual(completed_resume, resumed)
            self.assertEqual(remote_calls, [])
            with self.assertRaises(FileExistsError):
                restore.restore_all_three(out)
            q4_config = out / "materialized" / "qwen3-4b" / "config.json"
            q4_config.write_bytes(b"bad")
            with self.assertRaisesRegex(RuntimeError, "tree changed"):
                restore.load_and_verify_restore(out)
            q4_config.write_bytes(payloads["q4-config"])
            empty = out / "materialized" / "qwen3-4b" / "empty"
            empty.mkdir()
            with self.assertRaisesRegex(RuntimeError, "unmanifested directory"):
                restore.load_and_verify_restore(out)
            empty.rmdir()
            (out / "unexpected").write_bytes(b"x")
            with self.assertRaisesRegex(RuntimeError, "unmanifested"):
                restore.load_and_verify_restore(out)


if __name__ == "__main__":
    unittest.main()
