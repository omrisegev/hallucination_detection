"""Atomic per-answer checkpoints, pinned run identities and prediction seals."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile
from .contracts import digest


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class RecordStore:
    def __init__(self, directory, identity):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.identity = digest(identity)
        self.manifest = self.directory / "RUN.json"
        self.lock = self.directory / "WRITER.lock"
        self.lock_fd = None

    def __enter__(self):
        self.lock_fd = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(self.lock_fd, str(os.getpid()).encode())
        try:
            if self.manifest.exists():
                if json.loads(self.manifest.read_text(encoding="utf8"))["identity"] != self.identity:
                    raise ValueError("resume identity changed")
            else:
                atomic_json(self.manifest, {"identity": self.identity})
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *args):
        os.close(self.lock_fd)
        self.lock_fd = None
        self.lock.unlink()

    def path(self, uid):
        return self.directory / (digest(uid) + ".record.json")

    def get(self, uid):
        path = self.path(uid)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf8"))
        if data["uid"] != uid or data["run_identity"] != self.identity:
            raise ValueError("corrupt checkpoint identity")
        return data["payload"]

    def put(self, uid, payload):
        if self.lock_fd is None:
            raise RuntimeError("write requires exclusive writer context")
        previous = self.get(uid)
        if previous is not None:
            if digest(previous) != digest(payload):
                raise ValueError("attempt to overwrite completed record")
            return
        atomic_json(self.path(uid), {"uid": uid, "run_identity": self.identity, "payload": payload})


def seal_predictions(predictions, expected_uids, expected_arms, path, lock_hash):
    if set(predictions) != set(expected_uids) or len(expected_uids) != len(set(expected_uids)):
        raise ValueError("missing, extra or duplicate answers")
    for row in predictions.values():
        if set(row["arms"]) != set(expected_arms):
            raise ValueError("incomplete arm accounting")
    payload = {"prediction_sha256": digest(predictions), "lock_sha256": lock_hash,
               "answers": len(predictions), "arms": sorted(expected_arms)}
    path = Path(path)
    if path.exists() and json.loads(path.read_text()) != payload:
        raise ValueError("sealed predictions changed")
    atomic_json(path, payload)
    return payload


def require_seal(predictions, seal, lock_hash):
    if digest(predictions) != seal["prediction_sha256"] or lock_hash != seal["lock_sha256"]:
        raise ValueError("unsealed/changed predictions or protocol")
