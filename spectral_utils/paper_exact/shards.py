"""
Sharded, atomic, append-only acquisition storage.

Handoff §3.3: shard at no more than 64 traces or 1 GB, whichever comes first; write a
shard atomically, then update INDEX.jsonl, STATUS.json and SHA-256 checksums; catch
SIGTERM, flush the active shard, and resume by stable trace key without duplicates.

The invariant that makes preemption safe: **a shard file, once in INDEX.jsonl, is never
rewritten.** Resume reads the index, collects the trace keys already stored, and skips
them. A crash between writing the shard and appending its index line loses at most the
active shard's traces, and the orphaned file is quarantined rather than silently reused —
if it were reused, a partially-written pickle would be indistinguishable from a complete
one and would corrupt every downstream count.

Per-trace records are dicts; see handoff §3.2 for the required key set, enforced by
`REQUIRED_TRACE_KEYS` when `strict_schema=True`.
"""
import glob
import json
import os
import pickle
import time
from datetime import datetime, timezone

from .manifest import sha256_file

MAX_TRACES_PER_SHARD = 64
MAX_BYTES_PER_SHARD = 1 << 30  # 1 GiB

#: Minimum per-trace schema (handoff §3.2). Acquisition-specific drivers add more.
REQUIRED_TRACE_KEYS = (
    "trace_key",       # stable, unique; the resume key
    "question_id",     # stable question/source ID (grouping unit for every bootstrap)
    "prompt_text",
    "prompt_token_ids",
    "gen_token_ids",
    "full_text",
)


class ShardWriter:
    """Append-only sharded writer with atomic shard commit and resume-by-key.

    Usage::

        w = ShardWriter(run_dir, expected_keys=all_keys)
        for key in w.pending():          # already-stored keys are skipped for you
            w.add(build_record(key))
            if w.stop_requested:          # set by the driver's SIGTERM handler
                break
        w.close()

    `close()` is idempotent and safe to call from a signal handler path.
    """

    def __init__(self, run_dir: str, expected_keys=None,
                 max_traces: int = MAX_TRACES_PER_SHARD,
                 max_bytes: int = MAX_BYTES_PER_SHARD,
                 strict_schema: bool = True):
        self.run_dir = run_dir
        self.shard_dir = os.path.join(run_dir, "shards")
        os.makedirs(self.shard_dir, exist_ok=True)
        self.index_path = os.path.join(run_dir, "INDEX.jsonl")
        self.status_path = os.path.join(run_dir, "STATUS.json")
        self.max_traces = int(max_traces)
        self.max_bytes = int(max_bytes)
        self.strict_schema = bool(strict_schema)
        self.expected_keys = list(expected_keys) if expected_keys is not None else None
        self.stop_requested = False

        self._buf = []
        self._buf_bytes = 0
        self._failed = []
        self._t0 = time.time()

        self._index = _read_index(self.index_path)
        self._done = set()
        for entry in self._index:
            self._done.update(entry["keys"])
        self._next_shard = (max((e["shard"] for e in self._index), default=-1)) + 1
        self._quarantine_orphans()

    # ── resume ───────────────────────────────────────────────────────────────────
    def done_keys(self) -> set:
        """Trace keys already committed to a shard."""
        return set(self._done)

    def pending(self) -> list:
        """Expected keys not yet committed, in the manifest's declared order."""
        if self.expected_keys is None:
            raise ValueError("pending() needs expected_keys")
        return [k for k in self.expected_keys if k not in self._done]

    def _quarantine_orphans(self):
        """Move any shard file not referenced by INDEX.jsonl out of the way.

        Such a file is the debris of a crash between `pickle.dump` and the index append.
        It may be truncated, so it must never be read back as data — but it may also hold
        the only copy of expensive traces, so it is preserved for forensic recovery
        instead of deleted.
        """
        referenced = {e["path"] for e in self._index}
        orphan_dir = os.path.join(self.run_dir, "quarantine")
        for path in sorted(glob.glob(os.path.join(self.shard_dir, "shard_*.pkl"))):
            rel = os.path.relpath(path, self.run_dir).replace(os.sep, "/")
            if rel in referenced:
                continue
            os.makedirs(orphan_dir, exist_ok=True)
            dest = os.path.join(orphan_dir, os.path.basename(path))
            os.replace(path, dest)
            print(f"[shards] quarantined unindexed shard -> {dest}", flush=True)

    # ── write path ───────────────────────────────────────────────────────────────
    def add(self, record: dict):
        """Buffer one trace record; flush automatically at the shard boundary."""
        if self.strict_schema:
            missing = [k for k in REQUIRED_TRACE_KEYS if k not in record]
            if missing:
                raise KeyError(f"trace record missing required keys {missing}")
        key = record["trace_key"]
        if key in self._done:
            raise KeyError(f"duplicate trace_key {key!r} — resume logic is broken")
        blob_size = len(pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL))
        self._buf.append(record)
        self._buf_bytes += blob_size
        if len(self._buf) >= self.max_traces or self._buf_bytes >= self.max_bytes:
            self.flush()

    def add_failure(self, trace_key: str, question_id, reason: str):
        """Record a trace that could not be produced, without pretending it succeeded.

        Failures are counted in STATUS.json and never silently dropped: a run whose
        finished count matches expectations only because failures vanished is exactly
        the kind of clean-looking corruption the acquisition gate exists to catch.
        """
        self._failed.append({"trace_key": trace_key, "question_id": question_id,
                             "reason": str(reason)[:2000]})
        self._write_status()

    def flush(self):
        """Commit the buffered traces as one shard: write, fsync, hash, index, status."""
        if not self._buf:
            return
        shard = self._next_shard
        name = f"shard_{shard:05d}.pkl"
        path = os.path.join(self.shard_dir, name)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(self._buf, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

        keys = [r["trace_key"] for r in self._buf]
        entry = {
            "shard": shard,
            "path": f"shards/{name}",
            "n_traces": len(self._buf),
            "bytes": os.path.getsize(path),
            "sha256": sha256_file(path),
            "keys": keys,
            "question_ids": sorted({str(r["question_id"]) for r in self._buf}),
            "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        with open(self.index_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
            f.flush()
            os.fsync(f.fileno())

        self._index.append(entry)
        self._done.update(keys)
        self._next_shard += 1
        self._buf, self._buf_bytes = [], 0
        self._write_status()
        print(f"[shards] committed {name}: {entry['n_traces']} traces, "
              f"{entry['bytes'] / 1e6:.1f} MB, total done={len(self._done)}", flush=True)

    def _write_status(self):
        status = {
            "run_dir": self.run_dir,
            "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_expected": len(self.expected_keys) if self.expected_keys is not None else None,
            "n_finished": len(self._done),
            "n_failed": len(self._failed),
            "n_shards": len(self._index),
            "bytes_total": sum(e["bytes"] for e in self._index),
            "elapsed_s": round(time.time() - self._t0, 1),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
            "failures": self._failed[-200:],
            "complete": (self.expected_keys is not None
                         and len(self._done) + len(self._failed) >= len(self.expected_keys)),
        }
        tmp = self.status_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(status, f, indent=2)
        os.replace(tmp, self.status_path)

    def close(self):
        self.flush()
        self._write_status()


def _read_index(index_path: str) -> list:
    """Read INDEX.jsonl, tolerating a torn final line from a kill mid-append."""
    if not os.path.exists(index_path):
        return []
    entries, torn = [], 0
    with open(index_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                torn += 1
    if torn:
        print(f"[shards] dropped {torn} torn INDEX.jsonl line(s) — their shards are "
              f"unindexed and will be quarantined", flush=True)
    return entries


def read_shards(run_dir: str, verify: bool = True):
    """Yield every committed trace record, in index order.

    `verify=True` re-hashes each shard first. That costs a full read of the acquisition,
    which is the point: an offline analysis that silently consumed a corrupted shard would
    put a wrong number in a table with no way to notice.
    """
    index = _read_index(os.path.join(run_dir, "INDEX.jsonl"))
    for entry in index:
        path = os.path.join(run_dir, entry["path"])
        if verify:
            got = sha256_file(path)
            if got != entry["sha256"]:
                raise ValueError(f"shard {entry['path']} sha256 mismatch: "
                                 f"index={entry['sha256']} disk={got}")
        with open(path, "rb") as f:
            for record in pickle.load(f):
                yield record


def verify_shards(run_dir: str) -> dict:
    """Full integrity check over a run directory. Returns a report dict."""
    index = _read_index(os.path.join(run_dir, "INDEX.jsonl"))
    problems, keys, n = [], set(), 0
    for entry in index:
        path = os.path.join(run_dir, entry["path"])
        if not os.path.exists(path):
            problems.append(f"missing shard file {entry['path']}")
            continue
        if sha256_file(path) != entry["sha256"]:
            problems.append(f"sha256 mismatch {entry['path']}")
        if os.path.getsize(path) != entry["bytes"]:
            problems.append(f"size mismatch {entry['path']}")
        dup = keys & set(entry["keys"])
        if dup:
            problems.append(f"duplicate trace keys in {entry['path']}: {sorted(dup)[:5]}")
        keys.update(entry["keys"])
        n += entry["n_traces"]
    return {
        "run_dir": run_dir,
        "n_shards": len(index),
        "n_traces": n,
        "n_unique_keys": len(keys),
        "bytes_total": sum(e.get("bytes", 0) for e in index),
        "problems": problems,
        "ok": not problems,
    }
