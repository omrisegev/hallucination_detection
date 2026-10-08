"""Python audit guard for the frozen, trusted external fit worker.

The controller constructs a sanitized code/input capsule and passes this policy
to a fresh worker.  The worker installs the hook before importing any scientific
project module, and the controller rejects sticky violations.  This is a guard
for hash-frozen first-party Python and trusted native dependencies, not a
sandbox for hostile Python or native code; Python's documentation explicitly
warns that ``sys.addaudithook`` is not a malicious-code containment boundary.
Python also does not emit audit events for every metadata operation (including
``os.stat`` and ``os.readlink`` on the supported runtime), so this tier does
not claim filesystem-metadata opacity.  Its narrower guarantee is that the
frozen trusted worker receives no target-bearing files in its capsule, and
registered attempts to open such files are denied and made sticky.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

try:  # macOS exposes the opened path through F_GETPATH.
    import fcntl
except ImportError:  # pragma: no cover - Windows is not a scientific target.
    fcntl = None


POLICY_SCHEMA_VERSION = "reconstruction-external-fit-audit-policy-v2"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    import hashlib

    return hashlib.sha256(value).hexdigest()


def _absolute(value: str | os.PathLike[str]) -> str:
    return os.path.normcase(os.path.realpath(os.path.abspath(os.fspath(value))))


def _open_file_descriptor_path(descriptor: int) -> str | None:
    """Resolve a regular-file descriptor without trusting inherited handles."""

    if fcntl is not None and hasattr(fcntl, "F_GETPATH"):
        try:
            payload = fcntl.fcntl(descriptor, fcntl.F_GETPATH, b"\0" * 1024)
        except (OSError, ValueError):
            return None
        value = payload.split(b"\0", 1)[0]
        if not value:
            return None
        return _absolute(os.fsdecode(value))
    probe = Path(f"/proc/self/fd/{descriptor}")
    try:
        return _absolute(os.readlink(probe))
    except OSError:
        return None


_VIOLATIONS: list[dict[str, str]] = []
_PROBE_ACTIVE = False


def build_fit_audit_policy(
    *,
    allowed_read_roots: Sequence[str | Path],
    allowed_read_files: Sequence[str | Path],
    allowed_write_roots: Sequence[str | Path],
    forbidden_probes: Sequence[Mapping[str, str]],
    allowed_native_roots: Sequence[str | Path] | None = None,
) -> dict[str, Any]:
    if allowed_native_roots is None:
        # Existing lane-specific launchers get a narrow native-runtime default;
        # code and prepared-data roots are intentionally excluded.
        allowed_native_roots = (
            Path(sys.prefix),
            Path(sys.base_prefix),
            Path("/usr"),
            Path("/System"),
            Path("/Library"),
        )
    value: dict[str, Any] = {
        "schema_version": POLICY_SCHEMA_VERSION,
        "default_action": "deny",
        "network": "deny",
        "subprocess": "deny",
        "allowed_read_roots": sorted({_absolute(path) for path in allowed_read_roots}),
        "allowed_read_files": sorted({_absolute(path) for path in allowed_read_files}),
        "allowed_write_roots": sorted({_absolute(path) for path in allowed_write_roots}),
        "allowed_native_roots": sorted({_absolute(path) for path in allowed_native_roots}),
        "ctypes_policy": (
            "runtime_roots_or_process_handle_with_trusted_string_reads"
        ),
        "integer_fd_policy": "resolved_regular_files_only",
        "forbidden_probes": [
            {"probe_id": str(item["probe_id"]), "path": _absolute(item["path"])}
            for item in forbidden_probes
        ],
    }
    probe_ids = [item["probe_id"] for item in value["forbidden_probes"]]
    if not probe_ids or len(set(probe_ids)) != len(probe_ids):
        raise ValueError("fit audit policy requires unique denial probes")
    value["policy_sha256"] = _sha256(_canonical_bytes(value))
    return value


def validate_fit_audit_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(policy)
    recorded = value.pop("policy_sha256", None)
    if recorded != _sha256(_canonical_bytes(value)):
        raise RuntimeError("external fit audit-policy hash failed")
    if value.get("schema_version") != POLICY_SCHEMA_VERSION:
        raise RuntimeError("unexpected external fit audit-policy schema")
    if value.get("default_action") != "deny":
        raise RuntimeError("external fit audit policy is not deny-default")
    if value.get("network") != "deny" or value.get("subprocess") != "deny":
        raise RuntimeError("external fit audit policy permits network or subprocess")
    for key in (
        "allowed_read_roots", "allowed_read_files", "allowed_write_roots",
        "allowed_native_roots",
    ):
        paths = value.get(key)
        if not isinstance(paths, list) or not paths:
            raise RuntimeError(f"external fit audit policy lacks {key}")
        normalized = sorted({_absolute(path) for path in paths})
        if paths != normalized:
            raise RuntimeError(f"external fit audit-policy {key} is not canonical")
    probes = value.get("forbidden_probes")
    if not isinstance(probes, list) or not probes:
        raise RuntimeError("external fit audit policy lacks denial probes")
    value["policy_sha256"] = recorded
    return value


def install_fit_audit_hook(policy: Mapping[str, Any]) -> str:
    """Install the irreversible allowlist hook and return its policy hash."""

    import sys

    value = validate_fit_audit_policy(policy)
    read_roots = tuple(value["allowed_read_roots"])
    read_files = frozenset(value["allowed_read_files"])
    write_roots = tuple(value["allowed_write_roots"])
    native_roots = tuple(value["allowed_native_roots"])

    def deny(event: str, path: str = "") -> None:
        if not _PROBE_ACTIVE:
            token = _sha256(path.encode("utf-8")) if path else ""
            _VIOLATIONS.append({"event": event, "path_sha256": token})
        raise PermissionError(f"external fit firewall denied {event}")

    def under(path: str, roots: tuple[str, ...]) -> bool:
        for root in roots:
            try:
                if os.path.commonpath((path, root)) == root:
                    return True
            except ValueError:
                continue
        return False

    def hook(event: str, args: tuple[Any, ...]) -> None:
        if event in {
            "subprocess.Popen", "os.system", "os.posix_spawn", "os.fork",
            "os.forkpty", "os.exec", "os.execve",
        } or event.startswith("os.exec") or event.startswith("os.spawn"):
            deny(event)
        if event.startswith("socket."):
            deny(event)
        if event == "ctypes.dlopen":
            # NumPy calls PyDLL(None) while importing ctypes.  Some trusted
            # dependencies also open absolute libraries from the frozen
            # runtime.  Libraries beside project code or prepared data are not
            # native-allowlisted.
            library = args[0] if args else None
            if library is None:
                return
            try:
                library_path = _absolute(library)
            except (TypeError, ValueError):
                deny(event)
            if under(library_path, native_roots):
                return
            deny(event, library_path)
        if event in {"ctypes.dlsym", "ctypes.dlsym/handle"}:
            # Resolution is required after an allowed dlopen.  This is one
            # reason this tier assumes trusted code/native dependencies.
            return
        if event == "ctypes.string_at":
            # PyTorch's frozen native runtime reads strings returned by its own
            # process-local C APIs through this helper.  No controller secret,
            # raw source, or label sidecar is mapped into the worker process;
            # this remains within the documented trusted-native-dependency tier.
            return
        if event.startswith("ctypes."):
            deny(event)
        if event in {"os.link", "os.symlink"}:
            deny(event)
        if event in {"os.rename", "os.replace"}:
            if len(args) < 2:
                deny(event)
            source, destination = _absolute(args[0]), _absolute(args[1])
            if under(source, write_roots) and under(destination, write_roots):
                return
            deny(event, source + "\0" + destination)
        if event in {
            "os.remove", "os.unlink", "os.rmdir", "os.truncate", "os.chmod",
            "os.chown", "os.lchown", "os.utime",
        }:
            if args and under(_absolute(args[0]), write_roots):
                return
            deny(event, "" if not args else _absolute(args[0]))
        if event == "os.mkdir":
            if args and under(_absolute(args[0]), write_roots):
                return
            deny(event, "" if not args else _absolute(args[0]))
        if event in {"os.listdir", "os.scandir", "os.chdir"}:
            path = _absolute(args[0] if args and args[0] is not None else os.getcwd())
            if under(path, read_roots) or under(path, write_roots):
                return
            deny(event, path)
        if event != "open" or not args:
            return
        target = args[0]
        if isinstance(target, int):
            if target in {0, 1, 2}:
                return
            descriptor_path = _open_file_descriptor_path(target)
            if descriptor_path is None:
                deny("open_file_descriptor", str(target))
            if (
                descriptor_path in read_files
                or under(descriptor_path, read_roots)
                or under(descriptor_path, write_roots)
            ):
                return
            deny("open_file_descriptor", descriptor_path)
        try:
            path = _absolute(target)
        except (TypeError, ValueError):
            deny("open_non_path")
        mode = args[1] if len(args) > 1 else "r"
        flags = args[2] if len(args) > 2 else 0
        writing = False
        if isinstance(mode, str):
            writing = any(token in mode for token in ("w", "a", "x", "+"))
        if isinstance(flags, int):
            writing = writing or bool(
                flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
            )
        if writing:
            if under(path, write_roots):
                return
            deny("open_write", path)
        if path in read_files or under(path, read_roots) or under(path, write_roots):
            return
        deny("open_read", path)

    sys.addaudithook(hook)
    return str(value["policy_sha256"])


def run_forbidden_read_probes(policy: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Prove that every controller-registered sentinel is unreadable."""

    global _PROBE_ACTIVE

    value = validate_fit_audit_policy(policy)
    results: list[dict[str, Any]] = []
    for item in value["forbidden_probes"]:
        denied = False
        _PROBE_ACTIVE = True
        try:
            try:
                with open(item["path"], "rb") as handle:  # noqa: PTH123 - deliberate probe
                    handle.read(1)
            except PermissionError:
                denied = True
        finally:
            _PROBE_ACTIVE = False
        if not denied:
            raise RuntimeError(
                f"external fit denial probe unexpectedly readable: {item['probe_id']}"
            )
        results.append({"probe_id": item["probe_id"], "read_denied": True})
    return results


def fit_firewall_violations() -> tuple[dict[str, str], ...]:
    """Return sticky non-probe violations; a controller must reject any."""

    return tuple(dict(item) for item in _VIOLATIONS)


__all__ = [
    "POLICY_SCHEMA_VERSION",
    "build_fit_audit_policy",
    "fit_firewall_violations",
    "install_fit_audit_hook",
    "run_forbidden_read_probes",
    "validate_fit_audit_policy",
]
