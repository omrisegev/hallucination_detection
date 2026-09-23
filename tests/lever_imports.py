"""Import helper for the 2026-09-23 lever tests: make `spectral_utils.<module>` importable
without executing the package __init__ (which imports torch)."""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def ensure_spectral_package() -> Path:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if "spectral_utils" not in sys.modules:
        try:
            importlib.import_module("spectral_utils")
        except Exception:
            pkg = types.ModuleType("spectral_utils")
            pkg.__path__ = [str(ROOT / "spectral_utils")]
            sys.modules["spectral_utils"] = pkg
    return ROOT
