"""Lightweight entrypoint avoiding optional model imports."""
from pathlib import Path
import sys
import types

root = Path(__file__).resolve().parents[1]
pkg = types.ModuleType('spectral_utils')
pkg.__path__ = [str(root / 'spectral_utils')]
sys.modules['spectral_utils'] = pkg
from spectral_utils.monday_workflow_guard import main

if __name__ == '__main__':
    main()
