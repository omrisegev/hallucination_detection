"""Local driver for results/prm_boost_external_v1/build_bank_external.py (unchanged; byte-identical to the
build_bank_external_v2.py that built AIRCC bank_v3, origin/claude/integration-2026-10-17).

Only change: no Qwen tokenizer.json is available on this machine, so resolve_digit_ids is replaced by the frozen
Qwen digit ids 15..24 (the AIRCC bank_v3 MANIFEST records matches_qwen_15_24: true for this tokenizer).

Run from the repo root (the builder imports scripts/rebuild_source_bank_v4.py from cwd), with python -I:
    python -I -B results/answer_mixture_gate_v1/trackT/build_bank_local.py --raw <external_telemetry.pkl> \
        --mask-code <repo root> --names-from <any 56-channel FEATURES.npz> --tokenizer NONE --out <dir> --workers 12
No labels are read.
"""
import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'results/prm_boost_external_v1'))
import build_bank_external as B  # noqa: E402

QWEN = (15, 16, 17, 18, 19, 20, 21, 22, 23, 24)


def _fixed(tokenizer_json, expected=None):
    return QWEN, {'tokenizer_json': None, 'digit_ids': list(QWEN), 'matches_qwen_15_24': True,
                  'note': 'frozen Qwen ids; no tokenizer.json locally (answer_mixture_gate_v1 driver)'}


if __name__ == '__main__':
    B.DT.resolve_digit_ids = _fixed
    B.main()
