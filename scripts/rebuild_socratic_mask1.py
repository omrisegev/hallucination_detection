"""Rebuild the Socratic Qwen3-8B mask1 cell after the 2026-10-11 first_token_mask fix (already-empty source
steps, kept as (k, k) in the external records, stay empty instead of raising). The first build (job 280515)
dropped 3 records for that reason; it is kept, renamed, as *_mask1_superseded_emptystep."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_mask1_cells import ext_cell  # noqa: E402

X = Path(sys.argv[1])
ext_cell(X / 'socratic_qwen3_8b_full_thinkfix_153589e', X / 'socratic_qwen3_8b_full_thinkfix_153589e_mask1')
print('SOCRATIC_MASK1_REBUILT', flush=True)
