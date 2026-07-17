"""
Reference macros as first-class bench rows (Step 186).

The hand-curated subsets we compare against — GOOD_5, GOOD_6, STABLE_H9,
top_macro_5, consensus_4, ALL_H16 — emitted through the SAME select→fuse→score
path as every learned selector, so they land in comparison.csv with identical
metrics (exact percentile-within-size, per-domain macro, paired Δ). This is
the apples-to-apples baseline the researcher reads the learned methods against;
it is not a "selector" (the subsets are fixed), just a convenient way to score
the fixed macros on the same basis.

Macro definitions mirror their canonical homes:
  GOOD_5, GOOD_6, STABLE_H9  — spectral_utils/subset_sweep.py
  top_macro_5, consensus_4   — scripts/score_repgrid.py (repgrid scoring path)
GOOD_6 is emitted only where `varentropy` exists (AIRCC-era repgrid c46 cells);
elsewhere it would silently collapse to GOOD_5, so it is skipped.
"""

import numpy as np

from ..subset_sweep import GOOD_5, GOOD_6, STABLE_H9, H16
from . import register

TOP_MACRO_5 = ['epr', 'spectral_entropy', 'hl_ratio', 'sw_var_peak', 'cusum_max']
CONSENSUS_4 = ['spectral_entropy', 'sw_var_peak', 'cusum_max', 'cusum_shift_idx']

MACROS = {
    'ref.GOOD_5': GOOD_5,
    'ref.GOOD_6': GOOD_6,
    'ref.STABLE_H9': STABLE_H9,
    'ref.top_macro_5': TOP_MACRO_5,
    'ref.consensus_4': CONSENSUS_4,
    'ref.ALL_H16': H16,
}


@register('reference_macros')
def reference_macros(cell, rng, cache=None):
    out = []
    for variant, names in MACROS.items():
        cols = [cell.pool.index(f) for f in names if f in cell.pool]
        if len(cols) < 3:
            continue                     # not scorable on this cell's pool
        if variant == 'ref.GOOD_6' and 'varentropy' not in cell.pool:
            continue                     # would collapse to GOOD_5 — skip
        out.append({'variant': variant, 'cols': np.sort(cols),
                    'diag': {'macro': variant.split('.', 1)[1],
                             'n_present': len(cols)}})
    return out


def smoke():
    import sys
    sys.path.insert(0, __file__.rsplit('spectral_utils', 1)[0] + 'scripts')
    from smoke_selectors import _tiny_ctx
    from ..selector_bench import UnlabeledCell

    ctx = _tiny_ctx()                    # full H16 pool present
    cell = UnlabeledCell.from_context(ctx)
    sels = reference_macros(cell, np.random.default_rng(0))
    names = {d['variant'] for d in sels}
    # every H16-only macro present; GOOD_6 skipped (no varentropy in H16)
    assert 'ref.GOOD_5' in names and 'ref.STABLE_H9' in names
    assert 'ref.consensus_4' in names and 'ref.ALL_H16' in names
    assert 'ref.GOOD_6' not in names, "GOOD_6 must skip when varentropy absent"
    g5 = next(d for d in sels if d['variant'] == 'ref.GOOD_5')
    assert [cell.pool[j] for j in g5['cols']] == sorted(GOOD_5, key=cell.pool.index)
    for d in sels:
        c = np.asarray(d['cols'])
        assert len(c) >= 3 and len(set(c.tolist())) == len(c)
    print(f"    [note] reference_macros smoke: {sorted(names)}")
