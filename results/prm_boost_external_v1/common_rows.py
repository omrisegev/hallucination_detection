"""Restrict every SCORES_<prm>.npz to the answers all PRMs could read (no context overflow).
python common_rows.py <scores dir> <out dir>"""
import sys, json
from pathlib import Path
import numpy as np
src, dst = Path(sys.argv[1]), Path(sys.argv[2]); dst.mkdir(parents=True, exist_ok=True)
S = {f.stem[7:]: dict(np.load(f)) for f in sorted(src.glob('SCORES_*.npz'))}
common = set.intersection(*[set(map(str, s['uids'])) for s in S.values()])
info = {'n_common': len(common), 'per_prm_n': {k: int(len(s['uids'])) for k, s in S.items()}}
for k, s in S.items():
    uids = np.array(list(map(str, s['uids']))); lens = s['nonempty_lengths']; ne = s['nonempty'].astype(bool)
    st = np.r_[0, np.cumsum(lens)]; keep_ans = np.array([u in common for u in uids])
    step_mask = np.repeat(keep_ans, lens)
    ne_idx = np.flatnonzero(ne); ne_keep = step_mask[ne_idx]
    out = {'uids': uids[keep_ans], 'nonempty': ne[step_mask], 'nonempty_lengths': lens[keep_ans],
           'native_valid': s['native_valid'][step_mask]}
    for key in s:
        if key.startswith('score__'):
            v = s[key]; assert len(v) == ne.sum(), (k, key)
            out[key] = v[ne_keep]
    np.savez_compressed(dst / f'SCORES_{k}.npz', **out)
(dst / 'COMMON_ROWS.json').write_text(json.dumps(info, indent=1)); print(info)
