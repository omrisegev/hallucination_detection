"""Run ONCE on the PC (needs hd_collection_data). Exports compact PRMBench per-step arrays of the STANDALONE
base-model cells (E.load('prmbench', '<prm>__standalone'), the loader of telemetry_only_vs_competitors_v1/compute.prmbench) for
test 2 (first-error localization on wrong answers). No fitting, no labels used except to be stored for scoring.
Output: cells/PRMB_<prm>.npz (ids, off, err, arm__<name> for every 1-D per-step arm) + EXPORT_MANIFEST.json.

    COLLECTION_DATA=<hd_collection_data> python -B results/rl_step_scoring_tests_v1/export_prmbench_sa.py
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'c4u_vs_competitors_v1'))
import c4u_competitors as M  # noqa: E402
C = M.C
OUT = HERE / 'cells'; OUT.mkdir(exist_ok=True)
man = {}
for prm in M.N.PRMS:
    c = C.E.load('prmbench', f'{prm}__standalone')      # same loader as compute.prmbench (ABL5 standalone cells)
    n = c.off[-1]
    d = dict(ids=np.asarray(c.ids).astype(str), off=np.asarray(c.off, np.int64), err=np.asarray(c.err, bool),
             cls=np.asarray(c.cls).astype(str), cat=np.asarray(c.cat).astype(str),
             **{f'arm__{k}': np.asarray(v, np.float64) for k, v in c.arms.items()
                if isinstance(v, np.ndarray) and v.ndim == 1 and len(v) == n})
    p = OUT / f'PRMB_{prm}.npz'; np.savez_compressed(p, **d)
    man[p.name] = dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(), answers=len(d['ids']), steps=int(n),
                       arms=sorted(k for k in d if k.startswith('arm__')))
    print(prm, len(d['ids']), n, man[p.name]['arms'], flush=True)
(OUT / 'EXPORT_MANIFEST.json').write_text(json.dumps(man, indent=1))
