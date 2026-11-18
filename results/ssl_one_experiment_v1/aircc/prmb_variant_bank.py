"""Id mapping around the UNCHANGED PRMBench bank builder, for the V1/V2 runs (RC_HANDOFF D.3). No labels are read.

The frozen builder (prm_standalone_telemetry_v1/code/build_bank.py) writes into the rows of the mask1 source bank and
requires EVERY PRMBench answer id of that bank (cell prmbench_qwen3_8b) in the raw pkl, with the same step count.
A variant row '<parent>__v1' has the parent's step count, but a new id, and some parents have no variant (skips).
  prepare: copy the variant pkl with idx '<parent>__vX' -> '<parent>'; parents without a variant row get the row of
           the stored standalone run (FILLER, same model; listed in MAP.json, never used: extract_steps.py reads only
           the variant uids). Then run the builder unchanged on the prepared pkl.
  rename:  in the builder's FEATURES.npz, set row_ids of the variant parents back to '<parent>__vX'. Values untouched.
    python prmb_variant_bank.py prepare VARIANT.pkl FILL.pkl BASE_FEATURES.npz OUT.pkl MAP.json
    python prmb_variant_bank.py rename BUILT_FEATURES.npz MAP.json OUT_FEATURES.npz
"""
import json, os, pickle, sys

import numpy as np

CELL = 'prmbench_qwen3_8b'


def prepare(var, fill, base, out, mp):
    V = pickle.load(open(var, 'rb'))
    B = np.load(base); cells = B['cells'].astype(str); rows = B['row_ids'].astype(str)
    need = [rows[i] for i in np.flatnonzero(cells == CELL)]
    prep, var_map, failed = {}, {}, []
    for e in V.values():
        u = str(e['idx']); p, sep, tag = u.rpartition('__')
        if not sep or tag not in ('v1', 'v2'): raise SystemExit('unexpected variant uid ' + u)
        if 'failure' in e: failed.append(u); continue
        r = dict(e); r['idx'] = p; prep[p] = r; var_map[p] = u
    F = pickle.load(open(fill, 'rb')); F = {str(e['idx']): e for e in F.values()}
    filler = []
    for p in need:
        if p not in prep:
            prep[p] = F[p]; filler.append(p)
    out_d = {i: prep[k] for i, k in enumerate(sorted(prep))}
    with open(out + '.tmp', 'wb') as f: pickle.dump(out_d, f)
    os.replace(out + '.tmp', out)
    json.dump(dict(variant_rows=len(var_map), failed_variant_rows=failed, filler_parents=filler, n_filler=len(filler),
                   base_answers=len(need), parent_to_variant=var_map), open(mp, 'w'), indent=1)
    print('PREPARE variant', len(var_map), 'filler', len(filler), 'failed', len(failed), flush=True)


def rename(built, mp, out):
    m = json.load(open(mp))['parent_to_variant']
    B = dict(np.load(built)); cells = B['cells'].astype(str); rows = B['row_ids'].astype(str).astype(object)
    n = 0
    for i in np.flatnonzero(cells == CELL):
        if rows[i] in m: rows[i] = m[rows[i]]; n += 1
    B['row_ids'] = rows.astype(str)
    np.savez_compressed(out, **B)
    print('RENAME', n, 'rows', flush=True)
    if n != len(m): raise SystemExit(f'renamed {n} of {len(m)}')


if __name__ == '__main__':
    {'prepare': prepare, 'rename': rename}[sys.argv[1]](*sys.argv[2:])
