"""Export compact ProcessBench per-step arrays for the first-error-rule study (run on the PC, once).

For each of the 4 PRMs it loads the same CAB cell that c4u_processbench_nc.py uses (MODE 'prm'), builds R1 with
c4u_competitors.build_X, runs the frozen C4U (c4u_new.c4u, unchanged) and saves:
  ids, off, keep, keep_gold              cell layout (answer ids in cell order, step offsets)
  X_R1                                   R1 matrix (n_steps x 18, float32): col0 answer-z PRM risk, col1 raw PRM p std, 16 telemetry
  arm__<name>                            every per-step arm of the cell (PRM, PRMPROB, ...)
  N16                                    stored answer-z N16 channels (float32)
  G16, digit_active                      global rank->normal of the raw N16 channels (no per-answer centring)
  c4u_rank, c4u_post, c4u_flags, c4u_k   frozen C4U outputs (rank, DS step posterior, flags, per-answer count)
  lay__uids, lay__nonempty, lay__nonempty_lengths, lay__native_valid   SCORES layout used by eval_processbench
and, for MODE 'sa' (standalone base-model telemetry, no PRM), the same layout + N16 + arms in sa_*.
Output: results/pb_first_error_rule_v1/cells/PB_<prm>.npz (+ EXPORT_MANIFEST.json with sha256).

    COLLECTION_DATA=<hd_collection_data> python results/pb_first_error_rule_v1/export_pb_cells.py
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'c4u_vs_competitors_v1'))
import c4u_competitors as M  # noqa: E402
C, A, E = M.C, M.A, M.E
OUT = HERE / 'cells'; OUT.mkdir(exist_ok=True)


def arms_of(c):
    n = c.off[-1]
    return {f'arm__{k}': np.asarray(v, np.float64) for k, v in c.arms.items()
            if isinstance(v, np.ndarray) and v.ndim == 1 and len(v) == n}


def main():
    man = {}
    for prm in M.N.PRMS:
        C.MODE, C.TELE, C.CELL_HOOK = 'prm', ['C4U'], None
        c = C.cab_cell('processbench', prm)
        X, ab, rep, g = M.build_X(c, prm, 'processbench'); assert rep == 'R1', rep
        r = M.run_c4u(c, X, ab, f'processbench/{prm}/{rep}')
        rv, act = M.raw16(c, prm, 'processbench')       # raw N16 values -> global rank->normal (no answer-z): keeps level
        G16 = np.column_stack([M.rank_gauss(rv[:, j], act[:, M.DIG.index(n)] if n in M.DIG else None)
                               for j, n in enumerate(M.N16)])
        S = np.load(A.cab_dir('prm', prm, 'processbench') / 'scores' / f'SCORES_prm_{A.PFILE.get(prm, prm)}_processbench.npz')
        d = dict(ids=np.asarray(c.ids).astype(str), off=np.asarray(c.off, np.int64), keep=np.asarray(c.keep, bool),
                 keep_gold=np.asarray(c.keep_gold, bool), X_R1=X.astype(np.float32), G16=G16.astype(np.float32), digit_active=np.asarray(act, bool),
                 N16=E.channels(c, 'N16').astype(np.float32),
                 c4u_rank=r['rank'], c4u_post=r['posterior'], c4u_flags=np.asarray(r['flags'], bool),
                 c4u_k=np.asarray(r['k'], np.int64), c4u_prevalence=np.array(r['prevalence']), **arms_of(c))
        for k in ('uids', 'nonempty', 'nonempty_lengths', 'native_valid'):
            d[f'lay__{k}'] = S[k]
        del c
        try:
            C.MODE = 'sa'
            s = C.cab_cell('processbench', prm)
            d.update(sa_ids=np.asarray(s.ids).astype(str), sa_off=np.asarray(s.off, np.int64),
                     sa_N16=E.channels(s, 'N16').astype(np.float32), **{f'sa_{k}': v for k, v in arms_of(s).items()})
            del s
        except Exception as e:  # standalone cell is optional for this study
            print(prm, 'standalone export skipped:', repr(e), flush=True)
        p = OUT / f'PB_{prm}.npz'; np.savez_compressed(p, **d)
        man[p.name] = dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size,
                           keys=sorted(d), n_steps=int(d['off'][-1]), n_answers=len(d['ids']),
                           c4u_flag_rate=float(d['c4u_flags'].mean()), c4u_prevalence=float(d['c4u_prevalence']))
        print(prm, man[p.name]['n_answers'], man[p.name]['n_steps'], man[p.name]['bytes'], flush=True)
    (HERE / 'cells' / 'EXPORT_MANIFEST.json').write_text(json.dumps(man, indent=1))


if __name__ == '__main__':
    main()
