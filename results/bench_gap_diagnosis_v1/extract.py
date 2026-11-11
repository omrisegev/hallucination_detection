"""Extract step-level arrays for the benchmark-gap diagnosis (2026-11-11).

For every (benchmark, PRM) cell in PRM-backbone mode: labels, keep mask, answer offsets, PRM risk / probability,
TELE16 score, the frozen C4U rank and flags (c4u_new.c4u, unchanged), the 18 C4U input columns, and the raw 16
telemetry channels when a raw bank is available. Cells are built with exactly the code of
results/c4u_tele16_all_benchmarks_v1/run_all.py (PRMBench/ProcessBench/CROP/DeltaBench) and c4u_new.load
(Socratic, R0); Hard2Verify uses the Socratic construction (R0) since the frozen recipe has no H2V cell.

    COLLECTION_DATA=C:/Users/omris/TAU/hd_collection_data python extract.py   -> cells/<bench>__<prm>.npz
No labels are used to build any score; labels are only stored for the diagnosis.
"""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'c4u_tele16_all_benchmarks_v1'))
import run_all as R  # noqa: E402
T, N, E, C, A = R.T, R.N, R.E, R.C, R.A
OUT = HERE / 'cells'; OUT.mkdir(exist_ok=True)
BENCHES = ['prmbench', 'processbench', 'crv_arith', 'crv_gsm8k', 'prm800k', 'deltabench_ext', 'socratic', 'hard2verify']
R.MODE = 'prm'; C.MODE = 'prm'


def cell(bench, prm):
    raw = None
    if bench == 'prmbench':
        c, ans_eval, X, ab, wk = N.load('prmbench', prm, 'R1')
        cat = c.cat; keep = c.keep        # raw PRM-backbone bank for PRMBench is not on this PC (rank-normal only)
    elif bench in ('socratic', 'hard2verify'):
        c = E.load(bench, prm); c.groups = np.array([u.split(':')[-1] for u in c.ids])
        Z = E.channels(c, 'N16'); zp = E.answer_z(c.arms[N.K.prm_arm(c)], c.off); rp = c.arms['PRMPROB']
        X = np.column_stack([zp, (rp - rp.mean()) / rp.std(), Z]); ab = {1: rp > .5}
        cat = c.cat; keep = c.keep
    else:
        c = C.cab_cell(bench, prm)
        X, ab, rep, g = T.build_X(c, prm, bench)
        raw, _ = T.raw16(c, prm, bench)
        keep = c.keep if hasattr(c, 'keep') and c.keep is not None else np.ones(c.off[-1], bool)
        if bench == 'processbench':
            cat = np.array([A.PB_GOLD[u]['subset'] for u in c.ids])
        else:
            cat = np.array([''] * len(c.ids))
    r = N.c4u(c, X, ab)
    prm_risk = c.arms['PRM'] if 'PRM' in c.arms else c.arms[N.K.prm_arm(c)]
    d = dict(err=c.err.astype(bool), keep=np.asarray(keep, bool), off=c.off, ids=c.ids.astype(str), cat=np.asarray(cat).astype(str),
             prm=prm_risk.astype(float), prmprob=(c.arms['PRMPROB'] if 'PRMPROB' in c.arms else 1. - c.extra['reward']).astype(float), tele=c.arms['TELE_N16'].astype(float),
             c4u=r['rank'], c4u_flags=r['flags'].astype(bool), tele_flags=c.flags('TELE_N16', 'GAP').astype(bool),
             prm_native=np.asarray(c.native_flag, bool) if getattr(c, 'native_flag', None) is not None else prm_risk > 0,
             X=X.astype(np.float32), K=len(r['groups']), prevalence=r['prevalence'],
             groups=np.array([','.join(map(str, g)) for g in r['groups']]))
    if raw is not None:
        d['raw16'] = raw.astype(np.float32)
    np.savez_compressed(OUT / f'{bench}__{prm}.npz', **d)
    print(bench, prm, len(c.ids), c.off[-1], 'err', round(float(c.err.mean()), 3), 'K', d['K'], 'raw', raw is not None, flush=True)


if __name__ == '__main__':
    todo = sys.argv[1:] or BENCHES
    for b in todo:
        for p in N.PRMS:
            if (OUT / f'{b}__{p}.npz').exists():
                continue
            try:
                cell(b, p)
            except Exception as ex:  # report and continue; a missing cell is listed in the handoff
                print('FAIL', b, p, repr(ex), flush=True)
