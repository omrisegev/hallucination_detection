"""ProcessBench with C4U under the NC rule (Omri, 2026-11-08), the same rule AUTO used in RESULTS_BACKBONE.csv.

NC (pb_metrics.nc_flags): gate = the PRM's native rule fires in the answer; k = number of steps the native rule
flags; flag the k steps with the highest C4U rank; prediction = the earliest flagged step. So the PRM decides
WHETHER an answer has an error and HOW MANY steps; C4U decides WHICH steps. C4U itself is the frozen recipe on R1
inputs (c4u_competitors.build_X / run_c4u, unchanged). Metrics: PRISM Table 6 per-subset F1 through
eval_processbench.evaluate (compute.processbench_core), and Mind the Gap SLA from the same NC flags.
Parity: PRM native average F1 and AUTO_N16_NC (stored) are read from RESULTS_BACKBONE.csv for comparison.

    COLLECTION_DATA=<hd_collection_data> python c4u_processbench_nc.py   # -> RESULTS_PB_NC.csv
"""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c4u_competitors as M  # noqa: E402
C, A, E = M.C, M.A, M.E
sys.path.insert(0, str(A.ROOT / 'results/prm_sub5_all_benchmarks_v1/vendor'))
import pb_metrics as PM  # noqa: E402


def main():
    C.MODE, C.TELE, C.CELL_HOOK = 'prm', ['C4U'], None
    for prm in M.N.PRMS:
        c = C.cab_cell('processbench', prm)
        X, ab, rep, g = M.build_X(c, prm, 'processbench'); assert rep == 'R1'
        M.run_c4u(c, X, ab, f'processbench/{prm}/{rep}')
        n0 = len(C.ROWS)
        S = np.load(A.cab_dir('prm', prm, 'processbench') / 'scores' / f'SCORES_prm_{A.PFILE.get(prm, prm)}_processbench.npz')
        lay = {k: S[k] for k in ('uids', 'nonempty', 'nonempty_lengths', 'native_valid')}
        R, _, _ = C.EP.evaluate(lay, {'PRM': c.arms['PRM'], 'C4U': c.arms['C4U']}, A.PB_GOLD, 'scored', A.PB_GOLD, False)
        R = {r['arm']: r for r in R}
        for name, label, rule in (('PRM_native', 'PRM native', 'native'), ('C4U_NC', 'C4U', 'NC')):
            r = R[name]
            for s in M.SUBSETS:
                for k, src in (('TPR', 'error_acc'), ('TNR', 'correct_acc'), ('F1', 'f1')):
                    C.add(prm, 'processbench', 'PRISM PB T6', label, f'{s} {k}', r[f'{s}_{src}'], rule=rule, input=rep)
            C.add(prm, 'processbench', 'PRISM PB T6', label, 'average F1', r['avg_f1'], rule=rule, input=rep)
        M.ref_gate(f'{prm} ProcessBench PRM native average F1 (EP) vs RESULTS_BACKBONE', R['PRM_native']['avg_f1'],
                   (prm, 'processbench', 'PRISM PB T6', 'PRM native', 'average F1', '', 'native', '', ''))
        # SLA from the same NC flags (all steps are non-empty on ProcessBench)
        ne = lay['nonempty'].astype(bool); assert ne.all()
        nat = np.zeros(len(ne), bool); nat[ne] = c.arms['PRM'] > 0           # eval_processbench PRM_native
        f, info = PM.nc_flags(lay['nonempty_lengths'], ne, nat, c.arms['C4U'])
        m = M.pb_metrics(c, M.pb_predict(c, f))
        assert abs(m['average F1'] - R['C4U_NC']['avg_f1']) < 1e-9, (m['average F1'], R['C4U_NC']['avg_f1'])
        for s in M.SUBSETS:
            C.add(prm, 'processbench', 'MtG T3 SLA', 'C4U', f'{s} SLA', m[s]['SLA'], rule='NC', input=rep)
        for k in ('macro SLA', 'overall SLA', 'macro SLA lo', 'macro SLA hi'):
            C.add(prm, 'processbench', 'MtG T3 SLA', 'C4U', k, m[k], rule='NC', input=rep)
        print(prm, 'C4U NC avg F1 %.2f  macro SLA %.2f  (gate %d answers, k0 %d)' % (R['C4U_NC']['avg_f1'], m['macro SLA'], info['n_gate'], info['n_k0']), flush=True)
        del c
    for r in C.ROWS:
        r.pop('standalone_model', None)
    E.write_csv(HERE / 'RESULTS_PB_NC.csv', C.ROWS)
    fails = [p['check'] for p in C.PAR if not p['ok']]
    print('rows', len(C.ROWS), 'parity', len(C.PAR), 'failed', fails)
    if fails: raise SystemExit(3)


if __name__ == '__main__':
    main()
