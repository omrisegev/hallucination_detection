"""Arm D: GSP-T (no PRM) with the hybrid step readout of the 11 token channels (2026-11-12).

Definition (same as bench_gap_diagnosis_v1/readout_alternatives/evaluate_hybrid.py, there on PRM-backbone telemetry):
for each of the 11 token channels of TELE16 (q15_H1 .. bocpd_p0), a step with fewer than X tokens takes the median of
its token values, every other step keeps the current top-10 readout (mean of the 10 highest token values). Both
readouts go through the same column construction as the frozen R1 input (global rank -> normal over the cell, then
answer-z), and the hybrid column takes the median column on short steps and the top-10 column on the other steps.
The 5 other channels (realized_z, realized_drv, 3 digit channels) are step statistics, not token readouts: unchanged.
X is one fixed cutoff for all benchmarks and PRMs. Main X = 10, sensitivity X = 30. No label is read.

Input: READOUT_STEPS/<bench>__<prm>.npz from extract_readouts.py run on the STANDALONE raw telemetry (PC, 2026-11-12).
Gate: the top-10 column rebuilt from the readout file must equal the stored R1 column (max abs diff < 1e-4).
Output: cells/<bench>__<prm>_hyb<X>.npz and RESULTS_hybrid.csv (GSP-T own count and GSP-T + GAP for each X).
"""
import json, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ablation as AB  # noqa: E402
E, T, Rn = AB.E, AB.T, AB.Rn
STEPS = Path(os.environ.get('READOUT_STEPS', Path(E.DATA) / 'readout_steps_sa_v1'))
CUTS = (10, 30)
TOK11 = T.N16[:11]


def align(c, S, bench=None, prm=None):
    if bench in ('socratic', 'hard2verify'):
        # same position rule as bracha_panel.sa_X: the cell keeps the PRM's nonempty steps (full position k);
        # the readout file holds one row per full step position of the standalone bank
        P = np.load(Path(E.DATA) / {'socratic': 'socratic', 'hard2verify': 'h2v'}[bench] / f'SCORES_{T.PFILE.get(prm, prm)}.npz')
        pne = P['nonempty'].astype(bool); pst = np.concatenate([[0], np.cumsum(P['nonempty_lengths'].astype(np.int64))])
        ppos = {u: j for j, u in enumerate(P['uids'].astype(str))}
        spos = {u: i for i, u in enumerate(S['uids'].astype(str))}; so, sne = S['offsets'], S['nonempty'].astype(bool)
        rows = []
        for u in c.ids:
            a, b = so[spos[u]], pst[ppos[u]]
            assert so[spos[u] + 1] - a == pst[ppos[u] + 1] - b, u
            for k in range(pst[ppos[u] + 1] - b):
                if pne[b + k]:
                    assert sne[a + k], (u, k); rows.append(a + k)
        rows = np.array(rows); assert len(rows) == c.off[-1]
        return rows
    pos = {u: i for i, u in enumerate(S['uids'].astype(str))}
    so, ne = S['offsets'], S['nonempty'].astype(bool); rows = []
    for u, a, b in zip(c.ids, c.off[:-1], c.off[1:]):
        r = np.arange(so[pos[u]], so[pos[u] + 1])
        if len(r) != b - a:
            r = r[ne[r]]
        assert len(r) == b - a, (u, len(r), b - a)
        rows.append(r)
    return np.concatenate(rows)


def column(raw, off):
    return E.answer_z(T.rank_gauss(raw), off)


def main():
    rows, gate = [], {}
    for bench in AB.BENCHES:
        for prm in AB.PRMS:
            p = STEPS / f'{bench}__{prm}.npz'
            if not p.exists():
                print('missing', p, flush=True); continue
            c, X, ab = AB.load_cell(bench, prm)
            S = np.load(p); ix = align(c, S, bench, prm)
            names = list(S['channel_names'].astype(str)); rn = list(S['readout_names'].astype(str))
            R = S['readouts'][ix]; ntok = S['ntok'][ix]
            top = np.column_stack([column(R[:, names.index(n), rn.index('top10')], c.off) for n in TOK11])
            med = np.column_stack([column(R[:, names.index(n), rn.index('median')], c.off) for n in TOK11])
            d = float(np.abs(top - X[:, 2:13]).max())
            gate[f'{bench}/{prm}'] = dict(max_abs_diff_top10_vs_R1=d, short_lt10=float((ntok < 10).mean()),
                                          short_lt30=float((ntok < 30).mean()), status='PASS' if d < 1e-4 else 'FAIL')
            print(bench, prm, gate[f'{bench}/{prm}'], flush=True)
            if d >= 1e-4:
                continue
            for cut in CUTS:
                Xt = X[:, 2:].copy()
                Xt[:, :11] = np.where((ntok < cut)[:, None], med, top)
                tel = Rn.c4u_t(c, Xt)
                A = {f'GSP-T hyb{cut} own count': (tel['rank'], tel['flags']),
                     f'GSP-T hyb{cut} + GAP': (tel['rank'], AB.gap(c, tel['rank']))}
                save = dict(ids=np.asarray(c.ids).astype(str), off=np.asarray(c.off))
                for k, (r, f) in A.items():
                    for mk, mv in AB.score(c, bench, r, f).items():
                        rows.append(dict(bench=bench, prm=prm, arm=k, metric=mk, value=mv))
                    save[f'rank|{k}'] = r; save[f'flags|{k}'] = f
                np.savez_compressed(HERE / 'cells' / f'{bench}__{prm}_hyb{cut}.npz', **save)
                gate[f'{bench}/{prm}'][f'hyb{cut}_prevalence'] = tel['prevalence']
            del c
    E.write_csv(HERE / 'RESULTS_hybrid.csv', rows)
    json.dump(gate, open(HERE / 'HYBRID_GATE.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
