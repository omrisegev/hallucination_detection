"""Evaluate alternative step readouts of the 11 bank11 telemetry channels against the PRM (2026-11-11).

Inputs: steps/<bench>__<prm>.npz from extract_readouts.py (kept outside git; path via READOUT_STEPS) and the diagnosis
cells ../cells/<bench>__<prm>.npz (labels, keep mask, PRM risk). Labels are used only for evaluation.

Per readout, every channel is answer-z-scored inside each answer (all steps of the answer), as the stored banks are.
Extra reference readout `top10_lenres`: top10 answer-z with a pooled, label-free linear removal of answer-z log(token
count). Composites: avg11 = plain mean of the 11 answer-z channels with the bank's fixed risk signs (already applied to
the token values); avg3 = mean of q15_H1, chosen_surprisal, logprob_margin.

Outputs: T_gate.csv, T_cells.csv (every bench x PRM x readout x signal), T_bench.csv (mean over the 4 PRMs).
"""
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import analyze as A  # noqa: E402

STEPS = Path(os.environ.get('READOUT_STEPS', HERE / 'steps'))
BUNDLE = Path(os.environ.get('COLLECTION_DATA', 'C:/Users/omris/TAU/hd_collection_data')) / 'c4u_r1_inputs/auto_raw_plus_z_bundle/inputs'
BENCHES = ['prmbench', 'socratic', 'hard2verify', 'processbench']
MAIN = ['q15_H1', 'chosen_surprisal', 'logprob_margin']


def answer_z(x, off):
    out = np.empty_like(x, dtype=float)
    for a, b in zip(off[:-1], off[1:]):
        v = x[a:b]; s = v.std()
        out[a:b] = (v - v.mean()) / s if s > 1e-8 else 0.
    return out


def align(d, S):
    """Rows of the readout file in cell step order."""
    pos = {u: i for i, u in enumerate(S['uids'].astype(str))}
    so, ne = S['offsets'], S['nonempty']; rows = []
    for u, a, b in zip(d['ids'], d['off'][:-1], d['off'][1:]):
        i = pos[u]; r = np.arange(so[i], so[i + 1])
        if len(r) != b - a:
            r = r[ne[r]]
        if len(r) != b - a:
            raise SystemExit(f'step mismatch {u}: cell {b - a}, readout {len(r)}')
        rows.append(r)
    return np.concatenate(rows)


def pair_auc(err, s, off, keep):
    """(a) first error vs correct before it, (b) first error vs correct after it, (c) later errors vs correct after the
    first error, (d) later errors vs correct before the first error. Pair-weighted P(error > correct), ties 1/2."""
    acc = {k: [0., 0] for k in 'abcd'}
    def add(k, e, c):
        if len(e) and len(c):
            m = e[:, None] - c[None, :]; acc[k][0] += (m > 0).sum() + .5 * (m == 0).sum(); acc[k][1] += m.size
    for a, b in zip(off[:-1], off[1:]):
        k = np.flatnonzero(keep[a:b]) + a; e = err[k]
        if not e.any(): continue
        f = np.argmax(e); fe = s[k[f]:k[f] + 1]
        before = s[k[:f]]; after_c = s[k[f + 1:][~e[f + 1:]]]; later_e = s[k[f + 1:][e[f + 1:]]]
        add('a', fe, before); add('b', fe, after_c); add('c', later_e, after_c); add('d', later_e, before)
    return {f'pair_{k}': (v[0] / v[1] if v[1] else np.nan) for k, v in acc.items()} | {f'pairs_n_{k}': v[1] for k, v in acc.items()}


def metrics(d, s):
    err, off, keep = d['err'], d['off'], d['keep']
    last = np.zeros(len(err), bool); last[off[1:] - 1] = True
    q = A.arm_quality(d, s)
    use, y = A.first_error_view(err, off, keep); si = np.minimum(A.step_index(off), A.POSCAP)
    r = dict(within=q['within'], pooled=q['pooled'], fe_strat=q['fe_strat'],
             within_nolast=float(np.nanmean(A.within(err, s, off, keep & ~last))),
             pooled_nolast=A.auc(err[keep & ~last], s[keep & ~last]),
             last_level_correct=float(s[last & keep & ~err].mean()) if (last & keep & ~err).any() else np.nan,
             last_level_wrong=float(s[last & keep & err].mean()) if (last & keep & err).any() else np.nan,
             nonlast_level_correct=float(s[~last & keep & ~err].mean()),
             nonlast_level_wrong=float(s[~last & keep & err].mean()) if (~last & keep & err).any() else np.nan)
    r.update(pair_auc(err, s, off, keep))
    return r


def main():
    gate, rows = [], []
    for b in BENCHES:
        for p in A.PRMS:
            f = STEPS / f'{b}__{p}.npz'
            if not f.exists():
                print('MISSING', b, p); continue
            d = A.load(b, p); S = np.load(f)
            ix = align(d, S); R = S['readouts'][ix]; names = list(S['channel_names'].astype(str))
            rn = list(S['readout_names'].astype(str)); ntok = S['ntok'][ix].astype(float); off = d['off']
            # ---- gate: current top-10 readout vs stored values
            top = R[:, :, 0]; g = dict(bench=b, prm=p, steps=len(ix))
            if b == 'processbench':
                g['vs'] = 'cell raw16 (sub5 bank_v3 raw)'; g['max_abs_diff'] = float(np.abs(top - d['raw16'][:, :11]).max())
            elif b == 'prmbench':
                B = np.load(BUNDLE / f'INPUTS_{p}.npz'); assert np.array_equal(B['row_ids'].astype(str), d['ids'])
                assert list(B['raw17_names'][1:12].astype(str)) == names
                g['vs'] = 'R1 bundle raw17'; g['max_abs_diff'] = float(np.abs(top - B['raw17'][:, 1:12]).max())
            else:
                Zs = d['X'][:, 2:13].astype(float); Zm = np.column_stack([answer_z(top[:, j], off) for j in range(11)])
                g['vs'] = 'stored answer-z N16 (float32); raw bank_v2 gated in extract'
                g['max_abs_diff'] = float(np.abs(Zm - Zs).max())
                g['extract_gate_raw_max_abs_diff'] = __import__('json').loads(str(S['info']))['gate']['max_abs_diff_all']
            gate.append(g); print('GATE', g, flush=True)
            # ---- readouts
            reads = {r: np.column_stack([answer_z(R[:, j, k], off) for j in range(11)]) for k, r in enumerate(rn)}
            lz = answer_z(np.log(ntok), off)
            Z0 = reads['top10']; L = np.column_stack([np.ones(len(lz)), lz])
            reads['top10_lenres'] = Z0 - L @ np.linalg.lstsq(L, Z0, rcond=None)[0]
            for r, Z in reads.items():
                sig = {n: Z[:, j] for j, n in enumerate(names) if n in MAIN}
                sig['avg11'] = Z.mean(1); sig['avg3'] = Z[:, [names.index(n) for n in MAIN]].mean(1)
                for sname, s in sig.items():
                    rows.append(dict(bench=b, prm=p, readout=r, signal=sname, **metrics(d, s)))
            rows.append(dict(bench=b, prm=p, readout='-', signal='PRM', **metrics(d, d['prm'])))
            rows.append(dict(bench=b, prm=p, readout='-', signal='PRM_answer_z', **metrics(d, d['X'][:, 0].astype(float))))
            rows.append(dict(bench=b, prm=p, readout='-', signal='TELE16_stored', **metrics(d, d['tele'])))
            print(b, p, 'done', flush=True)
    pd.DataFrame(gate).to_csv(HERE / 'T_gate.csv', index=False)
    T = pd.DataFrame(rows); T.to_csv(HERE / 'T_cells.csv', index=False)
    G = T.groupby(['bench', 'readout', 'signal'], sort=False).mean(numeric_only=True).reset_index()
    G.insert(3, 'n_prms', T.groupby(['bench', 'readout', 'signal'], sort=False).size().values)
    G.to_csv(HERE / 'T_bench.csv', index=False)


if __name__ == '__main__':
    main()
