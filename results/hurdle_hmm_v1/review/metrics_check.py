"""Independent macro-F1, P1/P2/P3 and verdict, compared with evaluate.py on SYNTHETIC cells only (no real label is
read). A fake bundle + fake fit dir is written to a temp folder, then evaluate.main runs on it.
Also tests what evaluate.py does when a secondary cell sits in the same fit folder."""
import importlib.util, json, pathlib, tempfile, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
HERE = pathlib.Path(__file__).resolve().parent
sp = importlib.util.spec_from_file_location('ev', HERE.parent / 'evaluate.py')
E = importlib.util.module_from_spec(sp); sp.loader.exec_module(E)


def my_macro_f1(y, f):
    y = np.asarray(y, bool); f = np.asarray(f, bool)
    out = []
    for pos in (True, False):
        t, p = (y == pos), (f == pos)
        tp = np.sum(t & p); fp = np.sum(~t & p); fn = np.sum(t & ~p)
        out.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(out))


rng = np.random.default_rng(3)
mx = 0
for _ in range(2000):
    n = rng.integers(1, 40); y = rng.random(n) < rng.random(); f = rng.random(n) < rng.random()
    mx = max(mx, abs(my_macro_f1(y, f) - E.macro_f1(y, f)))
print('macro-F1 max |diff| over 2000 random cases:', mx)

PRMS = ['qwen_prm_7b', 'qwen_prm_72b', 'pure_prm', 'rlhflow_prm']
cells = [f'AUTO_prm_{b}_{p}' for b in ('processbench', 'crv_arith', 'crv_gsm8k') for p in PRMS] + \
        [f'TELE16_sa_prmbench_{p}' for p in PRMS]


def make(tmp, cell, rng):
    A = 300; n = rng.integers(2, 10, A); off = np.concatenate([[0], np.cumsum(n)]); N = off[-1]
    ha = rng.random(A) < rng.uniform(.1, .9); y = np.zeros(N, bool)
    for a in np.where(ha)[0]:
        y[off[a] + rng.integers(0, n[a])] = True
    m = rng.random(N) < .9; ev = rng.random(A) < .95
    raw = rng.normal(size=A) + ha
    np.savez(tmp / 'b' / f'{cell}.npz', offsets=off, step_err_LABEL=y, step_mask=m, step_keep_eval=m, ans_evaluated=ev,
             ans_has_err_LABEL=ha, ans_raw_fused_answer_max=raw)
    pA = np.clip(ha * .6 + rng.random(A) * .5, 0, 1); py = rng.random(N)
    fb = py > .7; fk = py > .8; fg = rng.random(N) < .3
    pi_hat = ha.mean() + rng.normal(0, .05)
    np.savez(tmp / 'f' / f'{cell}.npz', pA=pA, py=py, flag_bayes=fb, flag_hhk=fk, flag_gap=fg,
             fit_json=json.dumps(dict(pi=pi_hat, delta=1., s0=1., f=.3, rho=.4, h=.1, e1=.1, converged=True)))
    return dict(cell=cell, P1=abs(pi_hat - ha.mean()), P2=my_macro_f1(y[m], fb[m]) - my_macro_f1(y[m], fg[m]),
                P3=roc_auc_score(ha[ev], pA[ev]) - roc_auc_score(ha[ev], raw[ev]))


def my_verdict(rows):
    d = pd.DataFrame(rows); d['bench'] = [[b for b in ('processbench', 'crv_arith', 'crv_gsm8k', 'prmbench') if b in c][0] for c in d.cell]
    v = {}
    for b, g in d.groupby('bench'):
        v[b] = dict(P1_pass=(g.P1 <= .05).sum() >= 3, P2_pass=(g.P2 > 0).sum() >= 3, P3_pass=(g.P3 >= -.02).sum() >= 3,
                    both=((g.P1 <= .05).sum() >= 3) and ((g.P2 > 0).sum() >= 3))
    nb1 = sum(x['P1_pass'] for x in v.values()); nb2 = sum(x['P2_pass'] for x in v.values())
    nboth = sum(x['both'] for x in v.values())
    line_sep = 'works' if nb1 >= 2 and nb2 >= 2 else ('partial' if nb1 >= 2 or nb2 >= 2 else 'fails')
    line_same = 'works' if nboth >= 2 else ('partial' if nb1 >= 2 or nb2 >= 2 else 'fails')
    return v, line_sep, line_same


agree = 0; mism_reading = 0; diffs = []
for rep in range(30):
    with tempfile.TemporaryDirectory() as t:
        t = pathlib.Path(t); (t / 'b').mkdir(); (t / 'f').mkdir(); (t / 'o').mkdir()
        rr = np.random.default_rng(rep); rows = [make(t, c, rr) for c in cells]
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            E.main(t / 'b', t / 'f', t / 'o')
        V = json.load(open(t / 'o' / 'VERDICT.json')); R = pd.read_csv(t / 'o' / 'RESULTS.csv').set_index('cell')
        for r in rows:
            diffs.append(max(abs(R.loc[r['cell'], 'P1_abs_err'] - r['P1']), abs(R.loc[r['cell'], 'P2_diff'] - r['P2']),
                             abs(R.loc[r['cell'], 'P3_diff'] - r['P3'])))
        v, line_sep, line_same = my_verdict(rows)
        agree += (V['line'] == line_sep) and all(V[b]['P1_pass'] == v[b]['P1_pass'] and V[b]['P2_pass'] == v[b]['P2_pass']
                                                 and V[b]['P3_pass'] == v[b]['P3_pass'] for b in v)
        mism_reading += line_sep != line_same
print(f'endpoint max |diff| over {len(diffs)} synthetic cells: {max(diffs):.2e}')
print(f'verdict agreement (benchmark passes + line, "separate benchmarks" reading): {agree}/30')
print(f'cases where the two readings of the line rule differ: {mism_reading}/30')

# secondary-cell contamination: add 4 secondary processbench cells with P1 far off, see whether evaluate counts them
with tempfile.TemporaryDirectory() as t:
    t = pathlib.Path(t); (t / 'b').mkdir(); (t / 'f').mkdir(); (t / 'o').mkdir()
    rr = np.random.default_rng(99); [make(t, c, rr) for c in cells]
    for p in PRMS:
        make(t, f'TELE16_sa_processbench_{p}', rr)
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        E.main(t / 'b', t / 'f', t / 'o')
    V = json.load(open(t / 'o' / 'VERDICT.json'))
    print('with 4 secondary cells in the fit dir, processbench n_cells =', V['processbench']['n_cells'])
    try:
        make(t, 'SUB5_sa_socratic_qwen_prm_7b', rr)
        with contextlib.redirect_stdout(io.StringIO()):
            E.main(t / 'b', t / 'f', t / 'o')
        print('unknown-benchmark cell: no error')
    except IndexError as e:
        print('unknown-benchmark cell: IndexError (evaluate crashes)')
