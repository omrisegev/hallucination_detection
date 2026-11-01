"""Neighbor-aware readout study (docs/experiments/NEIGHBOR_READOUT_PREREG_2026-11-01.md).

    python run.py freeze            # write PROTOCOL.json with the file hashes (once, before any label)
    python run.py fit [workers]     # label-free stage: fits/<cell>.json + .npz; reads scores and offsets only
    python run.py evaluate          # labels: CURVES.csv, RESULTS.csv, CONTRASTS.csv, VERDICT.json

Both fit and evaluate refuse to run if a frozen file hash differs.
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_k] = '1'
import hashlib, json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'results/prm_boost_collection_v1'))
import evaluate_collection as E  # noqa: E402
from spectral_utils import neighbor_readout as NR  # noqa: E402

FROZEN = ['spectral_utils/neighbor_readout.py', 'tests/test_neighbor_readout.py', 'results/neighbor_readout_v1/run.py',
          'docs/experiments/NEIGHBOR_READOUT_PREREG_2026-11-01.md', 'results/prm_boost_collection_v1/evaluate_collection.py']
BENCHES = ['prmbench', 'socratic', 'hard2verify']
PRMS = ['qwen_prm_7b', 'qwen_prm_72b', 'pure_prm', 'rlhflow_prm']
ARMS = ['CHAIN_N16', 'TELE_N16']
CELLS = [(b, p, a) for b in BENCHES for p in PRMS for a in ARMS]
FITS = HERE / 'fits'


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def freeze():
    out = HERE / 'PROTOCOL.json'
    if out.exists(): raise SystemExit('PROTOCOL.json exists; frozen hashes are never rewritten')
    json.dump(dict(frozen_at=time.strftime('%Y-%m-%d %H:%M:%S'), files={f: sha(f) for f in FROZEN},
                   cells=[list(c) for c in CELLS], note='written before any label of this study was read'),
              open(out, 'w'), indent=1)
    print('frozen', out)


def check_hashes():
    P = json.loads((HERE / 'PROTOCOL.json').read_text())
    bad = [f for f, h in P['files'].items() if sha(f) != h]
    if bad: raise SystemExit(f'hash mismatch, refusing to run: {bad}')


def tag(cell): return '__'.join(cell)


# ----------------------------------------------------------------------------------------------- label-free stage
def fit_cell(cell):
    b, p, a = cell
    if (FITS / f'{tag(cell)}.json').exists(): return tag(cell), 'done'
    c = E.load(b, p); s = E.answer_z(c.arms[a], c.off); off = c.off.copy(); ids = c.ids.copy()
    del c                                                        # labels never reach the fits
    t0 = time.time()
    k_gap = E.gap_counts(s, off)
    h1 = NR.hmm_fit(s, off, position=True); h0 = NR.hmm_fit(s, off, position=False)
    v1 = NR.variogram_fit(s, off); g1 = NR.gcv_lambda(s, off)
    np.savez(FITS / f'{tag(cell)}.npz', s=s, off=off, ids=ids, k_gap=k_gap,
             h1_logit=NR.hmm_logit(s, off, h1), h0_logit=NR.hmm_logit(s, off, h0))
    json.dump(E_json(dict(cell=list(cell), H1=h1, H1_NOPOS=h0, V1=v1, G1=g1, seconds=time.time() - t0)),
              open(FITS / f'{tag(cell)}.json', 'w'), indent=1)
    return tag(cell), f'ok {time.time() - t0:.0f}s H1 valid={h1["valid"]} V1 valid={v1["valid"]} G1 lam={g1["lam"]}'


def E_json(v):
    if isinstance(v, dict): return {str(k): E_json(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [E_json(x) for x in v]
    if isinstance(v, np.ndarray): return E_json(v.tolist())
    if isinstance(v, np.generic): v = v.item()
    if isinstance(v, float) and not np.isfinite(v): return None
    return v


def fit(workers):
    check_hashes(); FITS.mkdir(exist_ok=True)
    import multiprocessing as mp
    with mp.get_context('spawn').Pool(workers) as pool:
        for name, st in pool.imap_unordered(fit_cell, CELLS):
            print(time.strftime('%H:%M:%S'), name, st, flush=True)


# ----------------------------------------------------------------------------------------------- evaluation stage
def put(c, name, score, flags):
    c.arms[name] = np.asarray(score, float); c.__dict__.setdefault('_f', {})[(name, 'FIX')] = np.asarray(flags, bool)
    return (name, 'FIX')


def count_mae(c, f):
    k = np.add.reduceat((f & c.keep).astype(int), c.off[:-1])
    m = np.add.reduceat(c.keep.astype(int), c.off[:-1]) > 0
    return float(np.mean(np.abs(k - c.true_count)[m]))


def eval_cell(cell):
    b, p, a = cell
    F = json.loads((FITS / f'{tag(cell)}.json').read_text()); D = np.load(FITS / f'{tag(cell)}.npz')
    c = E.load(b, p)
    assert np.array_equal(D['off'], c.off) and np.array_equal(D['ids'], c.ids.astype(D['ids'].dtype))
    s, off, k = D['s'], c.off, D['k_gap']
    assert np.array_equal(s, E.answer_z(c.arms[a], c.off))
    top = lambda sc, kk=k: E.topk_flags(sc, off, kk)
    gap = (a, 'GAP'); W = E.boot_weights(c)
    curves, rows, con = [], [], []
    base = dict(benchmark=b, prm=p, arm=a)

    def record(readout, key, stage, fast=False, extra=None):
        r, C, _ = E.measure(c, *key, fast=fast)
        row = dict(base, readout=readout, stage=stage, official=r['official'], within_auc=r['within_auc'],
                   count_mae=count_mae(c, c.flags(*key)), mean_flags=r['mean_flags_per_answer'],
                   TPR=r['TPR'], TNR=r['TNR'], **(extra or {}))
        row.update({k2: v for k2, v in r.items() if k2.startswith('official__')})
        return row

    # stage 1: label-using diagnostic curves (count fixed at k_GAP; RECOUNT also re-runs GAP)
    for form, grid, fn in (('SYM', NR.SYM_GRID, NR.sym), ('LIFT', NR.SYM_GRID, NR.lift), ('LAP', NR.LAP_GRID, NR.lap),
                           ('SYM_RECOUNT', NR.SYM_GRID, NR.sym)):
        for g in grid:
            sc = fn(s, off, g)
            kk = E.gap_counts(E.answer_z(sc, off), off) if form == 'SYM_RECOUNT' else k
            curves.append(record(f'{form}({g})', put(c, f'{form}_{g}', sc, top(sc, kk)), 'stage1', fast=True,
                                 extra=dict(form=form, param=g)))
    # references
    for rule in ('GAP', 'K2', 'ORACLE'):
        rows.append(record(rule, (a, rule), 'reference'))
    # stage 2 candidates (fallback to GAP when the fit is invalid)
    cands = {}
    for name, key in (('H1', 'h1_logit'), ('H1_NOPOS', 'h0_logit')):
        fit_ = F[name]; lo = D[key]
        if fit_['valid']:
            cands[f'{name}a'] = (lo, top(lo)); cands[f'{name}b'] = (lo, lo > 0)
        else:
            cands[f'{name}a'] = cands[f'{name}b'] = (s, top(s))
    w = F['V1']['w'] if F['V1']['valid'] else 0.
    cands['V1'] = (NR.sym(s, off, w), top(NR.sym(s, off, w)))
    lam = F['G1']['lam']; cands['G1'] = (NR.lap(s, off, lam), top(NR.lap(s, off, lam)))
    fb = dict(H1a=not F['H1']['valid'], H1b=not F['H1']['valid'], H1_NOPOSa=not F['H1_NOPOS']['valid'],
              H1_NOPOSb=not F['H1_NOPOS']['valid'], V1=not F['V1']['valid'], G1=False)
    keys = {}
    for name, (sc, fl) in cands.items():
        keys[name] = put(c, name, sc, fl)
        rows.append(record(name, keys[name], 'candidate', extra=dict(fallback_to_gap=fb[name],
                                                                     V1_w=w if name == 'V1' else None,
                                                                     G1_lam=lam if name == 'G1' else None)))
    pairs = [(n, gap) for n in cands] + [(keys['H1b'], (a, 'K2')), (keys['H1a'], keys['H1_NOPOSa']),
                                         (keys['H1b'], keys['H1_NOPOSb'])]
    for x, y in pairs:
        xk = keys[x] if isinstance(x, str) else x
        for r in E.contrast(c, xk, y, W):
            con.append(dict(r, **base))
    return curves, rows, con


def verdict(curves, rows, con):
    get = lambda L, **kw: [r for r in L if all(r[k] == v for k, v in kw.items())]

    def one(r): assert len(r) == 1, r; return r[0]

    def offi(b, p, a, ro): return one(get(rows, benchmark=b, prm=p, arm=a, readout=ro))['official']

    def ci(b, p, a, cname, metric):
        return one(get(con, benchmark=b, prm=p, arm=a, contrast=cname, metric=metric))

    out = {}
    for a in ARMS:
        for b in ('prmbench', 'socratic'):
            margin = .5 if b == 'prmbench' else .005
            gap_name = lambda n: f'{n}[FIX] - {a}[GAP]'
            p1 = [ci(b, p, a, gap_name('H1a'), 'official')['ci_low'] > 0 for p in PRMS]
            p1w = [ci(b, p, a, gap_name('H1a'), 'within_auc')['ci_high'] >= 0 for p in PRMS]
            p2k = [offi(b, p, a, 'H1b') >= offi(b, p, a, 'K2') - margin for p in PRMS]
            mae = lambda p, ro: one(get(rows, benchmark=b, prm=p, arm=a, readout=ro))['count_mae']
            p2m = [mae(p, 'H1b') < mae(p, 'GAP') for p in PRMS]
            P = dict(P1=dict(official_ci_low_gt0=p1, within_not_worse=p1w, pass_=sum(p1) >= 3 and all(p1w)),
                     P2=dict(h1b_ge_k2_minus_margin=p2k, mae_below_gap=p2m, pass_=sum(p2k) >= 3 and sum(p2m) >= 3))
            if b == 'prmbench':
                p3 = {}
                for cand, form in (('H1a', 'SYM'), ('V1', 'SYM'), ('G1', 'LAP')):
                    ratios = []
                    for p in PRMS:
                        cv = get(curves, benchmark=b, prm=p, arm=a, form=form)
                        g0 = offi(b, p, a, 'GAP'); peak = max(r['official'] for r in cv)
                        ratios.append((offi(b, p, a, cand) - g0) / (peak - g0) if peak - g0 > 0 else None)
                    p3[cand] = dict(ratio=ratios, pass_=sum(r is not None and r >= .5 for r in ratios) >= 3)
                best, lift_ge = [], []
                for p in PRMS:
                    pk = {f: max(r['official'] for r in get(curves, benchmark=b, prm=p, arm=a, form=f))
                          for f in ('SYM', 'LIFT', 'LAP')}
                    best.append(max(pk, key=pk.get)); lift_ge.append(pk['LIFT'] >= pk['SYM'])
                P.update(P3=p3, P4=dict(best_form=best, lift_ge_sym=lift_ge))
            out[f'{a}/{b}'] = P
    for a in ARMS:                                               # T: a rule passing on PRMBench must pass on Socratic
        for rule in ('P1', 'P2'):
            if out[f'{a}/prmbench'][rule]['pass_']:
                out[f'{a}/transfer_{rule}'] = out[f'{a}/socratic'][rule]['pass_']
    return out


def evaluate():
    check_hashes()
    missing = [tag(c) for c in CELLS if not (FITS / f'{tag(c)}.json').exists()]
    if missing: raise SystemExit(f'fits missing: {missing}')
    curves, rows, con = [], [], []
    for cell in CELLS:
        t0 = time.time(); cv, rw, cn = eval_cell(cell); curves += cv; rows += rw; con += cn
        print(time.strftime('%H:%M:%S'), tag(cell), f'{time.time() - t0:.0f}s', flush=True)
    E.write_csv(HERE / 'CURVES.csv', curves); E.write_csv(HERE / 'RESULTS.csv', rows); E.write_csv(HERE / 'CONTRASTS.csv', con)
    json.dump(E_json(verdict(curves, rows, con)), open(HERE / 'VERDICT.json', 'w'), indent=1)
    print('written CURVES.csv RESULTS.csv CONTRASTS.csv VERDICT.json')


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'freeze': freeze()
    elif cmd == 'fit': fit(int(sys.argv[2]) if len(sys.argv) > 2 else 6)
    elif cmd == 'evaluate': evaluate()
    else: raise SystemExit(cmd)
