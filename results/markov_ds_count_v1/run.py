"""Multi-voter Markov Dawid-Skene error count (docs/experiments/MARKOV_DS_COUNT_PREREG_2026-11-01.md, amendment A).

    python run.py freeze            # PROTOCOL.json with the file hashes (once, before any label)
    python run.py fit [workers]     # label-free stage: fits/<cell>.json + .npz (voters, theta, k-hat per candidate)
    python run.py evaluate          # labels: RESULTS.csv, CONTRASTS.csv, MECHANISM.csv, VERDICT.json

Both fit and evaluate refuse to run if a frozen file hash differs.
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_k] = '1'
import hashlib, json, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / 'results/prm_boost_collection_v1', ROOT / 'results/prm_boost_math_rigor_v1'):
    sys.path.insert(0, str(p))
import evaluate_collection as E  # noqa: E402
from spectral_utils import markov_ds_count as M  # noqa: E402
from spectral_utils import prm_anchored_lsml as A  # noqa: E402

FROZEN = ['spectral_utils/markov_ds_count.py', 'tests/test_markov_ds_count.py', 'results/markov_ds_count_v1/run.py',
          'docs/experiments/MARKOV_DS_COUNT_PREREG_2026-11-01.md', 'results/prm_boost_collection_v1/evaluate_collection.py']
BENCHES = ['socratic', 'hard2verify', 'prmbench']
PRMS = ['qwen_prm_7b', 'qwen_prm_72b', 'pure_prm', 'rlhflow_prm']
ARMS = ['AUTO', 'SUB5', 'TELE16']
NMAX = {'socratic': 16, 'hard2verify': 16, 'prmbench': 12}
CELLS = [(b, p, a) for b in BENCHES for p in PRMS for a in ARMS]
FITS = HERE / 'fits'
NB = ROOT / 'results/neighbor_readout_v1'
STORED = {'AUTO': ('ext', 'AUTO'), 'SUB5': ('ext', 'SUB5'), 'TELE16': ('arm', 'TELE_N16')}
NB_ARM = {'AUTO': ('fits_ext', 'AUTO'), 'SUB5': ('fits_ext', 'SUB5'), 'TELE16': ('fits', 'TELE_N16')}
N16N = ['q15_H1', 'q15_VE1', 'chosen_surprisal', 'logprob_margin', 'true_tail50', 'energy_level', 'energy_innovation',
        'top15_turnover', 'top50_js', 'dominant_freq16', 'bocpd_p0', 'realized_z', 'realized_drv', 'digit_alternative',
        'digit_spread', 'digit_alternative_innovation']
SUB5_IDX = np.array([N16N.index(n) for n in ['q15_H1', 'chosen_surprisal', 'top15_turnover', 'dominant_freq16',
                                             'bocpd_p0', 'digit_spread']])
CANDS = ['MDS', 'IND', 'SINGLE', 'SHARE', 'TOPK']


def sha(path): return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def freeze():
    out = HERE / 'PROTOCOL.json'
    if out.exists(): raise SystemExit('PROTOCOL.json exists; frozen hashes are never rewritten')
    json.dump(dict(frozen_at=time.strftime('%Y-%m-%d %H:%M:%S'), files={f: sha(f) for f in FROZEN},
                   cells=[list(c) for c in CELLS], nmax=NMAX, note='written before any label of this study was read'),
              open(out, 'w'), indent=1)
    print('frozen', out)


def check_hashes():
    P = json.loads((HERE / 'PROTOCOL.json').read_text())
    bad = [f for f, h in P['files'].items() if sha(f) != h]
    if bad: raise SystemExit(f'hash mismatch, refusing to run: {bad}')


def tag(cell): return '__'.join(cell)


def E_json(v):
    if isinstance(v, dict): return {str(k): E_json(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [E_json(x) for x in v]
    if isinstance(v, np.ndarray): return E_json(v.tolist())
    if isinstance(v, np.generic): v = v.item()
    if isinstance(v, float) and not np.isfinite(v): return None
    return v


# ----------------------------------------------------------------------------------------------- voters (label-free)
def stored_score(c, b, p, a):
    kind, name = STORED[a]
    if kind == 'arm': return c.arms[name].astype(float)
    D = np.load(NB / 'fits_ext' / 'scores' / f'{b}__{p}.npz'); assert np.array_equal(D['off'], c.off)
    return D[f'score__{name}'].astype(float)


def voters_s8(c, b, a, X):
    import stage8_auto_forced_k as S8
    if a == 'AUTO':
        zp = E.answer_z(c.arms['PRM_Q80' if b == 'prmbench' else 'PRM'], c.off)
        s, d = S8.fit(np.column_stack([zp, X]))
    else:
        s, d = S8.fit(X, anchor=X.mean(1))
    return s, np.asarray(d['virtual_classifiers'], float), dict(K=int(d['K']), groups=np.asarray(d['c'], int).tolist())


def voters_sub5(c, X):
    """stage2.TeleCell.build + evaluate(SUB5) arithmetic, label-free part only; returns fused and the group components."""
    off = c.off; ans = A.Answers(off); n16 = X.shape[1]
    lens = np.repeat(np.diff(off), np.diff(off)); pool = c.keep & (lens > 1)
    Z = np.column_stack([E.answer_z(X[:, k], off) for k in range(n16)])
    sd = np.array([[X[a_:b_, k].std() for k in range(n16)] for a_, b_ in zip(off[:-1], off[1:])])
    const = np.repeat(sd < 1e-8, np.diff(off), axis=0)
    V = np.empty_like(Z)
    for k in range(n16):
        f = E.topk_flags(Z[:, k], off, E.gap_counts(Z[:, k], off)); f[const[:, k]] = False
        V[:, k] = np.where(f, 1., -1.)
    Vp = V[pool]; R = np.cov(np.ascontiguousarray(Vp).T)
    cols = SUB5_IDX; K, cg = A.groups_from_cov(R[np.ix_(cols, cols)])
    if K < 3: return None, None, dict(K=int(K), groups=np.asarray(cg).tolist())
    mean = Z[:, cols].mean(1); fm = ans.gap_flags(ans.z(mean))
    vmean = np.where(fm, 1., -1.); vmean_p = vmean[pool]
    xi_p, comps = [], []
    for g in range(K):
        idx = cols[cg == g]
        w, _ = A.sml_weights_nonneg(R[np.ix_(idx, idx)])
        if len(idx) == 1:
            a_p = Vp[:, idx[0]]
        else:
            sp = Vp[:, idx] @ w; a_p = np.where(sp > 0, 1., np.where(sp < 0, -1., vmean_p))
        xi_p.append(a_p); comps.append(ans.z(Z[:, idx] @ w))
    wb = A.ds_em_votes(np.column_stack(xi_p), anchor=None)['w']
    fused = sum(wb[g] * comps[g] for g in range(K))
    return fused, np.column_stack(comps), dict(K=int(K), groups=np.asarray(cg).tolist(), between_w=wb.tolist())


def build_voters(b, p, a):
    c = E.load(b, p); X = E.channels(c, 'N16'); f = stored_score(c, b, p, a)
    if a == 'SUB5':
        s, VC, info = voters_sub5(c, X)
        if s is None: return c.off, c.ids, f, None, dict(info, parity=None, arm_valid=False)
        parity = float(np.max(np.abs(s - f))); assert parity < 1e-10, ('SUB5 parity', b, p, parity)
    else:
        s, VC, info = voters_s8(c, b, a, X)
        parity = float(abs(np.corrcoef(s, f)[0, 1])); assert parity > 1 - 1e-9, (a, 'parity', b, p, parity)
    off, ids = c.off.copy(), c.ids.copy()
    del c                                                        # labels never reach the fits
    Vz = np.column_stack([E.answer_z(VC[:, g], off) for g in range(VC.shape[1])])
    fz = E.answer_z(f, off)
    signs = np.array([1. if np.corrcoef(Vz[:, g], fz)[0, 1] >= 0 else -1. for g in range(Vz.shape[1])])
    return off, ids, f, Vz * signs, dict(info, parity=parity, signs=signs.tolist(), arm_valid=Vz.shape[1] >= 3)


# ----------------------------------------------------------------------------------------------- label-free stage
def fit_cell(cell):
    b, p, a = cell
    if (FITS / f'{tag(cell)}.json').exists(): return tag(cell), 'done'
    t0 = time.time()
    off, ids, f, Vz, info = build_voters(b, p, a)
    fz = E.answer_z(f, off); nmax = NMAX[b]; n = np.diff(off)
    k_gap = E.gap_counts(fz, off)
    out = dict(cell=list(cell), voters=info, nmax=nmax, n_answers=int(len(n)),
               n_fit_answers=int(((n >= 3) & (n <= nmax)).sum()), n_long_fallback=int((n > nmax).sum()))
    ks, extra = {}, {}

    def exact(name, V, markov):
        X = M.Exact(V, off, nmax)
        F = M.fit_exact(V, off, nmax, markov=markov, X=X)
        k, marg = M.counts_exact(F['theta'], X, markov)
        long = k < 0
        k = np.where(long, k_gap, k)
        if not F['valid']: k = k_gap.copy()
        out[name] = {kk: F[kk] for kk in F if kk != 'starts'} | dict(starts=[{x: s[x] for x in ('start', 'nll', 'success', 'nit')}
                                                                            for s in F['starts']])
        ks[name] = k; extra[f'marg_{name}'] = marg
        return F

    if info['arm_valid']:
        Fm = exact('MDS', Vz, True); exact('IND', Vz, False)
        T = M.TopK(Vz, f, off); FT = M.fit_topk(Vz, f, off, T=T)
        out['TOPK'] = {kk: FT[kk] for kk in FT if kk != 'starts'}
        ks['TOPK'] = T.counts(FT['theta']) if FT['valid'] else k_gap.copy()
        ks['SHARE'] = M.share_counts(Fm['implied_share'], off) if Fm['valid'] else k_gap.copy()
        out['SHARE'] = dict(defined=bool(Fm['valid']), share=Fm['implied_share'])
    else:
        for name in ('MDS', 'IND', 'TOPK'):
            out[name] = dict(valid=False, reason='arm has K < 3 voters'); ks[name] = k_gap.copy()
        ks['SHARE'] = k_gap.copy(); out['SHARE'] = dict(defined=False, share=None)
    exact('SINGLE', fz[:, None], True)
    out['seconds'] = time.time() - t0
    np.savez(FITS / f'{tag(cell)}.npz', off=off, ids=ids, f=f, Vz=Vz if Vz is not None else np.zeros((len(f), 0)),
             k_gap=k_gap, **{f'k_{nm}': v for nm, v in ks.items()}, **extra)
    json.dump(E_json(out), open(FITS / f'{tag(cell)}.json', 'w'), indent=1)
    v = lambda nm: out[nm].get('valid')
    return tag(cell), (f'ok {out["seconds"]:.0f}s K={info["K"]} MDS valid={v("MDS")} IND={v("IND")} '
                       f'SINGLE={v("SINGLE")} TOPK={v("TOPK")} share={out["MDS"].get("implied_share")}')


def fit(workers):
    check_hashes(); FITS.mkdir(exist_ok=True)
    import multiprocessing as mp
    order = sorted(CELLS, key=lambda c: (c[0] != 'prmbench', c))                 # slowest first
    with mp.get_context('spawn').Pool(workers) as pool:
        for name, st in pool.imap_unordered(fit_cell, order):
            print(time.strftime('%H:%M:%S'), name, st, flush=True)


# ----------------------------------------------------------------------------------------------- evaluation stage
def put(c, name, flags):
    c.arms[name] = c.arms[c._fname]; c.__dict__.setdefault('_f', {})[(name, 'FIX')] = np.asarray(flags, bool)
    return (name, 'FIX')


def per_answer_abs(c, f):
    k = np.add.reduceat((f & c.keep).astype(int), c.off[:-1]); t = c.true_count
    m = np.add.reduceat(c.keep.astype(int), c.off[:-1]) > 0
    return k - t, m


def mae_contrast(c, fa, fb, W):
    Wm, inv, ng = W
    da, m = per_answer_abs(c, fa); db, _ = per_answer_abs(c, fb)
    G = np.zeros((ng, 3)); np.add.at(G, inv[m], np.column_stack([np.abs(da[m]), np.abs(db[m]), np.ones(m.sum())]))
    t = Wm @ G; d = (t[:, 0] - t[:, 1]) / t[:, 2]
    return float(d[0]), float(np.quantile(d[1:], .025)), float(np.quantile(d[1:], .975))


def eval_cell(cell):
    b, p, a = cell
    F = json.loads((FITS / f'{tag(cell)}.json').read_text()); D = np.load(FITS / f'{tag(cell)}.npz')
    c = E.load(b, p)
    assert np.array_equal(D['off'], c.off) and np.array_equal(D['ids'], c.ids.astype(D['ids'].dtype))
    off = c.off; f = D['f']; fz = E.answer_z(f, off)
    c.arms['_F'] = f; c._fname = '_F'
    W = E.boot_weights(c); base = dict(benchmark=b, prm=p, arm=a)
    flags = {nm: E.topk_flags(fz, off, D[f'k_{nm}']) for nm in CANDS}
    flags['GAP'] = E.topk_flags(fz, off, E.gap_counts(fz, off)); flags['K2'] = E.topk_flags(fz, off, 2)
    flags['ORACLE'] = E.topk_flags(fz, off, c.true_count)
    nd, na = NB_ARM[a]; hf = NB / nd / f'{b}__{p}__{na}'
    if hf.with_suffix('.json').exists():
        H = np.load(hf.with_suffix('.npz')); HJ = json.loads(hf.with_suffix('.json').read_text())
        assert np.allclose(H['s'], fz)
        flags['H1b'] = (H['h1_logit'] > 0) if HJ['H1']['valid'] else flags['GAP']
    keys = {nm: put(c, nm, fl) for nm, fl in flags.items()}
    rows, con = [], []
    t = c.true_count; m = np.add.reduceat(c.keep.astype(int), off[:-1]) > 0
    for nm, fl in flags.items():
        r, _, _ = E.measure(c, *keys[nm], fast=False)
        dlt, _ = per_answer_abs(c, fl); kk = dlt + t
        fit_ = F.get(nm, {})
        rows.append(dict(base, readout=nm, official=r['official'], count_mae=float(np.abs(dlt[m]).mean()),
                         count_bias=float(dlt[m].mean()), spearman_count=float(spearmanr(kk[m], t[m])[0]),
                         mean_flags=r['mean_flags_per_answer'], TPR=r['TPR'], TNR=r['TNR'],
                         valid=fit_.get('valid'), reason=fit_.get('reason'), implied_share=fit_.get('implied_share'),
                         e=fit_.get('e'), rho=fit_.get('rho'), K=F['voters']['K'],
                         n_long_fallback=F['n_long_fallback'], n_answers=F['n_answers'],
                         **{k2: v for k2, v in r.items() if k2.startswith('official__')}))
    pairs = [('MDS', 'GAP'), ('MDS', 'SHARE'), ('MDS', 'IND'), ('MDS', 'SINGLE'), ('MDS', 'K2'), ('MDS', 'TOPK'),
             ('IND', 'GAP'), ('SINGLE', 'GAP'), ('SHARE', 'GAP'), ('TOPK', 'GAP')]
    if 'H1b' in flags: pairs += [('MDS', 'H1b')]
    for x, y in pairs:
        d0, lo, hi = mae_contrast(c, flags[x], flags[y], W)
        con.append(dict(base, contrast=f'{x} - {y}', metric='count_mae', delta=d0, ci_low=lo, ci_high=hi))
        for r in E.contrast(c, keys[x], keys[y], W):
            if r['metric'] == 'official':
                con.append(dict(base, contrast=f'{x} - {y}', metric='official', delta=r['delta'], ci_low=r['ci_low'],
                                ci_high=r['ci_high']))
    # mechanism (E4)
    mech = dict(base, true_step_share=float(c.err[c.keep].mean()), K=F['voters']['K'])
    if F['voters'].get('arm_valid') and F['MDS'].get('valid') is not None:
        Vz = D['Vz']; keep = c.keep if b != 'prmbench' else None
        wa = [float(np.nanmean(E.per_answer_auc(c.err, Vz[:, g], off, keep))) for g in range(Vz.shape[1])]
        mech.update(implied_share=F['MDS']['implied_share'], e=F['MDS']['e'], rho=F['MDS']['rho'],
                    delta=F['MDS']['delta'], voter_within_auc=wa,
                    spearman_delta_auc=float(spearmanr(F['MDS']['delta'], wa)[0]) if len(wa) >= 3 else None)
        mg = D['marg_MDS']; ok = np.isfinite(mg) & c.keep
        pos = np.concatenate([np.linspace(0, 1, n_) if n_ > 1 else [0.] for n_ in np.diff(off)])
        mech.update(spearman_posterior_position=float(spearmanr(mg[ok], pos[ok])[0]),
                    spearman_truth_position=float(spearmanr(c.err[ok], pos[ok])[0]),
                    within_auc_posterior=float(np.nanmean(E.per_answer_auc(c.err, np.where(np.isfinite(mg), mg, 0), off, c.keep))),
                    within_auc_fused=float(np.nanmean(E.per_answer_auc(c.err, f, off, c.keep))))
    for nm in ('MDS', 'GAP'):
        r = next(r for r in rows if r['readout'] == nm); mech[f'spearman_count_{nm}'] = r['spearman_count']
    return rows, con, mech


def verdict(rows, con):
    def one(L):
        assert len(L) == 1, L
        return L[0]
    ci = lambda p, a, cn, mt: one([r for r in con if r['benchmark'] == 'socratic' and r['prm'] == p and r['arm'] == a
                                   and r['contrast'] == cn and r['metric'] == mt])
    out = {}
    for a in ARMS:
        e1 = []
        for p in PRMS:
            mds = one([r for r in rows if r['benchmark'] == 'socratic' and r['prm'] == p and r['arm'] == a and r['readout'] == 'MDS'])
            shr = one([r for r in rows if r['benchmark'] == 'socratic' and r['prm'] == p and r['arm'] == a and r['readout'] == 'SHARE'])
            ok_fit = bool(mds['valid']) and shr['valid'] is not False
            parts = dict(mds_valid=bool(mds['valid']),
                         mae_vs_gap=ci(p, a, 'MDS - GAP', 'count_mae')['ci_high'] < 0,
                         mae_vs_share=ci(p, a, 'MDS - SHARE', 'count_mae')['ci_high'] < 0,
                         official_vs_gap=ci(p, a, 'MDS - GAP', 'official')['ci_low'] > -.005)
            parts['pass'] = ok_fit and parts['mae_vs_gap'] and parts['mae_vs_share'] and parts['official_vs_gap']
            e1.append(dict(prm=p, **parts))
        e2 = [ci(p, a, 'MDS - IND', 'count_mae')['ci_high'] < 0 for p in PRMS]
        e3 = [ci(p, a, 'MDS - SINGLE', 'count_mae')['ci_high'] < 0 for p in PRMS]
        out[a] = dict(E1=dict(per_prm=e1, pass_=sum(x['pass'] for x in e1) >= 3),
                      E2=dict(per_prm=e2, pass_=sum(e2) >= 3), E3=dict(per_prm=e3, pass_=sum(e3) >= 3))
    return out


def evaluate():
    check_hashes()
    missing = [tag(c) for c in CELLS if not (FITS / f'{tag(c)}.json').exists()]
    if missing: raise SystemExit(f'fits missing: {missing}')
    rows, con, mech = [], [], []
    for cell in CELLS:
        t0 = time.time(); rw, cn, me = eval_cell(cell); rows += rw; con += cn; mech.append(me)
        print(time.strftime('%H:%M:%S'), tag(cell), f'{time.time() - t0:.0f}s', flush=True)
    E.write_csv(HERE / 'RESULTS.csv', rows); E.write_csv(HERE / 'CONTRASTS.csv', con)
    E.write_csv(HERE / 'MECHANISM.csv', [E_json(m) for m in mech])
    json.dump(E_json(verdict(rows, con)), open(HERE / 'VERDICT.json', 'w'), indent=1)
    print('written RESULTS.csv CONTRASTS.csv MECHANISM.csv VERDICT.json')


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'freeze': freeze()
    elif cmd == 'fit': fit(int(sys.argv[2]) if len(sys.argv) > 2 else 6)
    elif cmd == 'evaluate': evaluate()
    elif cmd == 'voters':                                         # label-free parity smoke for one cell
        o, i, f, V, info = build_voters(*sys.argv[2:5]); print({k: v for k, v in info.items() if k != 'signs'}, None if V is None else V.shape)
    else: raise SystemExit(cmd)
