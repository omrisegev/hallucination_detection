# PATCHED COPY of results/monday_lr_reference_v1/run.py (thinkfix_rescore_v1, variant=thinkfix): V1/V2 -> this variant's nested_banks; ext.FE set by the wrapper.
"""monday_lr_reference_v1: SUPERVISED logistic-regression reference per nested bank (labelled reference, not a method).

Stage 'fit'      : per bank -> cache/<bank>.pkl (LR nested OOF on PRMBench, all-fold LR applied to ProcessBench,
                   label-free fit_cell on the PRMBench cell for the comparison vectors and method scores).
Stage 'aggregate': LR_SCORES.csv, LR_COEFFICIENTS.csv, COEF_COMPARISON.csv, GAPS.csv, PARITY.json.
Stage 'figures'  : figures/L1, L2, L3 (light + dark, PNG + SVG).
Protocol: PROTOCOL.json. Usage: python run.py [fit|aggregate|figures|all]
"""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '2'
import gc, hashlib, importlib.util, itertools, json, pickle, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

MAIN = Path(r'C:\Users\omris\TAU\hallucination_detection'); OUT = Path(__file__).resolve().parent; CACHE = OUT / 'cache'
EXT = MAIN / 'results/monday_external_panel_v1'; V1 = Path(r'/shared/cycle2_tau_averbuch_prj/omrisegev1/results/thinkfix_rescore_v1/thinkfix/nested_banks'); V2 = Path(r'/shared/cycle2_tau_averbuch_prj/omrisegev1/results/thinkfix_rescore_v1/thinkfix/nested_banks')
SE = MAIN / '.worktrees/readout-quickest-detection-v1/results/step_evidence_v1'
spec = importlib.util.spec_from_file_location('ext_score', EXT / 'score.py'); ext = importlib.util.module_from_spec(spec); spec.loader.exec_module(ext)
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import log_loss  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

NB = json.loads((V2 / 'BANKS.json').read_text(encoding='utf8'))
BANKS = ['N11', 'N14', 'N13', 'N16', 'N22', 'N25', 'N53', 'N56', 'N40', 'N43']        # v2 order (TCI seeds)
PLOT_ORDER = [['N11', 'N13', 'N22', 'N40', 'N53'], ['N14', 'N16', 'N25', 'N43', 'N56']]
DIGIT_BANKS = set(PLOT_ORDER[1])
ext.BN = dict(NB)
OB = json.loads((MAIN / 'results/estimator_input_v1/run_20261004_resume/INPUT_MANIFEST.json').read_text(encoding='utf8'))['banks']
PRM = 'prmbench_qwen3_8b'; METHODS4 = ['HEM_MOM_cc', 'MEAN_MOM_cc', 'MEAN_MOM_ga', 'TCI_ONLY_ga']
SCORERS = METHODS4 + ['HEM_MOM_ga', 'ALL_EQ', 'DS_FILTER_EQ']
C_GRID = (.001, .01, .1, 1., 10., 100., 1000.); DRAWS, SEED = 1000, 20261007
EPS8 = 8 * np.finfo(float).eps


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def family_map():
    core = OB['B13'][:11]; real = OB['B13'][11:]; dig = OB['B16'][-3:]
    e20 = [c for c in OB['B20'] if c not in core]; hist = set(NB['N40']) - set(NB['N22'])
    fam = {}
    for c in NB['N56']:
        fam[c] = ('core11' if c in core else 'realized2' if c in real else 'digit3' if c in dig else 'ct7_ext9' if c in e20
                  else 'hist18' if c in hist else 'other13')
    return fam


# ------------------------------------------------------------------ data
def load_population():
    A = pd.read_csv(SE / 'OOF_ANSWERS.csv', encoding='utf-8-sig')
    with np.load(SE / 'OOF_STEP_SCORES.npz') as z: OFF = z['offsets'].astype(np.int64); LAB = z['labels'].astype(bool)
    F = np.load(ext.FE / 'source/FEATURES.npz')
    if list(map(str, F['uids'])) != A.uid.astype(str).tolist() or not np.array_equal(F['offsets'], OFF): raise SystemExit('source order')
    fr = json.loads((SE / 'INPUT_FREEZE.json').read_text(encoding='utf-8-sig'))
    meta = {m['idx']: m for m in pickle.load(open(fr['prm_metadata']['path'], 'rb')).values()}
    cells = A.cell.to_numpy(); ids = A.id.astype(str).to_numpy(); pb = np.array([c.startswith('pb_') for c in cells])
    noncontrol = np.array([(not pb[i]) and meta[ids[i]]['classification'] != 'correct' for i in range(len(A))])
    return A, OFF, LAB, F, noncontrol


def auc(y, s):
    n1 = int(y.sum()); n0 = len(y) - n1
    return np.nan if not n1 or not n0 else (rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def topk(s, k):
    f = np.zeros(len(s), bool); f[np.lexsort((np.arange(len(s)), -s))[:k]] = True; return f


def conf(f, y):
    v, g = ~f, ~y
    return [(v & g).sum(), (v & ~g).sum(), (~v & ~g).sum(), (~v & g).sum()]


def argmax_first(s):
    return int(np.flatnonzero(s >= s.max() - EPS8)[0])


def per_answer(score, off, y, nonctl):
    """dev_run.fill for PRMBench answers: within-AUC, top1-on-error, top-1 confusion (non-control), argmax."""
    n = len(off) - 1; out = {'auc': np.full(n, np.nan), 'top1err': np.full(n, np.nan), 'c1': np.zeros((n, 4), int), 'argmax': np.zeros(n, int)}
    for j in range(n):
        a, b = off[j], off[j + 1]; s = np.asarray(score[a:b], float); yy = y[a:b]; am = argmax_first(s); out['argmax'][j] = am
        out['auc'][j] = auc(yy, s)
        if yy.any(): out['top1err'][j] = float(yy[am])
        if nonctl[j]: out['c1'][j] = conf(topk(s, 1), yy)
    return out


# ------------------------------------------------------------------ LR (mirrors monday_fusion_lr.nested_lr)
def model_fit(x, y, c):
    if len(np.unique(y)) != 2: raise ValueError('Training fold has only one class')
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=c, class_weight='balanced', solver='lbfgs', max_iter=3000, random_state=20261003)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(scaler.transform(x), y)
    return scaler, model


def balanced_loss(y, prob):
    counts = np.bincount(y, minlength=2)
    if (counts == 0).any(): raise ValueError('Inner validation lacks a class')
    return float(log_loss(y, prob, sample_weight=len(y) / (2 * counts[y]), labels=[0, 1]))


def nested_lr(x, y, folds):
    oof = np.full(len(x), np.nan); diag = []; models = []
    for outer in range(5):
        losses, nonconv = [], []
        for c in C_GRID:
            trial = []
            try:
                for inner in range(5):
                    if inner == outer: continue
                    tr = (folds != outer) & (folds != inner); va = folds == inner
                    sc, fit = model_fit(x[tr], y[tr], c)
                    trial.append(balanced_loss(y[va], fit.predict_proba(sc.transform(x[va]))[:, 1]))
                losses.append(float(np.mean(trial)))
            except ConvergenceWarning:
                losses.append(float('inf')); nonconv.append(c)
        if not np.isfinite(losses).any(): raise RuntimeError(f'no converged C in outer fold {outer}')
        best = min(range(len(C_GRID)), key=lambda j: (losses[j], C_GRID[j])); c = C_GRID[best]
        tr = folds != outer; te = folds == outer
        sc, fit = model_fit(x[tr], y[tr], c)
        oof[te] = fit.decision_function(sc.transform(x[te]))
        models.append((sc, fit))
        diag.append({'outer_fold': outer, 'C': c, 'inner_balanced_log_loss': losses, 'nonconverged_C': nonconv,
                     'n_train_steps': int(tr.sum()), 'n_test_steps': int(te.sum()), 'coef_std': fit.coef_[0].tolist(),
                     'intercept': float(fit.intercept_[0]), 'mean': sc.mean_.tolist(), 'scale': sc.scale_.tolist(),
                     'coef_eff': (fit.coef_[0] / sc.scale_).tolist()})
    if not np.isfinite(oof).all(): raise RuntimeError('Incomplete OOF scores')
    c_all = float(np.median([d['C'] for d in diag]))
    sc, fit = model_fit(x, y, c_all)
    return oof, diag, models, (sc, fit), c_all


# ------------------------------------------------------------------ stage fit
def fit_stage():
    CACHE.mkdir(exist_ok=True)
    A, OFF, LAB, F, noncontrol = load_population()
    cells = A.cell.to_numpy(); fold = A.fold.to_numpy().astype(int); target = A.target.to_numpy().astype(int)
    sorted_cells = sorted(set(cells)); ci_prm = sorted_cells.index(PRM)
    prm_ans = np.flatnonzero(cells == PRM); pb_cells = sorted(c for c in sorted_cells if c.startswith('pb_'))
    for bi, bk in enumerate(BANKS):
        path = CACHE / f'{bk}.pkl'
        if path.exists(): print(bk, 'cached', flush=True); continue
        t0 = time.time(); pop = ext.Population(F, [bk]); bank = pop.cell_bank(bk, prm_ans)
        V = bank['V']; off = bank['off']; y = LAB[bank['rows']]; ns = np.diff(off); folds = np.repeat(fold[prm_ans], ns)
        nonctl = noncontrol[prm_ans]; names = bank['names']
        if not np.array_equal(bank['rows'], np.concatenate([np.arange(OFF[i], OFF[i + 1]) for i in prm_ans])): raise SystemExit('row map')
        res = {'bank': bk, 'names': names, 'n_steps': len(V), 'n_answers': len(prm_ans), 'lf_signs_prm': bank['diagnostics']['lf_signs']}
        # label-free fit (PRMBench cell), same seed rule as v2 dev_run
        try:
            sc, surv, d = ext.fit_cell(pop, bank, 1000 * bi + ci_prm); res['fit_failure'] = None
        except RuntimeError as exc:
            st0 = ext.fit_content(pop, bank, recipes=[]); sc = {k: st0['scores'][k] for k in ('ALL_EQ', 'DS_FILTER_EQ') if k in st0['scores']}
            surv = st0['surv']; d = {'FAILED': str(exc)[:300]}; res['fit_failure'] = str(exc)[:300]
        res['fit_diag'] = d; res['survivors'] = [names[i] for i in surv]
        print(bk, 'label-free fit', round(time.time() - t0), flush=True)
        lev = pop.level[bank['rows']]
        sg = np.array([(np.sign(np.corrcoef(V[:, j], lev)[0, 1]) or 1.) if V[:, j].std() > 0 else 1. for j in range(V.shape[1])])
        sc['ALL_EQ_ORIENTED'] = (V * sg).mean(1); res['oriented_signs'] = sg
        res['methods'] = {k: per_answer(s, off, y, nonctl) for k, s in sc.items()}
        if 'MEAN_MOM_ga' in sc:
            X1 = np.column_stack([np.ones(len(V)), V]); beta, *_ = np.linalg.lstsq(X1, sc['MEAN_MOM_ga'], rcond=None)
            pred = X1 @ beta; r2 = 1 - np.sum((sc['MEAN_MOM_ga'] - pred) ** 2) / np.sum((sc['MEAN_MOM_ga'] - sc['MEAN_MOM_ga'].mean()) ** 2)
            res['implied_mean_mom'] = beta[1:]; res['implied_r2'] = float(r2)
            res['implied_within_auc_check'] = float(np.nanmean(per_answer(pred, off, y, nonctl)['auc']))
        # supervised LR
        t1 = time.time()
        oof, ldiag, models, (sca, fita), c_all = nested_lr(V, y.astype(int), folds)
        res['lr'] = per_answer(oof, off, y, nonctl); res['lr_diag'] = ldiag; res['lr_C_all'] = c_all
        res['lr_all_coef_eff'] = fita.coef_[0] / sca.scale_; res['lr_all_coef_std'] = fita.coef_[0]
        ins = fita.decision_function(sca.transform(V)); res['lr_in_sample'] = per_answer(ins, off, y, nonctl)
        print(bk, 'LR', round(time.time() - t1), 'C', [d_['C'] for d_ in ldiag], 'oof auc', round(float(np.nanmean(res['lr']['auc'])), 4), flush=True)
        # ProcessBench transfer
        pbres = {}
        prm_sign = {c: v['sign'] for c, v in bank['diagnostics']['lf_signs'].items()}
        for ce in pb_cells:
            ans = np.flatnonzero(cells == ce); bc = pop.cell_bank(bk, ans); Vc = bc['V'].copy(); flips = 0
            for j, c in enumerate(bc['names']):
                if c in bc['diagnostics']['lf_signs']:
                    f_ = prm_sign[c] * bc['diagnostics']['lf_signs'][c]['sign']; Vc[:, j] *= f_; flips += f_ < 0
            s_all = fita.decision_function(sca.transform(Vc)); s_fm = np.zeros(len(Vc))
            for k in range(5):
                rows = np.repeat(fold[ans] == k, np.diff(bc['off']))
                if rows.any(): s_fm[rows] = models[k][1].decision_function(models[k][0].transform(Vc[rows]))
            o = bc['off']
            pbres[ce] = {'answers': ans, 'target': target[ans], 'aligned_flips': int(flips),
                         'argmax_all': np.array([argmax_first(s_all[o[j]:o[j + 1]]) for j in range(len(ans))]),
                         'argmax_fold': np.array([argmax_first(s_fm[o[j]:o[j + 1]]) for j in range(len(ans))])}
        res['pb'] = pbres; res['runtime_s'] = round(time.time() - t0)
        with open(path.with_suffix('.tmp'), 'wb') as f: pickle.dump(res, f)
        path.with_suffix('.tmp').replace(path)
        print(bk, 'done', res['runtime_s'], flush=True)
        del pop, bank, V, sc; gc.collect()


# ------------------------------------------------------------------ stage scores (step-level agreement; added after first aggregate)
def demean(x, off):
    out = np.asarray(x, float).copy()
    for a, b in zip(off[:-1], off[1:]): out[a:b] -= out[a:b].mean()
    return out


def scores_stage():
    """Rebuild the PRMBench bank, recompute MEAN_MOM (fit_content, same call fit_cell uses for MEAN_MOM_ga) and the LR OOF
    step score from the saved fold models; min-norm implied weights; within-answer step-score correlations."""
    A, OFF, LAB, F, noncontrol = load_population(); cells = A.cell.to_numpy(); prm_ans = np.flatnonzero(cells == PRM)
    fold = A.fold.to_numpy().astype(int)
    for bk in BANKS:
        path = CACHE / f'{bk}_scores.pkl'
        if path.exists(): continue
        R = pickle.load(open(CACHE / f'{bk}.pkl', 'rb')); t0 = time.time()
        pop = ext.Population(F, [bk]); bank = pop.cell_bank(bk, prm_ans); V = bank['V']; off = bank['off']; y = LAB[bank['rows']]
        folds = np.repeat(fold[prm_ans], np.diff(off))
        st = ext.fit_content(pop, bank, recipes=['MEAN_EQ', 'MEAN_MOM', 'HEM_EQ', 'HEM_MOM'])
        sc = {'MEAN_MOM_ga': st['scores']['MEAN_MOM'], 'ALL_EQ': st['scores']['ALL_EQ'], 'DS_FILTER_EQ': st['scores']['DS_FILTER_EQ'],
              'ALL_EQ_ORIENTED': (V * R['oriented_signs']).mean(1)}
        lr = np.zeros(len(V))
        for d in R['lr_diag']:
            m = folds == d['outer_fold']
            lr[m] = ((V[m] - np.array(d['mean'])) / np.array(d['scale'])) @ np.array(d['coef_std']) + d['intercept']
        nonctl = noncontrol[prm_ans]
        par = {k: float(np.nanmax(np.abs(np.nan_to_num(per_answer(sc[k], off, y, nonctl)['auc'], nan=-1) - np.nan_to_num(R['methods'][k]['auc'], nan=-1))))
               for k in sc}
        par['LR_OOF'] = float(np.nanmax(np.abs(np.nan_to_num(per_answer(lr, off, y, nonctl)['auc'], nan=-1) - np.nan_to_num(R['lr']['auc'], nan=-1))))
        X1 = np.column_stack([np.ones(len(V)), V]); beta, _, rank, sv = np.linalg.lstsq(X1, sc['MEAN_MOM_ga'], rcond=1e-6)
        pred = X1 @ beta; yy = sc['MEAN_MOM_ga']; r2 = 1 - np.sum((yy - pred) ** 2) / np.sum((yy - yy.mean()) ** 2)
        D = demean(V, off); cov = D.T @ D / len(D)
        dl = demean(lr, off); corr = {}
        for k, s in sc.items():
            ds = demean(s, off); corr[k] = float(dl @ ds / np.sqrt((dl @ dl) * (ds @ ds)))
        dp = demean(pred, off); corr['MEAN_MOM_IMPLIED_linear'] = float(dl @ dp / np.sqrt((dl @ dl) * (dp @ dp)))
        out = {'implied_minnorm': beta[1:], 'implied_r2_minnorm': float(r2), 'lstsq_rank': int(rank), 'n_params': X1.shape[1],
               'cond': float(sv[0] / sv[-1]), 'cov_within': cov, 'score_corr_within': corr, 'auc_parity_max_abs_diff': par,
               'implied_minnorm_within_auc': float(np.nanmean(per_answer(pred, off, y, nonctl)['auc']))}
        with open(path, 'wb') as f: pickle.dump(out, f)
        print(bk, 'scores', round(time.time() - t0), 'parity', {k: f'{v:.1e}' for k, v in par.items()}, 'rank', rank, '/', X1.shape[1], 'R2', round(r2, 4), flush=True)
        del pop, bank, V, st; gc.collect()


# ------------------------------------------------------------------ stage aggregate
def cosine(a, b):
    na, nb_ = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb_)) if na > 0 and nb_ > 0 else np.nan


def spear(a, b):
    if np.ptp(a) == 0 or np.ptp(b) == 0: return np.nan
    return float(spearmanr(a, b).correlation)


def panel_values():
    """Our methods' ProcessBench hit and v1/v2 panel PRMBench numbers (v2 where complete)."""
    rows = []
    for src in (V2, V1):
        p = src / 'PANEL_A_ABSOLUTE.csv'
        if p.exists():
            d = pd.read_csv(p); d['source'] = str(p); rows.append(d)
    pa = pd.concat(rows).drop_duplicates(['bank', 'scorer'], keep='first') if rows else pd.DataFrame()
    orows = []
    for src in (V2, V1):
        p = src / 'ORIENTED_CONTROL.csv'
        if p.exists():
            d = pd.read_csv(p); d['source'] = str(p); orows.append(d)
    oc = pd.concat(orows).drop_duplicates(['bank'], keep='first')
    return pa, oc


def aggregate_stage():
    A, OFF, LAB, F, noncontrol = load_population(); del F
    cells = A.cell.to_numpy(); groups = A.source_group.to_numpy(); target = A.target.to_numpy().astype(int)
    prm_ans = np.flatnonzero(cells == PRM); pb_idx = np.flatnonzero(cells != PRM)
    pb_cells = sorted(set(cells[pb_idx])); fam = family_map()
    pg, pinv = np.unique(groups[prm_ans], return_inverse=True); bg, binv = np.unique(groups[pb_idx], return_inverse=True)
    rng = np.random.default_rng(SEED)
    Wp = np.vstack([np.ones(len(pg)), rng.multinomial(len(pg), np.full(len(pg), 1 / len(pg)), size=DRAWS)]).astype(float)
    Wb = np.vstack([np.ones(len(bg)), rng.multinomial(len(bg), np.full(len(bg), 1 / len(bg)), size=DRAWS)]).astype(float)

    def gsum(X, inv, G):
        out = np.zeros((G, X.shape[1])); np.add.at(out, inv, X); return out

    def prm_stats(st):
        a = st['auc']; t1 = st['top1err']
        X = np.column_stack([np.nan_to_num(a), np.isfinite(a), np.nan_to_num(t1), np.isfinite(t1)]).astype(float)
        S = Wp @ gsum(X, pinv, len(pg)); out = {'within_auc': S[:, 0] / S[:, 1], 'top1_on_error': S[:, 2] / S[:, 3]}
        c = Wp @ gsum(st['c1'].astype(float), pinv, len(pg)); tp, fp, tn, fn = c.T
        with np.errstate(invalid='ignore', divide='ignore'):
            out['prmscore_top1'] = .5 * (2 * tp / (2 * tp + fp + fn) + 2 * tn / (2 * tn + fp + fn))
        out['n_auc_answers'] = int(np.isfinite(a).sum())
        return out

    def pb_hit(am_full):
        """am_full: argmax per answer over the full population (PB rows used)."""
        hit = am_full[pb_idx] == target[pb_idx]; err = target[pb_idx] >= 0; cols = []
        for c in pb_cells: ck = cells[pb_idx] == c; cols += [hit & err & ck, err & ck]
        S = (Wb @ gsum(np.column_stack(cols).astype(float), binv, len(bg))).reshape(len(Wb), len(pb_cells), 2)
        return np.nanmean(S[:, :, 0] / S[:, :, 1], 1)

    pa, oc = panel_values()
    PA1 = np.load(V1 / 'PER_ANSWER.npz'); PA2 = np.load(V2 / 'PER_ANSWER.npz') if (V2 / 'PER_ANSWER.npz').exists() else None
    ci = lambda x: (float(x[0]), float(np.nanquantile(x[1:], .025)), float(np.nanquantile(x[1:], .975)))
    score_rows, coef_rows, comp_rows, gap_rows, parity = [], [], [], [], {'method_auc_vs_v1_max_abs_diff': {}, 'survivors_equal_v1': {}}
    fd = json.loads((V1 / 'FIT_DIAGNOSTICS.json').read_text(encoding='utf8'))
    lr_ref = pd.read_csv(MAIN / 'results/estimator_input_v1/run_20261004_resume/LR_METRICS.csv')
    lr_ref = lr_ref[(lr_ref.cell == PRM) & (lr_ref.score_scope == 'OOF') & (lr_ref.decision_rule == 'TOP_1') & (lr_ref.position_variant == 'CONTENT')
                    & (lr_ref.variant == 'channels_answer') & (lr_ref.recipe == 'LR_full')].set_index('bank').within_answer_auc
    for bk in BANKS:
        path = CACHE / f'{bk}.pkl'
        if not path.exists(): print('missing', bk); continue
        R = pickle.load(open(path, 'rb')); names = R['names']; n = len(names)
        full = lambda arr: arr
        # ---- scores
        stats = {'LR': prm_stats(R['lr']), 'LR_IN_SAMPLE': prm_stats(R['lr_in_sample'])}
        for k, st in R['methods'].items(): stats[k] = prm_stats(st)
        am = {}
        for tag in ('argmax_all', 'argmax_fold'):
            a_ = np.full(len(A), -9, int)
            for ce, r in R['pb'].items(): a_[r['answers']] = r[tag]
            am[tag] = pb_hit(a_)
        flips = int(sum(r['aligned_flips'] for r in R['pb'].values()))
        def pbrow(scorer):
            q = pa[(pa.bank == bk) & (pa.scorer == scorer)] if len(pa) else pa
            return (float(q.pb_argmax_hit.iloc[0]), q.source.iloc[0]) if len(q) and np.isfinite(q.pb_argmax_hit.iloc[0]) else (np.nan, '')
        for k, st in stats.items():
            row = {'bank': bk, 'n_channels': n, 'digits': bk in DIGIT_BANKS, 'scorer': k,
                   'role': 'supervised reference' if k.startswith('LR') else 'method' if k in METHODS4 + ['HEM_MOM_ga'] else 'control',
                   'prmb_within_auc': st['within_auc'][0], 'prmb_top1_on_error': st['top1_on_error'][0], 'prmb_prmscore_top1': st['prmscore_top1'][0],
                   'n_auc_answers': st['n_auc_answers'], 'prmb_source': 'this run (recomputed)'}
            if k == 'LR':
                row.update({'pb_argmax_hit': am['argmax_all'][0], 'pb_label': 'trained on PRMBench, applied to ProcessBench (all-fold model)',
                            'pb_argmax_hit_fold_matched': am['argmax_fold'][0], 'pb_aligned_column_flips': flips,
                            'C_per_fold': json.dumps([d['C'] for d in R['lr_diag']]), 'C_all_fold': R['lr_C_all'],
                            'nonconverged_inner_C': json.dumps([d['nonconverged_C'] for d in R['lr_diag']])})
            elif k == 'LR_IN_SAMPLE':
                row.update({'pb_label': 'in-sample diagnostic (all-fold model on its own training data)'})
            elif k == 'ALL_EQ_ORIENTED':
                q = oc[oc.bank == bk]
                row.update({'pb_argmax_hit': float(q.pb_argmax_hit.iloc[0]) if len(q) else np.nan, 'pb_source': q.source.iloc[0] if len(q) else 'pending (v2 not complete)'})
            else:
                h, s_ = pbrow(k); row.update({'pb_argmax_hit': h, 'pb_source': s_ or 'pending/incomplete'})
            score_rows.append(row)
        # ---- parity
        if bk in ('N13', 'N16'):
            parity[f'{bk}_LR_oof_within_auc_vs_estimator_input_v1_{ {"N13": "B13", "N16": "B16"}[bk]}'] = {
                'this_run': stats['LR']['within_auc'][0], 'reference': float(lr_ref[{'N13': 'B13', 'N16': 'B16'}[bk]])}
        for tag, PA in (('v1', PA1), ('v2', PA2)):
            if PA is None: continue
            diffs = {}
            for k in SCORERS:
                key = f'{bk}__{k}__auc'
                if key in PA.files and k in R['methods']:
                    diffs[k] = float(np.nanmax(np.abs(np.nan_to_num(PA[key][prm_ans], nan=-1) - np.nan_to_num(R['methods'][k]['auc'], nan=-1))))
            if diffs: parity.setdefault(f'method_auc_vs_{tag}_max_abs_diff', {})[bk] = diffs
        if f'{bk}/{PRM}' in fd: parity['survivors_equal_v1'][bk] = fd[f'{bk}/{PRM}'].get('survivors') == R['survivors']
        # ---- gaps
        meth = {k: stats[k]['within_auc'] for k in METHODS4 if k in stats}
        best = max(meth, key=lambda k: meth[k][0]) if meth else None
        g = {'bank': bk, 'n_channels': n, 'digits': bk in DIGIT_BANKS, 'lr_within_auc': stats['LR']['within_auc'][0],
             'best_method': best, 'best_within_auc': meth[best][0] if best else np.nan,
             'methods_min_auc': min(v[0] for v in meth.values()) if meth else np.nan, 'methods_mean_auc': float(np.mean([v[0] for v in meth.values()])) if meth else np.nan}
        for ref in ('best', 'MEAN_MOM_ga', 'ALL_EQ', 'ALL_EQ_ORIENTED', 'DS_FILTER_EQ'):
            r_ = meth[best] if ref == 'best' else stats[ref]['within_auc'] if ref in stats else None
            if r_ is None: continue
            d_ = stats['LR']['within_auc'] - r_; pt, lo, hi = ci(d_)
            g[f'gap_vs_{ref}'] = pt
            if ref in ('best', 'MEAN_MOM_ga', 'ALL_EQ', 'ALL_EQ_ORIENTED'): g[f'gap_vs_{ref}_lo95'] = lo; g[f'gap_vs_{ref}_hi95'] = hi
        # ProcessBench gap vs MEAN_MOM_ga with per-answer argmax from v1/v2 PER_ANSWER
        lr_am = np.full(len(A), -9, int)
        for ce, r in R['pb'].items(): lr_am[r['answers']] = r['argmax_all']
        for PA in (PA2, PA1):
            if PA is None: continue
            key = f'{bk}__MEAN_MOM_ga__argmax'; fk = f'{bk}__MEAN_MOM_ga__failed'
            if key in PA.files and not PA[fk][pb_idx].any():
                pt, lo, hi = ci(pb_hit(lr_am) - pb_hit(PA[key])); g.update({'pb_gap_vs_MEAN_MOM_ga': pt, 'pb_gap_vs_MEAN_MOM_ga_lo95': lo, 'pb_gap_vs_MEAN_MOM_ga_hi95': hi}); break
        g['pb_lr_hit'] = am['argmax_all'][0]
        mm = pa[(pa.bank == bk) & pa.scorer.isin(METHODS4)] if len(pa) else pa
        g['pb_methods_mean_hit'] = float(mm.pb_argmax_hit.mean()) if len(mm) and mm.pb_argmax_hit.notna().all() else np.nan
        g['pb_methods_max_hit'] = float(mm.pb_argmax_hit.max()) if len(mm) and mm.pb_argmax_hit.notna().all() else np.nan
        gap_rows.append(g)
        # ---- coefficients
        CE = np.array([d['coef_eff'] for d in R['lr_diag']]); CS = np.array([d['coef_std'] for d in R['lr_diag']])
        w = CE.mean(0); keep = np.array([c in R['survivors'] for c in names], float)
        SP = CACHE / f'{bk}_scores.pkl'; SC_ = pickle.load(open(SP, 'rb')) if SP.exists() else None
        imp_ols = R.get('implied_mean_mom', np.full(n, np.nan))
        imp = SC_['implied_minnorm'] if SC_ else imp_ols                     # min-norm truncated SVD (rcond 1e-6): stable under near-duplicate channels
        cov = SC_['cov_within'] if SC_ else None
        if SC_: parity.setdefault('scores_stage_recomputed_auc_max_abs_diff', {})[bk] = SC_['auc_parity_max_abs_diff']
        ksd = float(np.sqrt((w @ cov @ w) / (imp @ cov @ imp))) if cov is not None else np.nan   # implied scaled to the LR step-score SD
        for j, c in enumerate(names):
            coef_rows.append({'bank': bk, 'channel': c, 'family': fam.get(c, '?'), 'coef_mean': w[j], 'coef_sd_folds': CE[:, j].std(ddof=1),
                              'coef_std_mean': CS[:, j].mean(), 'coef_std_sd_folds': CS[:, j].std(ddof=1), 'coef_all_fold_eff': R['lr_all_coef_eff'][j],
                              'sign_stable_5_of_5': bool(np.all(np.sign(CE[:, j]) == np.sign(w[j]))), 'oriented_sign': R['oriented_signs'][j],
                              'ds_kept': bool(keep[j]), 'implied_mean_mom_weight': imp[j], 'implied_mean_mom_weight_plain_ols': imp_ols[j],
                              'implied_mean_mom_weight_scaled_to_lr_score_sd': imp[j] * ksd})
        pairs = [cosine(CE[a], CE[b]) for a, b in itertools.combinations(range(5), 2)]
        vecs = {'ALL_EQ': np.ones(n), 'ORIENTED': R['oriented_signs'], 'DS_KEEP': keep, 'MEAN_MOM_IMPLIED': imp}
        top = np.argsort(-np.abs(w))[:5]
        for vn, v in vecs.items():
            if not np.isfinite(v).all(): continue
            if vn == 'DS_KEEP': sa = float(np.mean((w > 0) == (keep > 0)))
            else: sa = float(np.mean(np.sign(w) == np.sign(v)))
            sk = {'ALL_EQ': 'ALL_EQ', 'ORIENTED': 'ALL_EQ_ORIENTED', 'DS_KEEP': 'DS_FILTER_EQ', 'MEAN_MOM_IMPLIED': 'MEAN_MOM_ga'}[vn]
            ccov = float(w @ cov @ v / np.sqrt((w @ cov @ w) * (v @ cov @ v))) if cov is not None else np.nan
            comp_rows.append({'bank': bk, 'n_channels': n, 'label_free_vector': vn, 'cosine': cosine(w, v), 'spearman': spear(w, v),
                              'score_corr_within': SC_['score_corr_within'][sk] if SC_ else np.nan, 'score_corr_scorer': sk,
                              'cosine_in_channel_covariance': ccov,
                              'sign_agreement': sa, 'lr_mass_on_ds_dropped': float(np.abs(w)[keep == 0].sum() / np.abs(w).sum()),
                              'n_ds_dropped': int((keep == 0).sum()), 'lr_fold_mean_pairwise_cosine': float(np.mean(pairs)),
                              'lr_channels_sign_stable_5_of_5': int(sum(np.all(np.sign(CE[:, j]) == np.sign(w[j])) for j in range(n))),
                              'lr_negative_channels': int((w < 0).sum()),
                              'implied_r2': SC_['implied_r2_minnorm'] if SC_ else R.get('implied_r2', np.nan),
                              'implied_within_auc_check': SC_['implied_minnorm_within_auc'] if SC_ else R.get('implied_within_auc_check', np.nan),
                              'implied_r2_plain_ols': R.get('implied_r2', np.nan), 'implied_lstsq_rank': SC_['lstsq_rank'] if SC_ else np.nan,
                              'top5_lr_channels': '; '.join(f'{names[j]} ({fam.get(names[j], "?")}, {w[j]:+.3f})' for j in top)})
    pd.DataFrame(score_rows).to_csv(OUT / 'LR_SCORES.csv', index=False)
    pd.DataFrame(coef_rows).to_csv(OUT / 'LR_COEFFICIENTS.csv', index=False)
    pd.DataFrame(comp_rows).to_csv(OUT / 'COEF_COMPARISON.csv', index=False)
    pd.DataFrame(gap_rows).to_csv(OUT / 'GAPS.csv', index=False)
    parity['panel_sources'] = sorted(set(pa.source)) if len(pa) else []
    (OUT / 'PARITY.json').write_text(json.dumps(parity, indent=1, default=float), encoding='utf8')
    print(pd.DataFrame(gap_rows).round(4).to_string())


# ------------------------------------------------------------------ stage figures
PAL = {'light': {'bg': '#fcfcfb', 'ink': '#0b0b0b', 'ours': '#2a78d6', 'lr': '#eb6834', 'third': '#1baf7a', 'gray': '#898781', 'grid': '#e4e3df', 'muted': '#5b5a56'},
       'dark': {'bg': '#1a1a19', 'ink': '#ffffff', 'ours': '#3987e5', 'lr': '#d95926', 'third': '#199e70', 'gray': '#898781', 'grid': '#3a3a38', 'muted': '#b5b4ae'}}


def _style(ax, p):
    ax.set_facecolor(p['bg'])
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'): ax.spines[s].set_color(p['muted'])
    ax.tick_params(colors=p['ink'], labelsize=10); ax.yaxis.label.set_color(p['ink']); ax.xaxis.label.set_color(p['ink'])
    ax.title.set_color(p['ink']); ax.grid(axis='y', color=p['grid'], lw=.7); ax.set_axisbelow(True)


def _save(fig, name, p, mode):
    fig.patch.set_facecolor(p['bg'])
    for ext_ in ('png', 'svg'):
        fig.savefig(OUT / 'figures' / f'{name}_{mode}.{ext_}', dpi=200, facecolor=p['bg'], bbox_inches='tight')


def figures_stage():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    (OUT / 'figures').mkdir(exist_ok=True)
    S = pd.read_csv(OUT / 'LR_SCORES.csv'); C = pd.read_csv(OUT / 'COEF_COMPARISON.csv'); K = pd.read_csv(OUT / 'LR_COEFFICIENTS.csv')
    order = PLOT_ORDER[0] + PLOT_ORDER[1]; xs = np.r_[np.arange(5), np.arange(5) + 6]
    ncount = {b: int(S[S.bank == b].n_channels.iloc[0]) for b in order if (S.bank == b).any()}
    for mode, p in PAL.items():
        plt.rcParams.update({'font.family': 'DejaVu Sans', 'text.color': p['ink']})
        # ---- L1
        fig, axes = plt.subplots(1, 2, figsize=(15, 4.6))
        for ax, metric, ylab in ((axes[0], 'prmb_within_auc', 'PRMBench within-answer AUC'), (axes[1], 'pb_argmax_hit', 'ProcessBench top-step first-error hit')):
            _style(ax, p)
            for x, b in zip(xs, order):
                q = S[S.bank == b]
                if not len(q): continue
                m = q[q.scorer.isin(METHODS4)][metric].dropna()
                if len(m) == 4:
                    ax.plot([x - .1, x - .1], [m.min(), m.max()], color=p['ours'], lw=6, alpha=.35, solid_capstyle='round')
                    ax.plot(x - .1, m.mean(), 'o', color=p['ours'], ms=7, zorder=4)
                for sc_, mk, off_ in (('ALL_EQ', 's', -.32), ('ALL_EQ_ORIENTED', 'D', .32)):
                    v = q[q.scorer == sc_][metric]
                    if len(v) and np.isfinite(v.iloc[0]): ax.plot(x + off_, v.iloc[0], mk, color=p['gray'], ms=6, mfc='none' if sc_ == 'ALL_EQ' else p['gray'], mew=1.5, zorder=3)
                v = q[q.scorer == 'LR'][metric]
                if len(v) and np.isfinite(v.iloc[0]): ax.plot(x + .12, v.iloc[0], '*', color=p['lr'], ms=14, zorder=5)
                if metric == 'pb_argmax_hit' and not (len(m) == 4):
                    ax.annotate('ours: n/a*' if b == 'N11' else 'ours: pending', (x, .405), ha='center', fontsize=8, color=p['muted'])
            ax.set_xticks(xs); ax.set_xticklabels([f'{b}\n({ncount.get(b, "?")} ch)' for b in order], fontsize=9)
            ax.set_ylabel(ylab, fontsize=11)
            ylo, yhi = ax.get_ylim(); ax.set_ylim(ylo, yhi + (yhi - ylo) * .12)
            for x0, lab in ((2, 'no digit channels'), (8, '+ 3 digit channels')):
                ax.text(x0, yhi + (yhi - ylo) * .05, lab, ha='center', fontsize=10, color=p['muted'])
            ax.axvline(5, color=p['grid'], lw=1)
        h = [Line2D([], [], marker='*', ls='', color=p['lr'], ms=13, label='Supervised LR (uses labels)'),
             Line2D([], [], marker='o', ls='-', lw=5, alpha=.6, color=p['ours'], ms=7, label='Our 4 label-free methods (range, mean)'),
             Line2D([], [], marker='s', ls='', color=p['gray'], mfc='none', mew=1.5, label='Equal weights (all +1)'),
             Line2D([], [], marker='D', ls='', color=p['gray'], label='Oriented equal weights')]
        leg = fig.legend(handles=h, loc='lower center', ncol=4, frameon=False, fontsize=10, bbox_to_anchor=(.5, -.04))
        for t in leg.get_texts(): t.set_color(p['ink'])
        axes[1].text(1, -.2, 'ProcessBench LR: trained on PRMBench, applied to ProcessBench.  *N11: one ProcessBench cell has only 2 groups, methods undefined', transform=axes[1].transAxes, ha='right', fontsize=8, color=p['muted'])
        fig.tight_layout(rect=(0, .06, 1, 1)); _save(fig, 'L1_lr_vs_labelfree', p, mode); plt.close(fig)
        # ---- L2: two panels, weight space (left) and step-score space (right); same 0-1 scale, no dual axis
        fig, axes = plt.subplots(1, 2, figsize=(16, 4.8), sharey=True)
        panels = [(axes[0], 'Channel weights (per-channel comparison)',
                   [('MEAN_MOM_IMPLIED', 'cosine', 'cosine(LR, implied MEAN_MOM weights)', p['ours'], 'o'),
                    ('ORIENTED', 'cosine', 'cosine(LR, oriented signs)', p['third'], 's'),
                    ('ORIENTED', 'sign_agreement', 'sign agreement (LR vs oriented signs)', p['gray'], '^'),
                    ('ALL_EQ', 'lr_mass_on_ds_dropped', 'share of LR |weight| on DS-dropped channels', p['lr'], 'D')]),
                  (axes[1], 'Step scores (within-answer correlation)',
                   [('MEAN_MOM_IMPLIED', 'score_corr_within', 'corr(LR score, MEAN_MOM score)', p['ours'], 'o'),
                    ('ORIENTED', 'score_corr_within', 'corr(LR score, oriented equal score)', p['third'], 's'),
                    ('ALL_EQ', 'score_corr_within', 'corr(LR score, equal-weight score)', p['gray'], '^')])]
        for ax, title, series in panels:
            _style(ax, p); ax.set_title(title, fontsize=12, loc='left')
            for vn, col, lab, color, mk in series:
                ys = []
                for b in order:
                    q = C[(C.bank == b) & (C.label_free_vector == vn)]
                    ys.append(q[col].iloc[0] if len(q) else np.nan)
                for blk in (slice(0, 5), slice(5, 10)):
                    ax.plot(xs[blk], np.array(ys)[blk], marker=mk, color=color, lw=1.8, ms=7, label=lab if blk.start == 0 else None)
            ax.set_xticks(xs); ax.set_xticklabels([f'{b}\n({ncount.get(b, "?")} ch)' for b in order], fontsize=9)
            ax.set_ylim(-.45, 1.1); ax.axvline(5, color=p['grid'], lw=1); ax.axhline(0, color=p['muted'], lw=.6)
            for x0, lab in ((2, 'no digit channels'), (8, '+ 3 digit channels')): ax.text(x0, 1.04, lab, ha='center', fontsize=10, color=p['muted'])
            leg = ax.legend(loc='upper center', bbox_to_anchor=(.5, -.15), ncol=2, frameon=False, fontsize=9.5)
            for t in leg.get_texts(): t.set_color(p['ink'])
        axes[0].set_ylabel('agreement (1 = same direction)', fontsize=11)
        fig.tight_layout(); _save(fig, 'L2_coef_agreement', p, mode); plt.close(fig)
        # ---- L3
        for b in ('N16', 'N56'):
            q = K[K.bank == b].reset_index(drop=True)
            if not len(q): continue
            lr = q.coef_mean.to_numpy(); im = q.implied_mean_mom_weight_scaled_to_lr_score_sd.to_numpy()
            fig, ax = plt.subplots(figsize=(max(9, .32 * len(q) + 3), 5.4)); _style(ax, p)
            x = np.arange(len(q)); wdt = .4
            lim = max(1.25 * np.quantile(np.abs(lr), .85), 1.15 * np.abs(im).max())
            clipped = np.abs(lr) > lim
            ax.bar(x - wdt / 2, np.clip(lr, -lim, lim), wdt, color=p['lr'], label='Supervised LR weight (mean of 5 folds)')
            ax.bar(x + wdt / 2, im, wdt, color=p['ours'], label='Implied MEAN_MOM weight (scaled: same step-score SD as LR)')
            ax.set_ylim(-lim * 1.08, lim * 1.08)
            for jj in np.flatnonzero(clipped):
                ax.text(jj - wdt / 2, np.sign(lr[jj]) * lim * .96, f'{lr[jj]:+.1f}', ha='center', va='top' if lr[jj] > 0 else 'bottom', fontsize=8,
                        color=p['ink'], rotation=90, bbox={'fc': p['bg'], 'ec': 'none', 'pad': 1})
            for jj in np.flatnonzero(~q.ds_kept.to_numpy()): ax.text(jj, -lim * 1.06, '×', ha='center', va='bottom', color=p['muted'], fontsize=10)
            ax.axhline(0, color=p['muted'], lw=.8)
            ax.set_xticks(x); ax.set_xticklabels(q.channel, rotation=60, ha='right', fontsize=8 if len(q) > 20 else 9)
            ax.set_ylabel('weight on answer-z channel', fontsize=11)
            note = '× = dropped by the DS filter' + ('; LR bars with a printed value are cut at the axis' if clipped.any() else '')
            ax.set_title(f'{b}: supervised vs label-free channel weights on PRMBench\n{note}', fontsize=11, loc='left', pad=34)
            leg = ax.legend(frameon=False, fontsize=9.5, loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.2)
            for t in leg.get_texts(): t.set_color(p['ink'])
            fig.tight_layout(); _save(fig, f'L3_weights_{b}', p, mode); plt.close(fig)
    print('figures written')


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else 'all'
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        if stage in ('fit', 'all'): fit_stage()
        if stage in ('scores', 'all'): scores_stage()
        if stage in ('aggregate', 'all'): aggregate_stage()
        if stage in ('figures', 'all'): figures_stage()
