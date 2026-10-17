# PATCHED COPY of results/monday_lr_weights_v1/run.py (thinkfix_rescore_v1, variant=old): REF -> this variant's lr_reference; N16 0.822366 gate enforced only for variant=old.
"""monday_lr_weights_v1: which channels get ~zero weight in the supervised LR reference, and is that zero set unique?

SUPERVISED analysis of monday_lr_reference_v1 (uses PRMBench step labels). Development data, CPU, no new inference.
Stages:
  gate            N16 nested LR refit must reproduce OOF within-AUC 0.822366 and the cached fold coefficients;
                  every bank's cached fold models are replayed on the rebuilt V (per-answer AUC identical).
  bank <BANK>     per bank -> cache/<BANK>.pkl: correlations, L2 all-fold path, fixed-C drop-one refits,
                  L1 path, restricted nested LR refits (R1/R2/R3 kept sets).
  aggregate       ZERO_WEIGHTS.csv, L1_PATH.csv, PAIRS.csv, RESTRICTED.csv, figures/W1.
Protocol: PROTOCOL.json (written before the run).
"""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '2'
import importlib.util, json, math, pickle, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

MAIN = Path(r'C:\Users\omris\TAU\hallucination_detection'); OUT = Path(__file__).resolve().parent; CACHE = OUT / 'cache'
REF = Path(r'/shared/cycle2_tau_averbuch_prj/omrisegev1/results/thinkfix_rescore_v1/old/lr_reference')
_spec = importlib.util.spec_from_file_location('lr_ref', REF / 'run_patched.py'); ref = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ref)
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

BANKS = ['N11', 'N13', 'N22', 'N40', 'N53', 'N14', 'N16', 'N25', 'N43', 'N56']
L1_C = np.logspace(-5, -1, 13)
NZ = 1e-8; SEED, DRAWS = 20261007, 1000

FAMILY_NAMES = {'F1': 'uncertainty level', 'F2': 'written-token surprise', 'F3': 'change over time', 'F4': 'shape within step',
                'F5': 'rhythm (rolling spectral)', 'F6': 'variability / drift', 'F7': 'digits'}
_FAM = {
    'F1': ['q15_H1', 'q15_VE1', 'logprob_margin', 'true_tail50', 'energy_level', 'ct7_H0lim', 'ct7_ve0', 'ct7_ve0.75', 'ct7_ve1',
           'hist_entropy_series', 'hist_topk_entropy_series', 'hist_topk_varentropy_series', 'hist_topk_renyi2_series',
           'hist_topk_tail_mass_series', 'hist_energy_series', 'hist_energy_rolling_min', 'hist_top1_logprob_series',
           'hist_logprob_margin_series'],
    'F2': ['chosen_surprisal', 'realized_z', 'realized_drv', 'ct7_chosen_std_excess', 'hist_spilled_series', 'hist_spilled_rolling_min'],
    'F3': ['energy_innovation', 'top15_turnover', 'top50_js', 'bocpd_p0', 'ct7_H0lim_prefix_innovation', 'ct7_bocpd_residual', 'evidence_drop_risk'],
    'F4': ['H1_first_token', 'H1_slope', 'H1_jump', 'H1_frac_above_z'],
    'F5': ['dominant_freq16', 'hist_entropy_rolling_spectral_entropy', 'hist_entropy_rolling_low_band_power', 'hist_entropy_rolling_high_band_power',
           'hist_entropy_rolling_hl_ratio', 'hist_entropy_rolling_dominant_freq', 'hist_entropy_rolling_spectral_centroid',
           'hist_entropy_stft_high_series', 'hist_entropy_stft_frame_entropy', 'hist_entropy_rolling_tail_ratio', 'hist_entropy_pe_series',
           'hist_entropy_rolling_rs_hurst'],
    'F6': ['hist_entropy_sw_var_series', 'hist_entropy_cusum_abs_series', 'hist_spilled_sw_var_series', 'hist_spilled_cusum_abs_series',
           'hist_energy_sw_var_series', 'hist_energy_cusum_abs_series'],
    'F7': ['digit_alternative', 'digit_spread', 'digit_alternative_innovation']}
FAMILY = {c: f for f, cs in _FAM.items() for c in cs}
assert len(FAMILY) == 56 and set(FAMILY) == set(ref.NB['N56'])


# ------------------------------------------------------------------ data
_POP = {}


def population():
    if not _POP:
        A, OFF, LAB, F, nonctl = ref.load_population()
        _POP.update(A=A, OFF=OFF, LAB=LAB, X=F['values'].astype(float), nm=[str(x) for x in F['names']],
                    DA=F['digit_active'].astype(bool), nonctl=nonctl)
    return _POP


def bank_data(bk):
    P = population(); A = P['A']; OFF = P['OFF']
    prm = np.flatnonzero(A.cell.to_numpy() == ref.PRM); ns = np.diff(OFF)
    rows = np.concatenate([np.arange(OFF[i], OFF[i + 1]) for i in prm]); off = np.r_[0, np.cumsum(ns[prm])]
    Vall = ref.ext.build_bank(ref.NB[bk], P['X'], P['nm'], P['DA'], OFF)
    V = Vall[rows]; y = P['LAB'][rows].astype(int); folds = np.repeat(A.fold.to_numpy().astype(int)[prm], ns[prm])
    return dict(V=V, y=y, folds=folds, off=off, nonctl=P['nonctl'][prm], names=list(ref.NB[bk]), groups=A.source_group.to_numpy()[prm])


def within_auc_answers(score, d):
    return ref.per_answer(score, d['off'], d['y'].astype(bool), d['nonctl'])['auc']


def load_ref(bk):
    return pickle.load(open(REF / 'cache' / f'{bk}.pkl', 'rb'))


# ------------------------------------------------------------------ gate
def gate_stage():
    out = {}
    for bk in BANKS:
        R = load_ref(bk); d = bank_data(bk)
        assert R['names'] == d['names'], bk
        s = np.zeros(len(d['V']))
        for dg in R['lr_diag']:
            m = d['folds'] == dg['outer_fold']
            s[m] = ((d['V'][m] - np.array(dg['mean'])) / np.array(dg['scale'])) @ np.array(dg['coef_std']) + dg['intercept']
        a = within_auc_answers(s, d)
        diff = float(np.nanmax(np.abs(np.nan_to_num(a, nan=-1) - np.nan_to_num(R['lr']['auc'], nan=-1))))
        out[bk] = {'replay_max_abs_per_answer_auc_diff': diff, 'oof_auc_cache': float(np.nanmean(R['lr']['auc']))}
        print(bk, 'replay diff', diff, flush=True)
        if diff > 1e-9: raise SystemExit(f'V mismatch {bk}')
    d = bank_data('N16'); t0 = time.time()
    oof, diag, _, _, c_all = ref.nested_lr(d['V'], d['y'], d['folds'])
    auc = float(np.nanmean(within_auc_answers(oof, d))); R = load_ref('N16')
    cdiff = max(float(np.max(np.abs(np.array(a['coef_std']) - np.array(b['coef_std'])))) for a, b in zip(diag, R['lr_diag']))
    out['N16_refit'] = {'oof_within_auc': auc, 'target': 0.822366, 'abs_diff': abs(auc - 0.822366), 'max_coef_diff_vs_cache': cdiff,
                        'C': [x['C'] for x in diag], 'runtime_s': round(time.time() - t0, 1)}
    print('N16 refit', out['N16_refit'], flush=True)
    out['N16_refit']['enforced'] = _ENFORCE_N16
    if abs(auc - 0.822366) > 1e-6 and _ENFORCE_N16: raise SystemExit('GATE FAIL: N16 AUC not reproduced')
    (OUT / 'GATE.json').write_text(json.dumps(out, indent=1), encoding='utf8')


# ------------------------------------------------------------------ helpers
def ref_weights(R):
    W = np.array([dg['coef_std'] for dg in R['lr_diag']])            # 5 x P
    return W, [dg['C'] for dg in R['lr_diag']]


def contrast_pairs(w, corr, names, thr=0.8):
    med = np.median(np.abs(w)); out = []
    P = len(w)
    for i in range(P):
        for j in range(i + 1, P):
            if abs(corr[i, j]) >= thr and np.sign(w[i]) * np.sign(w[j]) * np.sign(corr[i, j]) < 0 and abs(w[i]) > med and abs(w[j]) > med:
                out.append((names[i], names[j]))
    return out


def kept_sets(w, corr, names):
    a = np.abs(w); P = len(w)
    r1 = a >= 0.05 * a.max()
    cp = contrast_pairs(w, corr, names); inpair = {c for p in cp for c in p}
    ref_max = max([a[k] for k in range(P) if names[k] not in inpair] or [a.max()])
    r2 = a >= 0.05 * ref_max
    nq = math.ceil(P / 4); bottom = np.argsort(a, kind='stable')[:nq]; r3 = np.ones(P, bool); r3[bottom] = False
    return {'R1': r1, 'R2': r2, 'R3': r3}, cp, float(ref_max)


def fit_fixed(V, y, folds, Cs, cols):
    """Outer-fold L2 fits with fixed per-fold C (reference model_fit); OOF scores + standardized coefs."""
    oof = np.zeros(len(V)); W = np.full((5, V.shape[1]), np.nan); ok = True
    for k in range(5):
        tr = folds != k; te = folds == k
        try:
            sc, m = ref.model_fit(V[tr][:, cols], y[tr], Cs[k])
        except ConvergenceWarning:
            ok = False; continue
        oof[te] = m.decision_function(sc.transform(V[te][:, cols])); W[k, cols] = m.coef_[0]
    return oof, W, ok


def l1_fit(Xt, y, C):
    m = LogisticRegression(l1_ratio=1.0, C=C, solver='liblinear', class_weight='balanced', max_iter=5000, tol=1e-4,
                           random_state=20261003, intercept_scaling=1.)
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter('always', ConvergenceWarning); m.fit(Xt, y)
    return m, any(issubclass(x.category, ConvergenceWarning) for x in wl)


# ------------------------------------------------------------------ bank stage
def bank_stage(bk):
    CACHE.mkdir(exist_ok=True); path = CACHE / f'{bk}.pkl'
    if path.exists(): print(bk, 'cached'); return
    t0 = time.time(); R = load_ref(bk); d = bank_data(bk); V, y, folds = d['V'], d['y'], d['folds']; names = d['names']; P = len(names)
    _POP.clear(); import gc; gc.collect()                                  # memory: keep only this bank's PRMBench rows
    Wf, Cs = ref_weights(R); w = Wf.mean(0)
    corr = np.corrcoef(V, rowvar=False)
    res = {'bank': bk, 'names': names, 'W_folds': Wf, 'C_folds': Cs, 'corr': corr, 'oof_auc_ref': R['lr']['auc']}
    sets, cp, ref_max = kept_sets(w, corr, names); res['sets'] = sets; res['contrast_pairs'] = cp; res['R2_ref_max'] = ref_max
    # L2 all-fold path over C_GRID (standardized coefs)
    path_l2 = {}
    sc_all = StandardScaler().fit(V)
    for C in ref.C_GRID:
        try:
            _, m = ref.model_fit(V, y, C); path_l2[C] = m.coef_[0].copy()
        except ConvergenceWarning:
            path_l2[C] = None
    res['l2_path'] = path_l2; print(bk, 'l2 path', round(time.time() - t0), flush=True)
    # fixed-C full + drop-one
    oof, Wfull, ok = fit_fixed(V, y, folds, Cs, np.arange(P))
    res['fixed_full'] = {'auc': within_auc_answers(oof, d), 'W': Wfull, 'ok': ok}
    drop = {}
    for j in range(P):
        cols = np.array([k for k in range(P) if k != j]); o, Wd, okd = fit_fixed(V, y, folds, Cs, cols)
        drop[names[j]] = {'auc': within_auc_answers(o, d), 'W': Wd, 'ok': okd}
    res['drop_one'] = drop; print(bk, 'drop-one', round(time.time() - t0), flush=True)
    # L1 path
    l1 = {'C': L1_C, 'coef': np.zeros((len(L1_C), 5, P)), 'auc': [], 'nonconv': []}
    for k in range(5):
        tr = folds != k; sc = StandardScaler().fit(V[tr]); Xt = sc.transform(V[tr]); Xte = sc.transform(V[folds == k])
        for ci, C in enumerate(L1_C):
            m, nc = l1_fit(Xt, y[tr], C); l1['coef'][ci, k] = m.coef_[0]
            if nc: l1['nonconv'].append((k, float(C)))
            l1.setdefault('scores', np.zeros((len(L1_C), len(V))))[ci, folds == k] = m.decision_function(Xte)
        print(bk, 'l1 fold', k, round(time.time() - t0), flush=True)
    l1['auc'] = [within_auc_answers(l1['scores'][ci], d) for ci in range(len(L1_C))]; del l1['scores']
    res['l1'] = l1
    # restricted nested LR (identical spec)
    restr = {}
    for tag, keep in sets.items():
        cols = np.flatnonzero(keep)
        if len(cols) == P: restr[tag] = {'same_as_full': True, 'cols': cols}; continue
        o, diag, _, _, _ = ref.nested_lr(V[:, cols], y, folds)
        restr[tag] = {'cols': cols, 'auc': within_auc_answers(o, d), 'C': [x['C'] for x in diag],
                      'W': np.array([x['coef_std'] for x in diag])}
        print(bk, 'restricted', tag, len(cols), round(float(np.nanmean(restr[tag]['auc'])), 4), round(time.time() - t0), flush=True)
    res['restricted'] = restr; res['runtime_s'] = round(time.time() - t0)
    with open(path.with_suffix('.tmp'), 'wb') as f: pickle.dump(res, f)
    path.with_suffix('.tmp').replace(path); print(bk, 'done', res['runtime_s'], flush=True)


# ------------------------------------------------------------------ aggregate
def aggregate_stage():
    P0 = population(); A = P0['A']; prm = np.flatnonzero(A.cell.to_numpy() == ref.PRM); groups = A.source_group.to_numpy()[prm]
    pg, pinv = np.unique(groups, return_inverse=True); rng = np.random.default_rng(SEED)
    Wp = np.vstack([np.ones(len(pg)), rng.multinomial(len(pg), np.full(len(pg), 1 / len(pg)), size=DRAWS)]).astype(float)

    def boot(a):
        X = np.column_stack([np.nan_to_num(a), np.isfinite(a)]).astype(float); G = np.zeros((len(pg), 2)); np.add.at(G, pinv, X)
        S = Wp @ G; return S[:, 0] / S[:, 1]

    def diff_ci(a, b):
        x = boot(a) - boot(b); return float(x[0]), float(np.quantile(x[1:], .025)), float(np.quantile(x[1:], .975))

    zrows, l1rows, prows, rrows, summ = [], [], [], [], {}
    for bk in BANKS:
        p = CACHE / f'{bk}.pkl'
        if not p.exists(): print('missing', bk); continue
        R = pickle.load(open(p, 'rb')); names = R['names']; P = len(names); Wf = R['W_folds']; w = Wf.mean(0); a = np.abs(w)
        corr = R['corr']; sets = R['sets']; full_auc = R['oof_auc_ref']; fixed_auc = R['fixed_full']['auc']; base_fixed = float(np.nanmean(fixed_auc))
        rank_abs = (-a).argsort().argsort() + 1
        # L1 entry C per fold
        coef = R['l1']['coef']; Cg = R['l1']['C']; entry = np.full((5, P), np.inf)
        for k in range(5):
            for j in range(P):
                nz = np.abs(coef[:, k, j]) > NZ
                if nz[-1]:
                    last_zero = np.flatnonzero(~nz)
                    entry[k, j] = Cg[last_zero[-1] + 1] if len(last_zero) else Cg[0]
        # drop-out order: larger entry C -> drops first; rank 1 = first to drop
        drop_rank = np.zeros((5, P))
        for k in range(5):
            e = np.where(np.isfinite(entry[k]), entry[k], 1e9)
            drop_rank[k] = pd.Series(-e).rank(method='average').to_numpy()
        for j, c in enumerate(names):
            dr = R['drop_one'][c]
            near_partner = [names[i] for i in np.argsort(-np.abs(corr[j])) if i != j][:1]
            zrows.append({'bank': bk, 'channel': c, 'family': FAMILY[c], 'family_name': FAMILY_NAMES[FAMILY[c]],
                          'w_std_mean': w[j], 'w_std_sd_folds': Wf[:, j].std(ddof=1), 'w_min_folds': Wf[:, j].min(), 'w_max_folds': Wf[:, j].max(),
                          'fold_C': '/'.join(f'{x:g}' for x in R['C_folds']),
                          'sign_same_as_mean_k_of_5': int((np.sign(Wf[:, j]) == np.sign(w[j])).sum()),
                          'fold_range_crosses_0': bool(Wf[:, j].min() < 0 < Wf[:, j].max()),
                          'abs_w_rank': int(rank_abs[j]), 'abs_w_share_of_max': a[j] / a.max(), 'abs_w_share_of_R2_ref': a[j] / R['R2_ref_max'],
                          'near_zero_R1_5pct_of_max': bool(not sets['R1'][j]), 'near_zero_R2_contrast_robust': bool(not sets['R2'][j]),
                          'bottom_quartile_R3': bool(not sets['R3'][j]),
                          'in_contrast_pair': any(c in pq for pq in R['contrast_pairs']),
                          'most_correlated_partner': near_partner[0], 'corr_with_partner': float(corr[j, names.index(near_partner[0])]),
                          'drop_one_auc_loss': base_fixed - float(np.nanmean(dr['auc'])),
                          'l1_entry_C_median': float(np.median(entry[:, j])), 'l1_dropout_rank_median': float(np.median(drop_rank[:, j])),
                          'l1_dropout_rank_min': float(drop_rank[:, j].min()), 'l1_dropout_rank_max': float(drop_rank[:, j].max())})
        nnz = (np.abs(coef) > NZ).sum(2)                                   # C x fold
        for ci, C in enumerate(Cg):
            ent = [names[j] for j in range(P) if np.median(entry[:, j]) == C]
            l1rows.append({'bank': bk, 'C': C, 'n_nonzero_mean': nnz[ci].mean(), 'n_nonzero_min': int(nnz[ci].min()), 'n_nonzero_max': int(nnz[ci].max()),
                           'oof_within_auc': float(np.nanmean(R['l1']['auc'][ci])), 'full_lr_oof_auc': float(np.nanmean(full_auc)),
                           'channels_entering_at_this_C_median': '; '.join(ent),
                           'nonconverged_folds': sum(1 for kk, cc in R['l1']['nonconv'] if abs(cc - C) < 1e-15)})
        # pairs
        pairset = set()
        for i in range(P):
            for j in range(i + 1, P):
                if abs(corr[i, j]) > 0.9: pairset.add((i, j, 'near_duplicate'))
        for (ca, cb) in R['contrast_pairs']:
            i, j = names.index(ca), names.index(cb)
            if abs(corr[i, j]) <= 0.9: pairset.add((min(i, j), max(i, j), 'contrast_pair'))
        for i, j, kind in sorted(pairset):
            s = np.sign(corr[i, j]) or 1.; ssum = Wf[:, i] + s * Wf[:, j]
            l2p = {C: v for C, v in R['l2_path'].items() if v is not None}
            first = []
            for k in range(5):
                ei, ej = entry[k, i], entry[k, j]
                first.append(names[i] if ei < ej else names[j] if ej < ei else 'same C')   # smaller entry C = enters first
            di = R['drop_one'][names[j]]; dj = R['drop_one'][names[i]]          # drop partner j -> survivor i
            prows.append({'bank': bk, 'a': names[i], 'b': names[j], 'kind': kind, 'corr': corr[i, j], 'family_a': FAMILY[names[i]], 'family_b': FAMILY[names[j]],
                          'w_a_folds': '/'.join(f'{x:+.3f}' for x in Wf[:, i]), 'w_b_folds': '/'.join(f'{x:+.3f}' for x in Wf[:, j]),
                          'w_a_mean': Wf[:, i].mean(), 'w_b_mean': Wf[:, j].mean(), 'sd_a': Wf[:, i].std(ddof=1), 'sd_b': Wf[:, j].std(ddof=1),
                          'signed_sum_mean': ssum.mean(), 'signed_sum_sd': ssum.std(ddof=1),
                          'l2_path_w_a': '/'.join(f'{C:g}:{v[i]:+.3f}' for C, v in l2p.items()),
                          'l2_path_w_b': '/'.join(f'{C:g}:{v[j]:+.3f}' for C, v in l2p.items()),
                          'l1_first_entering_per_fold': '/'.join(first),
                          'drop_b_survivor_a_w_mean': np.nanmean(di['W'][:, i]), 'drop_a_survivor_b_w_mean': np.nanmean(dj['W'][:, j]),
                          'drop_b_auc_loss': base_fixed - float(np.nanmean(di['auc'])), 'drop_a_auc_loss': base_fixed - float(np.nanmean(dj['auc']))})
        # restricted
        for tag, rr in R['restricted'].items():
            if rr.get('same_as_full'):
                rrows.append({'bank': bk, 'kept_set': tag, 'n_kept': P, 'n_dropped': 0, 'oof_auc_restricted': float(np.nanmean(full_auc)),
                              'oof_auc_full': float(np.nanmean(full_auc)), 'diff': 0., 'lo': 0., 'hi': 0., 'dropped': ''}); continue
            dpt, lo, hi = diff_ci(rr['auc'], full_auc)
            rrows.append({'bank': bk, 'kept_set': tag, 'n_kept': len(rr['cols']), 'n_dropped': P - len(rr['cols']),
                          'oof_auc_restricted': float(np.nanmean(rr['auc'])), 'oof_auc_full': float(np.nanmean(full_auc)), 'diff': dpt, 'lo': lo, 'hi': hi,
                          'C_folds': '/'.join(f'{x:g}' for x in rr['C']),
                          'dropped': '; '.join(names[j] for j in range(P) if j not in set(rr['cols']))})
        summ[bk] = {'fixed_full_auc': base_fixed, 'ref_auc': float(np.nanmean(full_auc)), 'contrast_pairs': R['contrast_pairs'], 'R2_ref_max': R['R2_ref_max'],
                    'drop_one_ok': all(v['ok'] for v in R['drop_one'].values()), 'fixed_full_ok': R['fixed_full']['ok'],
                    'l1_nonconverged': R['l1']['nonconv'], 'runtime_s': R['runtime_s'], 'fixed_full_max_coef_diff_vs_ref': float(np.nanmax(np.abs(R['fixed_full']['W'] - Wf)))}
    pd.DataFrame(zrows).to_csv(OUT / 'ZERO_WEIGHTS.csv', index=False)
    pd.DataFrame(l1rows).to_csv(OUT / 'L1_PATH.csv', index=False)
    pd.DataFrame(prows).to_csv(OUT / 'PAIRS.csv', index=False)
    pd.DataFrame(rrows).to_csv(OUT / 'RESTRICTED.csv', index=False)
    (OUT / 'AGGREGATE_CHECKS.json').write_text(json.dumps(summ, indent=1, default=str), encoding='utf8')
    print('aggregate written')


# ------------------------------------------------------------------ figure
def figure_stage():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    Z = pd.read_csv(OUT / 'ZERO_WEIGHTS.csv')
    pal = {'F1': '#2a78d6', 'F2': '#eb6834', 'F3': '#1baf7a', 'F4': '#eda100', 'F5': '#e87ba4', 'F6': '#008300', 'F7': '#4a3aa7'}
    bg, ink, ink2, grid = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
    fig, axes = plt.subplots(2, 1, figsize=(17, 11.5), gridspec_kw={'height_ratios': [1, 1.25]}, facecolor=bg)
    for ax, bk in zip(axes, ['N16', 'N56']):
        z = Z[Z.bank == bk].sort_values('w_std_mean').reset_index(drop=True); x = np.arange(len(z))
        R = pickle.load(open(CACHE / f'{bk}.pkl', 'rb')); band = 0.05 * R['R2_ref_max']; band1 = 0.05 * z.w_std_mean.abs().max()
        ax.set_facecolor(bg)
        ax.axhspan(-band, band, color='#d9d8d2', alpha=.7, lw=0, zorder=0)
        LIM = 1.6; wv = z.w_std_mean.to_numpy(); wc = np.clip(wv, -LIM, LIM)
        lo_e = np.clip(wv - z.w_min_folds.to_numpy(), 0, None); hi_e = np.clip(z.w_max_folds.to_numpy() - wv, 0, None)
        inside = np.abs(wv) <= LIM
        ax.bar(x, wc, width=.72, color=[pal[f] for f in z.family], zorder=2)
        ax.errorbar(x[inside], wv[inside], yerr=[lo_e[inside], hi_e[inside]], fmt='none', ecolor=ink2, elinewidth=1, capsize=2, zorder=3)
        for xi in np.flatnonzero(~inside):
            lo_, hi_ = z.w_min_folds.iloc[xi], z.w_max_folds.iloc[xi]
            ax.text(xi, np.sign(wv[xi]) * LIM * 1.03, f'{wv[xi]:+.2f} (cut)', ha='center', rotation=90,
                    va='bottom' if wv[xi] > 0 else 'top', fontsize=8, color=ink)
        ax.axhline(0, color=ink2, lw=.8, zorder=1)
        ax.set_xticks(x); ax.set_xticklabels(z.channel, rotation=75, ha='right', fontsize=8 if len(z) > 20 else 9.5, color=ink)
        for lab, nzr in zip(ax.get_xticklabels(), z.w_std_mean.abs() < band):
            if nzr: lab.set_fontweight('bold')
        ax.set_xlim(-.7, len(z) - .3); ax.set_ylim(-LIM * 1.95, LIM * 1.95)
        ax.set_ylabel('standardized LR weight (mean of 5 folds)', color=ink, fontsize=10)
        ax.grid(axis='y', color=grid, lw=.6, zorder=0); ax.set_axisbelow(True)
        for s_ in ('top', 'right'): ax.spines[s_].set_visible(False)
        for s_ in ('left', 'bottom'): ax.spines[s_].set_color(ink2)
        ax.tick_params(colors=ink2)
        nz = int((z.w_std_mean.abs() < band).sum())
        ax.set_title(f'{bk} ({len(z)} channels): {nz} near zero (bold names, grey band |w| < {band:.3f} = 5% of the largest weight outside collinear contrast pairs).' + chr(10) +
                     f'Bars beyond ±{LIM:g} are cut and labelled with their mean (fold ranges in ZERO_WEIGHTS.csv). A plain 5%-of-max rule would give ±{band1:.2f}, set by the digit contrast pair.',
                     loc='left', fontsize=10.5, color=ink)
    handles = [Patch(color=pal[f], label=f'{FAMILY_NAMES[f]}') for f in pal] + [Patch(color='#d9d8d2', label='near-zero band')]
    leg = fig.legend(handles=handles, loc='upper center', ncol=8, frameon=False, fontsize=9.5, bbox_to_anchor=(.5, 1.0))
    for t in leg.get_texts(): t.set_color(ink)
    fig.suptitle('Supervised LR weights per channel on PRMBench (sorted). Opposite weights on near-duplicate channels are a collinearity effect, not two separate signals.',
                 y=.965, fontsize=11.5, color=ink, x=.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, .95))
    (OUT / 'figures').mkdir(exist_ok=True)
    for ext_ in ('png', 'svg'): fig.savefig(OUT / 'figures' / f'W1_weights_N16_N56.{ext_}', dpi=150, facecolor=bg)
    print('figure written')


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else 'all'
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        if stage == 'gate': gate_stage()
        elif stage == 'bank': bank_stage(sys.argv[2])
        elif stage == 'aggregate': aggregate_stage()
        elif stage == 'figure': figure_stage()
        elif stage == 'all':
            gate_stage(); [bank_stage(b) for b in BANKS]; aggregate_stage(); figure_stage()
