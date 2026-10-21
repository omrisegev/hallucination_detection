"""Plan section 2.2 (competitor document): each paper's metric as the paper defines it, for 5 PRMs.

    python collection_competitors.py [prism|crop|slider|ares|h2v|leaderboard|all]

Protocols are those of results/prmbench_paper_metric_parity_v1/{prism,crop,slider}_metrics.py, extended from one PRM
(Qwen-7B, locally refitted scores) to the 5 PRMs of the stage-1 export. The Qwen-7B rows of PRISM, CROP and SLIDER are
checked against that package's PRISM_RESULTS.json / CROP_RESULTS.json / SLIDER_RESULTS.json (COMPETITOR_PARITY.csv).

Arms: PRM native; PRM same rule (GAP, better scale); TELE_N16; CHAIN_N16; CHAIN_N13; LSML_CONT (label-selected).
Label use, per item (also in every output row's 'access' / 'threshold_access' column):
  * PRISM matched-TPR rows: the global threshold is chosen WITH labels to hit TPR 87.30 (a reading, not a rule).
  * CROP: every proxy's threshold is calibrated with labels on a 20% calibration split (CROP's own access); the two
    LR proxies are supervised. Our fused scores themselves are label-free (fitted once on all PRMBench answers).
  * SLIDER: threshold from a 1-D logistic regression on held-out labelled steps (their protocol, ambiguous).
  * ARES: threshold chosen by 5-fold CV on labels (their protocol); full PRMBench, so the row is APPROXIMATE.
Outputs in tables/ (COMP_*.csv) and figures/data/ (CSV behind every figure).
"""
import json, sys
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import evaluate_collection as E

T = E.TABLES; FD = E.HERE / 'figures/data'
PAR = E.ROOT / 'results/prmbench_paper_metric_parity_v1'
FUSED = ['TELE_N16', 'CHAIN_N16', 'CHAIN_N13', 'LSML_CONT']


def best_scale(c):
    a, b = ('PRM_Q80', 'PRM_Q80_PROB') if c.bench == 'prmbench' else ('PRM', 'PRMPROB')
    return max((a, b), key=lambda x: E.measure(c, x, 'GAP', fast=True)[0]['official'])


def arm_rules(c):
    """(label, arm, rule) rows reported for every competitor item."""
    nat = 'PRM_native' if c.bench == 'prmbench' else 'PRM'
    rows = [('PRM native', nat, 'native'), ('PRM same rule (GAP, better scale)', best_scale(c), 'GAP')]
    for a in FUSED:
        if a in c.arms:
            rows += [(f'{a} (GAP)', a, 'GAP'), (f'{a} (Q80)', a, 'Q80')]
    return rows


def thr_score(c, arm):
    """Score on which a GLOBAL threshold is swept: the PRM's own risk for the PRM, answer-z for fused arms."""
    return c.arms[arm] if arm in ('PRM_native', 'PRM') else E.answer_z(c.arms[arm], c.off)


# ============================================================================================== PRISM
PRISM_T2 = {'Baseline (Qwen-PRM-7B)': dict(FPR=69.34, FNR=4.27, Precision=89.40, TPR=95.36, TNR=30.66, PRMScore=65.50),
            'PRISM': dict(FPR=47.13, FNR=12.69, Precision=91.93, TPR=87.30, TNR=52.86, PRMScore=68.00)}
PRISM_T1 = {'Qwen-PRM-7B (paper)': (95.36, 30.66, 65.5), 'ThinkPRM': (83.3, 50.9, 64.3), 'GPT-4o': (82.9, 58.2, 66.8),
            'PRISM': (87.30, 52.86, 68.0)}


def prism(prms=E.PRMS):
    T1, T7, MT, CURVE, HIST, PAIR = [], [], [], [], [], []
    for prm in prms:
        c = E.load_prmbench(prm)
        all_rows = np.ones_like(c.keep)
        base_fpr = None
        for label, arm, rule in arm_rules(c):
            f = c.flags(arm, rule); ps = E.official(c, f, rule == 'native')[0]
            for pool, m in (('non-control (6,211 answers)', c.keep), ('with 758 control rows', all_rows)):
                cc = _counts(c, f, m); r = E.rates(cc)
                if pool.startswith('non') and rule == 'native': base_fpr = r['FPR']
                T1.append(dict(prm=prm, row=label, arm=arm, rule=rule, pool=pool, access=E.access_of(arm), **r,
                               PRMScore=ps if pool.startswith('non') else np.nan,
                               FPR_reduction_vs_native=100 * (base_fpr - r['FPR']) / base_fpr if base_fpr else np.nan,
                               flag_rate=float(f[m].mean())))
            T7.append(dict(prm=prm, row=label, arm=arm, rule=rule, access=E.access_of(arm), Overall=ps, **E.category_scores(c, f)))
        # matched TPR 87.30 (label-chosen reading) and the FPR-FNR curve (Fig 5b), on PRM, CHAIN_N16, LSML_CONT
        for arm in ('PRM_native', 'CHAIN_N16', 'LSML_CONT', 'CHAIN_N13', 'TELE_N16', 'PRM_Q80_PROB'):
            v = thr_score(c, arm) if arm != 'PRM_Q80_PROB' else c.arms[arm]
            ts = np.quantile(v[c.keep], np.linspace(0.6, 0.95, 701))
            best = min((abs(E.rates(_counts(c, v >= t, c.keep))['TPR'] - 87.30), t) for t in ts)[1]
            f = v >= best; r = E.rates(_counts(c, f, c.keep))
            MT.append(dict(prm=prm, arm=arm, threshold_access='global threshold chosen WITH labels to match TPR 87.30',
                           threshold=float(best), PRMScore=E.official(c, f)[0], **r))
            if arm in ('PRM_native', 'CHAIN_N16', 'LSML_CONT', 'CHAIN_N13'):
                for q in np.linspace(.005, .995, 199):
                    t = np.quantile(v[c.keep], q); r = E.rates(_counts(c, v >= t, c.keep))
                    CURVE.append(dict(prm=prm, arm=arm, kind='global threshold sweep', quantile=q, FPR=r['FPR'], FNR=r['FNR'], TPR=r['TPR']))
        for label, arm, rule in arm_rules(c):
            if rule in ('GAP', 'Q80', 'native'):
                r = E.rates(_counts(c, c.flags(arm, rule), c.keep))
                CURVE.append(dict(prm=prm, arm=arm, kind=f'operating point {rule}', quantile=np.nan, FPR=r['FPR'], FNR=r['FNR'], TPR=r['TPR']))
        # Fig 7: score histograms on error vs correct steps (non-control); PRM on reward, fused on answer-z
        for arm, v, kind in (('PRM reward', c.extra['reward'], 'reward'), ('CHAIN_N16', E.answer_z(c.arms['CHAIN_N16'], c.off), 'answer-z'),
                             ('LSML_CONT', E.answer_z(c.arms['LSML_CONT'], c.off), 'answer-z')):
            edges = np.linspace(0, 1, 41) if kind == 'reward' else np.linspace(-3, 4, 41)
            for lab, m in (('error step', c.err & c.keep), ('correct step', ~c.err & c.keep)):
                h, _ = np.histogram(np.clip(v[m], edges[0], edges[-1]), edges, density=True)
                for lo, hi, d in zip(edges[:-1], edges[1:], h):
                    HIST.append(dict(prm=prm, arm=arm, scale=kind, steps=lab, bin_lo=lo, bin_hi=hi, density=d, n=int(m.sum())))
        # Fig 4: within-answer pairwise accuracy (error step scored above valid step) by answer-z score margin
        for arm in ('PRM_Q80', 'CHAIN_N16', 'LSML_CONT', 'CHAIN_N13'):
            z = E.answer_z(c.arms[arm], c.off); D = []
            for a, b in zip(c.off[:-1], c.off[1:]):
                e = z[a:b][c.err[a:b]]; ok = z[a:b][~c.err[a:b]]
                if len(e) and len(ok): D.append((e[:, None] - ok[None, :]).ravel())
            D = np.concatenate(D); mg = np.abs(D); edges = np.quantile(mg, np.linspace(0, 1, 11))
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (mg >= lo) & (mg <= hi)
                PAIR.append(dict(prm=prm, arm=arm, margin_lo=lo, margin_hi=hi, n_pairs=int(m.sum()),
                                 pairwise_acc=float((D[m] > 0).mean() + .5 * (D[m] == 0).mean())))
            PAIR.append(dict(prm=prm, arm=arm, margin_lo='all', margin_hi='all', n_pairs=len(D),
                             pairwise_acc=float((D > 0).mean() + .5 * (D == 0).mean())))
    for name, rows in (('COMP_PRISM_T1_T2.csv', T1), ('COMP_PRISM_T7_CATEGORIES.csv', T7), ('COMP_PRISM_MATCHED_TPR.csv', MT)):
        E.write_csv(T / name, rows)
    E.write_csv(FD / 'prism_fig5b_fpr_fnr.csv', CURVE + [dict(prm='qwen_prm_7b', arm='PRISM (paper)', kind='published point',
                                                             FPR=47.13, FNR=12.69, TPR=87.30)])
    E.write_csv(FD / 'prism_fig7_histograms.csv', HIST); E.write_csv(FD / 'prism_fig4_pairwise_by_margin.csv', PAIR)
    E.write_csv(T / 'COMP_PRISM_PAPER_ROWS.csv', [dict(source='PRISM T2', row=k, access='trained (PRM800K pairs)' if k == 'PRISM' else 'trained by its authors', **v)
                                                  for k, v in PRISM_T2.items()] +
                [dict(source='PRISM T1', row=k, PosAcc=v[0], NegAcc=v[1], PRMScore=v[2]) for k, v in PRISM_T1.items()])
    return T1, T7, MT


def _counts(c, f, m):
    y_valid = ~c.err; p_valid = ~f
    return np.array([np.sum(y_valid & p_valid & m), np.sum(~y_valid & p_valid & m),
                     np.sum(~y_valid & ~p_valid & m), np.sum(y_valid & ~p_valid & m)])


# ============================================================================================== CROP
SEEDS = list(range(2806, 2816)); N_TRACES = 5000
ALPHAS = [.01, .02, .025, .05, .10, .15, .20]
CROP_T6 = {'Random': (.4994, 8.7, 4.7), 'Token/format': (.7751, 16.2, 5.0), 'Trace features': (.7812, 15.9, 4.6),
           'Direct PRM': (.8010, 4.8, 0.6), 'Trace+PRM': (.8336, 21.8, 5.0)}


def split_traces(has_err, seed):
    """Verbatim from prmbench_paper_metric_parity_v1/crop_metrics.py."""
    rng = np.random.default_rng(seed)
    tr, ca, te = [], [], []
    for key in (False, True):
        idx = np.flatnonzero(has_err == key).tolist(); rng.shuffle(idx)
        n = len(idx); n_tr = int(round(.6 * n)); n_ca = int(round(.2 * n))
        tr += idx[:n_tr]; ca += idx[n_tr:n_tr + n_ca]; te += idx[n_tr + n_ca:]
    for part in (tr, ca, te): rng.shuffle(part)
    return np.array(tr), np.array(ca), np.array(te)


def prefix_len(s, lam):
    hit = np.flatnonzero(s > lam)
    return int(hit[0]) if len(hit) else len(s)


def crop_lambda(scores, labels, cal, grid, alpha):
    losses = np.zeros((len(grid), len(cal)))
    for k, i in enumerate(cal):
        fe = np.flatnonzero(labels[i])
        if not len(fe): continue
        losses[:, k] = grid >= np.max(scores[i][:fe[0] + 1])
    risk = (1 + losses.sum(1)) / (len(cal) + 1)
    ok = np.flatnonzero(risk <= alpha)
    return (grid[ok[np.argmax(grid[ok])]] if len(ok) else grid.min()), bool(len(ok))


def crop_eval(scores, labels, test, lam):
    M = np.array([prefix_len(scores[i], lam) for i in test]); Tn = np.array([len(scores[i]) for i in test])
    cont = np.array([bool(m > 0 and labels[i][:m].any()) for m, i in zip(M, test)])
    return M, Tn, cont


def crop_subset(c):
    meta = E.prmbench_meta(); pos = {i: j for j, i in enumerate(c.ids)}
    # Hub order = prmbench_meta.json row order (== crop_metrics.py's sorted(int pkl keys); checked equal 2026-10-21)
    order = [r['idx'] for r in json.loads(E.META_JSON.read_text())['rows']]
    sub = [i for i in order if meta[i]['classification'] != 'correct'][:N_TRACES]
    J = np.array([pos[i] for i in sub]); sl = [slice(c.off[j], c.off[j + 1]) for j in J]
    labels = []
    for i, s in zip(sub, sl):                                # CROP's label rule
        e0 = {max(int(e) - 1, 0) for e in meta[i]['error_steps']}
        labels.append(np.array([t in e0 for t in range(s.stop - s.start)]))
    has_err = np.array([len(meta[i]['error_steps']) > 0 for i in sub])
    return sub, J, sl, labels, has_err


def crop(prms=E.PRMS):
    rows, alpha_rows, len_rows, pos_rows, cross = [], [], [], [], []
    for prm in prms:
        c = E.load_prmbench(prm)
        sub, J, sl, labels, has_err = crop_subset(c)
        n = np.diff(c.off); step_idx = np.concatenate([np.arange(1, k + 1) for k in n]).astype(float)
        tot = np.repeat(n, n).astype(float); tok = np.expm1(c.extra['log_step_tokens'])
        pref = np.concatenate([np.r_[0, np.cumsum(tok[a:b])[:-1]] for a, b in zip(c.off[:-1], c.off[1:])])
        atok = np.repeat(np.add.reduceat(tok, c.off[:-1]), n)
        F_pos = np.column_stack([step_idx, tot, step_idx / tot, tok, pref, atok])
        ch13 = E.channels(c, 'N13'); F_tele = np.column_stack([c.arms['PRM_native'], ch13])
        risk1 = 1 - c.extra['reward']; fixed = np.linspace(0, 1, 101)
        per = lambda v: [v[s] for s in sl]
        res = {}
        for seed in SEEDS:
            tr, ca, te = split_traces(has_err, seed)
            rows_tr = np.concatenate([np.arange(sl[k].start, sl[k].stop) for k in tr]); ytr = np.concatenate([labels[k] for k in tr])
            qgrid = lambda v: np.quantile(v[rows_tr], np.linspace(0, 1, 201))
            rr = np.random.default_rng(seed)
            cand = {'Random': ([rr.random(s.stop - s.start) for s in sl], fixed, 'fixed101', 'fixed control'),
                    'Direct PRM (1-reward)': (per(risk1), fixed, 'fixed101', E.ACCESS['PRM']),
                    'Direct PRM (1-reward), quantile grid': (per(risk1), qgrid(risk1), 'train-quantile201', E.ACCESS['PRM'])}
            allrows = np.concatenate([np.arange(s.start, s.stop) for s in sl])
            for nm, F in (('Position/length LR (supervised)', F_pos), ('Telemetry13+PRM LR (supervised)', F_tele)):
                mdl = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                                    LogisticRegression(C=1.0, solver='lbfgs', max_iter=500, class_weight='balanced', random_state=seed))
                mdl.fit(F[rows_tr], ytr); p = np.zeros(len(F)); p[allrows] = mdl.predict_proba(F[allrows])[:, 1]
                cand[nm] = (per(p), fixed, 'fixed101', E.ACCESS['LR'])
            for a in ('TELE_N16', 'CHAIN_N16', 'CHAIN_N13', 'LSML_CONT'):
                cand[a] = (per(c.arms[a]), qgrid(c.arms[a]), 'train-quantile201', E.access_of(a))
            yte = np.concatenate([labels[k] for k in te])
            for nm, (sc, grid, gname, acc) in cand.items():
                d = res.setdefault(nm, dict(grid=gname, access=acc, auroc=[], per_alpha={a: [] for a in ALPHAS}))
                d['auroc'].append(roc_auc_score(yte, np.concatenate([sc[k] for k in te])))
                for al in ALPHAS:
                    lam, feas = crop_lambda(sc, labels, ca, grid, al)
                    M, Tn, cont = crop_eval(sc, labels, te, lam)
                    d['per_alpha'][al].append(dict(kept=100 * float(np.mean(M / np.maximum(Tn, 1))), risk=100 * float(cont.mean()),
                                                   lam=float(lam), feasible=feas))
                    if al == .05:
                        for k_te, (m_, t_, ct) in zip(te, zip(M, Tn, cont)):
                            fe = np.flatnonzero(labels[k_te])
                            len_rows.append(dict(prm=prm, proxy=nm, seed=seed, T=int(t_), kept_frac=m_ / max(t_, 1), contaminated=bool(ct),
                                                 first_err_rel=(fe[0] + .5) / t_ if len(fe) else np.nan))
        for nm, d in res.items():
            au = np.array(d['auroc'])
            for al in ALPHAS:
                K = np.array([x['kept'] for x in d['per_alpha'][al]]); Rk = np.array([x['risk'] for x in d['per_alpha'][al]])
                r = dict(prm=prm, proxy=nm, access=d['access'], grid=d['grid'], alpha=al, auroc_mean=au.mean(), auroc_sd=au.std(ddof=1),
                         kept_mean=K.mean(), kept_sd=K.std(ddof=1), kept_ci95=1.96 * K.std(ddof=1) / np.sqrt(len(K)),
                         risk_mean=Rk.mean(), risk_sd=Rk.std(ddof=1), n_infeasible=sum(not x['feasible'] for x in d['per_alpha'][al]),
                         n_seeds=len(K), subset=f'first {N_TRACES} non-control rows (APPROXIMATE, see crop_metrics.py)',
                         note='CROP itself used Qwen-7B only; the other PRMs are our extension' if prm != 'qwen_prm_7b' else '')
                (rows if al == .05 else alpha_rows).append(r)
                if al == .05: alpha_rows.append(r)
        # cross-domain (Table 17 style): lambda calibrated on PRMBench (all 5,000 traces), applied to Socratic
        try:
            s_c = E.load_external('socratic', prm)
        except FileNotFoundError:
            continue
        s_lab = [s_c.err[a:b] & s_c.keep[a:b] for a, b in zip(s_c.off[:-1], s_c.off[1:])]
        for a in ('PRM_native', 'CHAIN_N16', 'CHAIN_N13', 'TELE_N16', 'LSML_CONT'):
            sa = {'PRM_native': 'PRM'}.get(a, a)
            if sa == 'LSML_CONT':
                v = E.load_frozen_lsml(s_c)
                if v is None: continue
            else:
                v = s_c.arms[sa]
            pb = c.arms[a]; grid = np.quantile(np.concatenate([pb[s] for s in sl]), np.linspace(0, 1, 201))
            for al in (.05, .10):
                lam, feas = crop_lambda(per(pb), labels, np.arange(len(sub)), grid, al)
                sc = [v[x:y] for x, y in zip(s_c.off[:-1], s_c.off[1:])]
                M, Tn, cont = crop_eval(sc, s_lab, np.arange(len(sc)), lam)
                cross.append(dict(prm=prm, proxy=a, alpha=al, calibrated_on='PRMBench (5,000 traces, labels)', tested_on='Socratic (non-empty steps)',
                                  lam=float(lam), feasible=feas, kept=100 * float(np.mean(M / np.maximum(Tn, 1))), risk=100 * float(cont.mean()),
                                  risk_minus_alpha=100 * float(cont.mean()) - 100 * al, n_test=len(sc),
                                  scale_note='PRM risk = logit(1-reward) on both; fused = their own scale on both' ))
        print('crop', prm, flush=True)
    E.write_csv(T / 'COMP_CROP_T6.csv', rows); E.write_csv(T / 'COMP_CROP_ALPHA.csv', alpha_rows)
    E.write_csv(T / 'COMP_CROP_CROSS_DOMAIN_SOCRATIC.csv', cross)
    E.write_csv(T / 'COMP_CROP_PAPER_ROWS.csv', [dict(source='CROP Table 6 (PRMBench, Qwen-PRM-7B, alpha .05)', proxy=k, auroc=v[0], kept=v[1], risk=v[2],
                                                      access='supervised LR' if k in ('Token/format', 'Trace features', 'Trace+PRM') else 'fixed') for k, v in CROP_T6.items()])
    # Fig 5 (risk/retention by trace length) and Fig 8 (contamination by first-error position)
    import collections
    L = collections.defaultdict(list)
    for r in len_rows: L[(r['prm'], r['proxy'])].append(r)
    f5, f8 = [], []
    for (prm, nm), rs in L.items():
        Tn = np.array([r['T'] for r in rs]); kf = np.array([r['kept_frac'] for r in rs]); ct = np.array([r['contaminated'] for r in rs])
        fr = np.array([r['first_err_rel'] for r in rs])
        for lo, hi in ((1, 8), (9, 12), (13, 16), (17, 10 ** 6)):
            m = (Tn >= lo) & (Tn <= hi)
            f5.append(dict(prm=prm, proxy=nm, steps_bin=f'{lo}-{hi if hi < 10 ** 6 else "max"}', n=int(m.sum()),
                           retention=100 * kf[m].mean(), risk=100 * ct[m].mean()))
        for nm2, (lo, hi) in (('early', (0, 1 / 3)), ('mid', (1 / 3, 2 / 3)), ('late', (2 / 3, 1.01))):
            m = (fr >= lo) & (fr < hi)
            f8.append(dict(prm=prm, proxy=nm, first_error=nm2, n=int(m.sum()), contamination=100 * ct[m].mean(), retention=100 * kf[m].mean()))
    E.write_csv(FD / 'crop_fig5_by_length.csv', f5); E.write_csv(FD / 'crop_fig8_by_first_error.csv', f8)
    E.write_csv(FD / 'crop_fig3_kept_vs_alpha.csv', alpha_rows)
    E.write_csv(FD / 'crop_fig4_auroc_vs_kept.csv', [r for r in rows])
    return rows


# ============================================================================================== SLIDER
def slider(prms=E.PRMS):
    rows = []
    for prm in prms:
        c = E.load_prmbench(prm)
        R = np.flatnonzero(c.cls == 'redundency'); sl = [slice(c.off[j], c.off[j + 1]) for j in R]
        for label, arm, rule in arm_rules(c):
            s = c.arms[arm]; flags = c.flags(arm, rule); res = dict(prm=prm, row=label, arm=arm, rule=rule, access=E.access_of(arm),
                                                                   n_problems=len(R))
            for scope in ('all_steps', 'steps_ge2'):
                keep = [np.arange(x.start, x.stop)[(slice(None) if scope == 'all_steps' else slice(1, None))] for x in sl]
                yy = np.concatenate([c.err[k] for k in keep]); ff = np.concatenate([flags[k] for k in keep])
                res[f'{scope}__label_free_{rule}__acc'] = 100 * float((ff == yy).mean())
                res[f'{scope}__label_free_{rule}__bal_acc'] = 50 * (ff[yy].mean() + (~ff[~yy]).mean())
                res[f'{scope}__majority_acc'] = 100 * float((~yy).mean())
                for cw in ('unweighted', 'balanced'):
                    acc, bal = [], []
                    for sd in range(10):
                        perm = np.random.default_rng(sd).permutation(len(keep)); h = len(keep) // 2
                        xh = np.concatenate([s[keep[k]] for k in perm[:h]])[:, None]; yh = np.concatenate([c.err[keep[k]] for k in perm[:h]])
                        xt = np.concatenate([s[keep[k]] for k in perm[h:]])[:, None]; yt = np.concatenate([c.err[keep[k]] for k in perm[h:]])
                        p = LogisticRegression(class_weight=None if cw == 'unweighted' else 'balanced').fit(xh, yh).predict(xt).astype(bool)
                        acc.append(100 * (p == yt).mean()); bal.append(50 * (p[yt].mean() + (~p[~yt]).mean()))
                    res[f'{scope}__LR_{cw}__acc'] = float(np.mean(acc)); res[f'{scope}__LR_{cw}__bal_acc'] = float(np.mean(bal))
            rows.append(res)
    rows.append(dict(prm='paper', row='Step-RRI (SLIDER, paper)', access='training-free score, LR threshold on held-out labels; extra GPT-4o-mini calls',
                     all_steps__paper_accuracy=61.91))
    rows.append(dict(prm='paper', row='InfoGain (paper, implied)', all_steps__paper_accuracy=48.38))
    rows.append(dict(prm='paper', row='Cosine (paper, implied)', all_steps__paper_accuracy=51.60))
    E.write_csv(T / 'COMP_SLIDER.csv', rows)
    return rows


# ============================================================================================== ARES
def ares(prms=E.PRMS):
    """Macro-F1 (= PRMScore form, pooled over non-control steps) with a global threshold chosen by 5-fold CV on labels:
    folds by source group; on each 4-fold train part choose the threshold (101 quantiles) maximizing macro-F1; apply to
    the held-out fold; report pooled macro-F1 of the out-of-fold decisions and the fold mean +- sd."""
    rows = []
    for prm in prms:
        c = E.load_prmbench(prm)
        ans_ok = c.cls != 'correct'; groups = c.groups
        ug = np.unique(groups[ans_ok]); rng = np.random.default_rng(20261021); fold_of = dict(zip(rng.permutation(ug), np.arange(len(ug)) % 5))
        step_fold = np.repeat([fold_of.get(g, -1) for g in groups], np.diff(c.off))
        for arm in ('PRM_native', 'TELE_N16', 'CHAIN_N16', 'CHAIN_N13', 'LSML_CONT'):   # one PRM row: a global threshold is scale-invariant
            v = thr_score(c, arm)
            oof = np.zeros(len(v), bool); fold_scores = []
            for k in range(5):
                trm = c.keep & (step_fold != k); tem = c.keep & (step_fold == k)
                ts = np.quantile(v[trm], np.linspace(0.01, 0.99, 99))
                best = max(ts, key=lambda t: E.official_from_counts(_counts(c, v >= t, trm), 'prmbench'))
                oof[tem] = v[tem] >= best
                fold_scores.append(float(E.official_from_counts(_counts(c, v >= best, tem), 'prmbench')))
            rows.append(dict(prm=prm, arm=arm, access=E.access_of(arm), threshold_access='5-fold CV on PRMBench labels (ARES protocol)',
                             macro_f1_oof=float(E.official_from_counts(_counts(c, oof, c.keep), 'prmbench')),
                             macro_f1_fold_mean=float(np.mean(fold_scores)), macro_f1_fold_sd=float(np.std(fold_scores, ddof=1)),
                             subset='full PRMBench non-control (ARES subset size not stated): APPROXIMATE'))
    rows += [dict(prm='paper', arm='ARES (GPT-4o-mini / Qwen3-4B entailment)', macro_f1_oof='.640 / .636', access='training-free, CV threshold, many LLM calls'),
             dict(prm='paper', arm='PRM alone, Qwen2.5-Math-PRM-7B backbone (ARES Table A7)', macro_f1_oof=.749),
             dict(prm='paper', arm='ARES with PRM-7B backbone (Table A7)', macro_f1_oof=.736, note='within one SD of the PRM: do not write "beats"')]
    E.write_csv(T / 'COMP_ARES.csv', rows)
    return rows


# ============================================================================================== Hard2Verify + leaderboards
def h2v(prms=E.PRMS):
    from collection_parity import h2v_common_uids
    common = h2v_common_uids(); rows = []
    for prm in prms:
        c = E.load_external('hard2verify', prm, uids_keep=common)
        v = E.load_frozen_lsml(c)
        if v is not None: c.arms['LSML_CONT'] = v
        for label, arm, rule in arm_rules(c):
            f = c.flags(arm, rule); bf1 = 100 * E.official(c, f, rule == 'native')[0]
            rows.append(dict(prm=prm, row=label, arm=arm, rule=rule, access=E.access_of(arm), step_balanced_f1=bf1,
                             **errorid(c, f, rule == 'native'), n_answers=len(c.ids)))
    rows += [dict(prm='paper', row='GPT-5 (Hard2Verify T2)', step_balanced_f1=85.83, access='prompted critic'),
             dict(prm='paper', row='Gemini-2.5-Pro (Hard2Verify T2)', step_balanced_f1=83.09, access='prompted critic'),
             dict(prm='paper', row='gpt-oss-120B (Hard2Verify T2)', step_balanced_f1=74.64, access='prompted critic'),
             dict(prm='paper', row='Qwen-PRM-7B (Hard2Verify T2)', step_balanced_f1=42.37, access='trained by its authors'),
             dict(prm='paper', row='Qwen-PRM-72B (Hard2Verify T2)', step_balanced_f1=35.50, access='trained by its authors'),
             dict(prm='paper', row='VerifySteer-Bi, Qwen3-8B-thinking / FARE-20B (ErrorID F1)', errorid_f1='33.4 / 51.1', access='labelled steering vectors + probe'),
             dict(prm='paper', row='Cons@8, Qwen3-8B-thinking / FARE-20B (ErrorID F1)', errorid_f1='31.2 / 42.7', access='self-consistency')]
    E.write_csv(T / 'COMP_HARD2VERIFY.csv', rows)
    return rows


def errorid(c, f, native):
    """Hard2Verify ErrorID reading (as in VerifySteer): predicted first error = first step predicted incorrect (empty steps
    count as incorrect; native rule uses the PRM's own decision there), else 'no error'. TPR = accuracy on fully correct
    responses, TNR = accuracy on incorrect responses, F1 = harmonic mean. Only included steps are considered."""
    G = E.gold('hard2verify'); ok_c, ok_e = [], []
    nst = c.off
    for j, u in enumerate(c.ids):
        g = G[u]; corr = np.asarray(g['correct'], bool); inc = np.asarray(g['include'], bool)
        pred_valid = np.zeros(len(corr), bool)
        ne_pos = _nonempty_positions(c, j, len(corr))
        pred_valid[ne_pos] = ~f[nst[j]:nst[j + 1]]
        if native:
            em = np.setdiff1d(np.arange(len(corr)), ne_pos); pred_valid[em] = c.empty['native_valid'][c.empty['answer'] == j]
        idx = np.flatnonzero(inc)
        true_fe = next((int(i) for i in idx if not corr[i]), -1); pred_fe = next((int(i) for i in idx if not pred_valid[i]), -1)
        (ok_c if true_fe == -1 else ok_e).append(pred_fe == true_fe)
    tpr, tnr = 100 * np.mean(ok_c), 100 * np.mean(ok_e)
    return dict(errorid_TPR_correct=tpr, errorid_TNR_incorrect=tnr, errorid_f1=2 * tpr * tnr / (tpr + tnr) if tpr + tnr else 0.,
                n_fully_correct=len(ok_c), n_incorrect=len(ok_e))


def _nonempty_positions(c, j, n):
    if not hasattr(c, '_ne_cache'):
        S = np.load(E.DATA / 'h2v' / f'SCORES_{E.EXT_NAME.get(c.prm, c.prm)}.npz')
        uids = S['uids'].astype(str); lens = S['nonempty_lengths']; ne = S['nonempty'].astype(bool); st = np.r_[0, np.cumsum(lens)]
        c._ne_cache = {u: np.flatnonzero(ne[st[k]:st[k + 1]]) for k, u in enumerate(uids)}
    p = c._ne_cache[c.ids[j]]; assert len(p) == c.off[j + 1] - c.off[j] and (p < n).all()
    return p


def leaderboard():
    rows = [dict(benchmark='PRMBench', row=r, value=v, access=a, source=s) for r, v, a, s in [
        ('Qwen2.5-Math-PRM-72B', 68.2, 'trained PRM', 'PRMBench T3'), ('PRISM (re-trained 7B)', 68.0, 'trained (PRM800K pairs)', 'PRISM T2'),
        ('PathFinder-PRM-7B', 67.7, 'trained PRM', 'its T1'), ('R-PRM-7B-DPO', 66.8, 'trained PRM (+ ~10 sampled critiques per step)', 'its T2'),
        ('ActPRM-X', 66.7, 'trained PRM', 'its T2'), ('Gemini-2.0-thinking-exp-1219', 68.8, 'prompted critic', 'PRMBench T3'),
        ('o1-mini (394-sample subset)', 68.8, 'prompted critic', 'PRMBench T3'), ('GPT-4o', 66.8, 'prompted critic', 'PRMBench T3'),
        ('Human (450 instances)', 83.8, 'human', 'PRMBench T3')]]
    rows += [dict(benchmark='Socratic-PRMBench', row=r, value=v, access=a, source='Socratic T3/T4') for r, v, a in [
        ('o3-mini', 75.7, 'prompted critic'), ('QwQ-32B', 73.8, 'prompted critic'), ('Gemini-2.5-Pro', 73.5, 'prompted critic'),
        ('DeepSeek-R1', 73.0, 'prompted critic'), ('GPT-4o', 70.8, 'prompted critic'), ('Qwen-PRM-7B (best PRM)', 68.0, 'trained PRM')]]
    main = E.read_csv(T / 'STATUS_MAIN.csv')
    bname = {'prmbench': 'PRMBench', 'socratic': 'Socratic-PRMBench', 'hard2verify': 'Hard2Verify (step Balanced F1, 177 common)'}
    for r in main:
        if r['row'] in ('PRM native',):
            rows.append(dict(benchmark=bname[r['benchmark']], row=f"ours: {E.PRM_LABEL[r['prm']]} native", value=float(r['native']),
                             access=r['access'], source='tables/STATUS_MAIN.csv'))
        if r['row'] in ('CHAIN_N16', 'CHAIN_N13', 'PRM same rule (better scale under GAP)', 'LSML_CONT'):
            rows.append(dict(benchmark=bname[r['benchmark']], row=f"ours: {E.PRM_LABEL[r['prm']]} {r['row']} (GAP)", value=float(r['GAP']),
                             access=r['access'], source='tables/STATUS_MAIN.csv'))
    rows += [dict(benchmark='Hard2Verify (step Balanced F1)', row=r, value=v, access=a, source='Hard2Verify T2') for r, v, a in [
        ('GPT-5', 85.83, 'prompted critic'), ('Gemini-2.5-Pro', 83.09, 'prompted critic'), ('Qwen-PRM-7B (paper)', 42.37, 'trained PRM')]]
    E.write_csv(T / 'COMP_LEADERBOARDS.csv', rows)


# ============================================================================================== parity vs the 1-PRM package
def parity():
    rows = []
    P = json.loads((PAR / 'PRISM_RESULTS.json').read_text())
    mine = {(r['arm'], r['rule'], r['pool']): r for r in E.read_csv(T / 'COMP_PRISM_T1_T2.csv') if r['prm'] == 'qwen_prm_7b'}
    for arm, rule in (('PRM_native', 'native'), ('CHAIN_N16', 'Q80'), ('CHAIN_N13', 'Q80'), ('LSML_CONT', 'Q80')):
        for k in ('FPR', 'FNR', 'TPR', 'TNR', 'Precision', 'PRMScore'):
            ref = P['rates'][arm][k]; ours = float(mine[(arm, rule, 'non-control (6,211 answers)')][k])
            rows.append(dict(item='PRISM T2 non-control', arm=arm, quantity=k, ours=ours, reference=ref, diff=ours - ref,
                             status='PASS' if abs(ours - ref) < 1e-6 else 'FAIL', source='prmbench_paper_metric_parity_v1/PRISM_RESULTS.json'))
    C = json.loads((PAR / 'CROP_RESULTS.json').read_text())['arms']
    mine = {r['proxy']: r for r in E.read_csv(T / 'COMP_CROP_T6.csv') if r['prm'] == 'qwen_prm_7b'}
    for ref_name, my_name in (('Direct PRM (1-reward)', 'Direct PRM (1-reward)'), ('Direct PRM (1-reward), quantile grid', 'Direct PRM (1-reward), quantile grid'),
                              ('Random', 'Random'), ('LSML_CONT', 'LSML_CONT'), ('CHAIN_N16', 'CHAIN_N16'), ('CHAIN_N13', 'CHAIN_N13'),
                              ('TELE13+PRM LR (supervised)', 'Telemetry13+PRM LR (supervised)'), ('Position/length LR (supervised)', 'Position/length LR (supervised)')):
        for k, mk in (('auroc', 'auroc_mean'), ('kept', 'kept_mean'), ('risk', 'risk_mean')):
            ref = C[ref_name][k]; ours = float(mine[my_name][mk]); tol = 1e-6
            gated = 'LR' not in ref_name                       # LR proxies use different (PRM-tokenizer) length features / same channels
            rows.append(dict(item='CROP T6 qwen_prm_7b', arm=my_name, quantity=k, ours=ours, reference=ref, diff=ours - ref,
                             status=('PASS' if abs(ours - ref) < tol else 'FAIL') if gated else 'INFO (features differ: see note)',
                             source='prmbench_paper_metric_parity_v1/CROP_RESULTS.json'))
    S = json.loads((PAR / 'SLIDER_RESULTS.json').read_text())['arms']
    mine = {(r['arm'], r['rule']): r for r in E.read_csv(T / 'COMP_SLIDER.csv') if r['prm'] == 'qwen_prm_7b'}
    for arm, rule in (('PRM_native', 'native'), ('CHAIN_N16', 'Q80'), ('LSML_CONT', 'Q80'), ('CHAIN_N13', 'Q80')):
        for k in ('all_steps__LR_balanced__acc', 'all_steps__LR_balanced__bal_acc', 'all_steps__majority_acc'):
            ref = S[arm][k]; ours = float(mine[(arm, rule)][k])
            rows.append(dict(item='SLIDER', arm=arm, quantity=k, ours=ours, reference=ref, diff=ours - ref,
                             status='PASS' if abs(ours - ref) < 1e-6 else 'FAIL', source='prmbench_paper_metric_parity_v1/SLIDER_RESULTS.json'))
        k_ref = 'all_steps__label_free_Q80__bal_acc'
        if arm != 'PRM_native':
            ref = S[arm][k_ref]; ours = float(mine[(arm, rule)][k_ref])
            rows.append(dict(item='SLIDER', arm=arm, quantity=k_ref, ours=ours, reference=ref, diff=ours - ref,
                             status='PASS' if abs(ours - ref) < 1e-6 else 'FAIL', source='prmbench_paper_metric_parity_v1/SLIDER_RESULTS.json'))
    E.write_csv(E.HERE / 'COMPETITOR_PARITY.csv', rows)
    from collections import Counter
    print('competitor parity', Counter(r['status'] for r in rows))
    for r in rows:
        if r['status'] == 'FAIL': print('  ', r)


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    T.mkdir(parents=True, exist_ok=True); FD.mkdir(parents=True, exist_ok=True)
    jobs = {'prism': prism, 'crop': crop, 'slider': slider, 'ares': ares, 'h2v': h2v, 'leaderboard': leaderboard, 'parity': parity}
    for k in (jobs if what == 'all' else [what]):
        print('==', k, flush=True); jobs[k]()
