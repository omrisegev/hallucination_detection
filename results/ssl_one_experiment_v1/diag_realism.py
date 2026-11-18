"""Why do the synthetic errors fail the realism check? Dev pool only (Omri 2026-11-18, decision card option B). CPU.

Reads the same step tables as run_analysis.py. Uses only the DEV side. The test side is never loaded into any statistic.
Gold labels select the real wrong / real clean dev steps and answers (as in the realism check of run_analysis.py).

Step level (PRMBench; needs V2):
  sets   S = V2 step k (synthetic wrong)   P = the parent's own step k   RW = real wrong steps   RC = real clean steps
  per channel (top10 readout; raw value and R0 value):
         d_real = (mean RW - mean RC) / sd RC        d_syn = (mean S - mean RC) / sd RC      loudness = d_syn / d_real
         AUC(RW vs RC), AUC(S vs P), AUC(S vs RW)
  C2ST (as the stop rule: R0 16 channels, balanced by length quintile, 5-fold by group): all, each channel alone,
         each channel left out, token 11 / step 5, nuisance only (position, length, step count), raw 16
  out-of-fold synthetic probability of each S row and RW row by stratum (relax level, same generator, length ratio,
         position distance, parent category, k position; RW by error category)
  spill-over: AUC(V2 step k-1 / k+1 vs parent step k-1 / k+1), raw and R0 (raw k-1 must be .5: the prefix is equal)
Answer level (both benchmarks): stage-A features with M = 1 (16 oriented ECDF maxima + Simes):
         AUC(V vs real wrong answers), AUC(real wrong vs real clean), C2ST all / one feature / leave one out, V2 and V1.

    python diag_realism.py --W <dir with steps/ inputs/> --gold <processbench_gold.json> --out <OUT>
"""
import argparse, json, time
from pathlib import Path

import numpy as np

import ssl_data as D
import ssl_lib as L
import ssl_stages as S
import run_analysis as RA

N16 = list(L.N16)
LOG = []


def log(*a):
    s = ' '.join(str(x) for x in a); LOG.append(s); print(s, flush=True)


def c2st_oof(Xa, Xb, ga, gb, folds=5):
    """L.c2st with the out-of-fold probabilities returned as well (same model, same folds)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X = np.r_[Xa, Xb]; y = np.r_[np.ones(len(Xa)), np.zeros(len(Xb))]; g = np.r_[ga, gb]; p = np.zeros(len(y))
    for tr, te in GroupKFold(folds).split(X, y, g):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return L.auroc(y.astype(bool), p), p[:len(Xa)], p[len(Xa):]


def col(X):
    return X[:, None] if X.ndim == 1 else X


def effect(Ssyn, RW, RC, P=None):
    sd = RC.std(0); sd[sd == 0] = 1
    d_real = (RW.mean(0) - RC.mean(0)) / sd; d_syn = (Ssyn.mean(0) - RC.mean(0)) / sd
    out = dict(d_real=d_real.tolist(), d_syn=d_syn.tolist(),
               loudness=np.where(np.abs(d_real) > .05, d_syn / np.where(np.abs(d_real) > .05, d_real, 1), np.nan).tolist(),
               auc_real=[L.auroc(np.r_[np.ones(len(RW)), np.zeros(len(RC))].astype(bool), np.r_[RW[:, j], RC[:, j]]) for j in range(RW.shape[1])],
               auc_syn_vs_realwrong=[L.auroc(np.r_[np.ones(len(Ssyn)), np.zeros(len(RW))].astype(bool), np.r_[Ssyn[:, j], RW[:, j]]) for j in range(RW.shape[1])])
    if P is not None:
        out['auc_syn_vs_parent'] = [L.auroc(np.r_[np.ones(len(Ssyn)), np.zeros(len(P))].astype(bool), np.r_[Ssyn[:, j], P[:, j]]) for j in range(RW.shape[1])]
    return out


def strata_means(p, keys, name, order=None):
    out = {}
    for v in ([o for o in order if o in set(keys)] if order else sorted(set(keys), key=str)):
        m = np.array([k == v for k in keys])
        out[str(v)] = dict(n=int(m.sum()), mean_p=float(p[m].mean()))
    return {name: out}


def binned(x, edges, labels):
    return [labels[int(np.searchsorted(edges, v, side='right'))] for v in x]


def step_level(Cd, V2d, side2, R):
    rawC, rawV = D.readout(Cd), D.readout(V2d)
    XC, XV = L.r0_cols(rawC, Cd.act, Cd.off), L.r0_cols(rawV, V2d.act, V2d.off)
    pj = RA.parent_index(V2d, Cd); nV = np.diff(V2d.off)
    kV = V2d.off[:-1] + V2d.k; kP = Cd.off[pj] + V2d.k
    w, c = Cd.wrong_step, Cd.clean_step
    nC = np.repeat(np.diff(Cd.off), np.diff(Cd.off))
    relC = Cd.pos / np.maximum(nC - 1, 1); relV = V2d.k / np.maximum(nV - 1, 1)
    nuisC = np.column_stack([relC, np.log(np.maximum(Cd.ntok, 1)), np.log(nC)])
    nuisV = np.column_stack([relV, np.log(np.maximum(V2d.ntok[kV], 1)), np.log(nV)])
    # balanced sets exactly as the stop rule (RA.c2st_steps)
    gC = np.repeat(Cd.group, np.diff(Cd.off))[w]; ntC = Cd.ntok[w]; ntV = V2d.ntok[kV]; gV = V2d.group
    edges = np.quantile(np.r_[ntC, ntV], [.2, .4, .6, .8])
    ia, ib = L.balance_by_strata(np.searchsorted(edges, ntV), np.searchsorted(edges, ntC), np.random.default_rng(RA.SEED))
    Xk, Xw = XV[kV], XC[w]
    out = dict(n=dict(syn=int(len(kV)), real_wrong=int(w.sum()), real_clean=int(c.sum()), balanced_each=int(len(ia))))
    out['effect_raw'] = effect(rawV[kV], rawC[w], rawC[c], rawC[kP])
    out['effect_R0'] = effect(XV[kV], XC[w], XC[c], XC[kP])
    out['effect_nuisance'] = dict(names=['rel_position', 'log_ntok', 'log_n_steps'], **effect(nuisV, nuisC[w], nuisC[c], None))
    cs = {}
    auc_all, p_syn, p_rw = c2st_oof(Xk[ia], Xw[ib], gV[ia], gC[ib]); cs['all16 (stop rule)'] = auc_all
    for j, nm in enumerate(N16):
        cs[f'only {nm}'] = c2st_oof(Xk[ia][:, [j]], Xw[ib][:, [j]], gV[ia], gC[ib])[0]
        keep = [i for i in range(16) if i != j]
        cs[f'drop {nm}'] = c2st_oof(Xk[ia][:, keep], Xw[ib][:, keep], gV[ia], gC[ib])[0]
    cs['token 11 only'] = c2st_oof(Xk[ia][:, :11], Xw[ib][:, :11], gV[ia], gC[ib])[0]
    cs['step 5 only'] = c2st_oof(Xk[ia][:, 11:], Xw[ib][:, 11:], gV[ia], gC[ib])[0]
    cs['nuisance only (position, length, step count)'] = c2st_oof(nuisV[ia], nuisC[w][ib], gV[ia], gC[ib])[0]
    cs['all16 + position'] = c2st_oof(np.c_[Xk[ia], nuisV[ia, :1]], np.c_[Xw[ib], nuisC[w][ib, :1]], gV[ia], gC[ib])[0]
    cs['raw 16 (no within-answer z)'] = c2st_oof(rawV[kV][ia], rawC[w][ib], gV[ia], gC[ib])[0]
    out['c2st'] = cs
    log('step C2ST all16', round(auc_all, 4), '(stop rule .8209)')
    # strata of the out-of-fold synthetic probability
    sc = [side2.get(u, {}) for u in V2d.uids[ia]]
    st = {}
    st.update(strata_means(p_syn, [s.get('relax_level', 'na') for s in sc], 'syn by relax_level'))
    st.update(strata_means(p_syn, [s.get('donor_same_generator', 'na') for s in sc], 'syn by donor_same_generator'))
    lr = np.array([s.get('len_ratio', np.nan) for s in sc], float)
    lab = ['len_ratio<.9', '.9-1.1', '>1.1', 'na']; lr = np.where(np.isfinite(lr), lr, np.inf)
    st.update(strata_means(p_syn, binned(lr, [.9, 1.1, np.inf], lab), 'syn by len_ratio', lab))
    dp = np.abs(np.array([s.get('rel_parent', np.nan) for s in sc], float) - np.array([s.get('rel_donor', np.nan) for s in sc], float))
    lab = ['<.05', '.05-.1', '.1-.2', '>.2', 'na']; dp = np.where(np.isfinite(dp), dp, np.inf)
    st.update(strata_means(p_syn, binned(dp, [.05, .1, .2, np.inf], lab), 'syn by |rel_parent - rel_donor|', lab))
    lab = ['rel<.25', '.25-.5', '.5-.75', '>.75']
    st.update(strata_means(p_syn, binned(relV[ia], [.25, .5, .75], lab), 'syn by k position', lab))
    if getattr(Cd, 'cls', None) is not None:
        st.update(strata_means(p_syn, list(Cd.cls[pj][ia]), 'syn by parent category'))
        clsW = np.repeat(Cd.cls, np.diff(Cd.off))[w][ib]
        st.update(strata_means(p_rw, list(clsW), 'real wrong by error category'))
    st.update(strata_means(p_rw, binned(relC[w][ib], [.25, .5, .75], lab), 'real wrong by position', lab))
    out['oof_probability_strata'] = st
    # spill-over to the neighbour steps
    sp = {}
    for dk, nm in ((-1, 'k-1'), (1, 'k+1')):
        ok = (V2d.k + dk >= 0) & (V2d.k + dk < nV)
        a, b = kV[ok] + dk, kP[ok] + dk
        sp[nm] = dict(n=int(ok.sum()),
                      auc_raw=[L.auroc(np.r_[np.ones(ok.sum()), np.zeros(ok.sum())].astype(bool), np.r_[rawV[a, j], rawC[b, j]]) for j in range(16)],
                      auc_R0=[L.auroc(np.r_[np.ones(ok.sum()), np.zeros(ok.sum())].astype(bool), np.r_[XV[a, j], XC[b, j]]) for j in range(16)])
    out['spillover_vs_parent'] = sp
    R['step'] = out
    return out


def answer_level(Cd, Vs, R, key):
    rawC = D.readout(Cd); ev, sF, *_ = S.step_eval(Cd, rawC)
    AM = S.AnswerModel(Cd, rawC, sF); one = np.ones(16)
    FC = RA.feats(AM, Cd, rawC, one)[0]; nC = np.diff(Cd.off); edges = np.quantile(nC, [.2, .4, .6, .8])
    names = N16 + ['simes']; y = Cd.y_ans.astype(bool); out = dict(features=names)
    out['auc_real_wrong_vs_clean'] = [L.auroc(y, FC[:, j]) for j in range(17)]
    for vn, Vd in Vs.items():
        FV = RA.feats(AM, Vd, D.readout(Vd), one)[0]; nv = np.diff(Vd.off)
        ia, ib = L.balance_by_strata(np.searchsorted(edges, nv), np.searchsorted(edges, nC[y]), np.random.default_rng(RA.SEED))
        A, B, ga, gb = FV[ia], FC[y][ib], Vd.group[ia], Cd.group[y][ib]
        r = dict(n_each=int(len(ia)), effect=effect(FV, FC[y], FC[~y]))
        cs = {'all17 (stop rule)': c2st_oof(A, B, ga, gb)[0]}
        for j, nm in enumerate(names):
            cs[f'only {nm}'] = c2st_oof(A[:, [j]], B[:, [j]], ga, gb)[0]
            keep = [i for i in range(17) if i != j]
            cs[f'drop {nm}'] = c2st_oof(A[:, keep], B[:, keep], ga, gb)[0]
        cs['step count only'] = c2st_oof(np.log(nv[ia])[:, None], np.log(nC[y][ib])[:, None], ga, gb)[0]
        r['c2st'] = cs; out[vn] = r
        log(key, 'answer C2ST', vn, round(cs['all17 (stop rule)'], 4))
    R[key] = out


def figures(R, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    SURF, INK2, GRID, S1, S2, S3 = '#fcfcfb', '#52514e', '#e4e3df', '#2a78d6', '#eb6834', '#1baf7a'
    plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF, 'axes.labelcolor': INK2,
                         'xtick.color': INK2, 'ytick.color': INK2, 'axes.edgecolor': GRID, 'axes.spines.top': False,
                         'axes.spines.right': False, 'font.size': 9.5, 'axes.titlesize': 10.5, 'axes.titleweight': 'bold',
                         'legend.frameon': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .6})
    st = R.get('prmbench', {}).get('step')
    if st:
        e = st['effect_raw']; y = np.arange(16)[::-1]
        fig, ax = plt.subplots(1, 2, figsize=(11.5, 5.6), gridspec_kw=dict(width_ratios=[1.25, 1]))
        ax[0].barh(y + .2, e['d_real'], .38, color=S1, label='real wrong steps')
        ax[0].barh(y - .2, e['d_syn'], .38, color=S2, label='synthetic wrong step (V2 step k)')
        ax[0].axvline(0, color=INK2, lw=.8); ax[0].set_yticks(y); ax[0].set_yticklabels(N16)
        ax[0].set_xlabel('mean shift from real clean steps (in clean-step SDs), raw top10 readout')
        ax[0].set_title('How far each error type moves each channel', loc='left')
        ax[0].legend(loc='upper center', bbox_to_anchor=(.5, -.12), ncol=2)
        c = st['c2st']
        ax[1].barh(y + .2, [c[f'only {n}'] for n in N16], .38, color=S2, label='this channel alone')
        ax[1].barh(y - .2, [c[f'drop {n}'] for n in N16], .38, color=S1, label='all channels except this one')
        ax[1].axvline(c['all16 (stop rule)'], color=INK2, lw=1, ls='--')
        ax[1].axvline(.65, color=S3, lw=1.5); ax[1].set_xlim(.45, .9); ax[1].set_yticks(y); ax[1].set_yticklabels([])
        ax[1].set_xlabel('classifier AUROC, synthetic vs real wrong\n(green line = pass limit .65, dashed = all 16 channels)')
        ax[1].set_title('Which channels make the synthetic step detectable', loc='left')
        ax[1].legend(loc='upper center', bbox_to_anchor=(.5, -.12), ncol=2)
        fig.suptitle('D1 PRMBench dev: synthetic foreign step vs real wrong step', x=.01, ha='left', fontsize=11.5)
        fig.tight_layout(); fig.savefig(out / 'D1_step_channels.png', dpi=160); plt.close(fig)
        so = st['oof_probability_strata']; panels = [k for k in ('syn by relax_level', 'syn by k position', 'syn by parent category',
                                                                 'real wrong by error category', 'real wrong by position') if k in so]
        fig, ax = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 4.6), sharey=True)
        ax = np.atleast_1d(ax)
        for a_, k in zip(ax, panels):
            d = so[k]; ks = list(d)
            a_.bar(range(len(ks)), [d[x]['mean_p'] for x in ks], color=S2 if k.startswith('syn') else S1)
            a_.set_xticks(range(len(ks))); a_.set_xticklabels([f"{x}\n(n={d[x]['n']})" for x in ks], rotation=60, ha='right', fontsize=7.5)
            a_.axhline(.5, color=INK2, lw=.8, ls='--'); a_.set_title(k, loc='left', fontsize=9.5)
        ax[0].set_ylabel('mean out-of-fold P(synthetic)')
        fig.suptitle('D2 PRMBench dev: which synthetic steps look synthetic, and which real errors look synthetic', x=.01, ha='left', fontsize=11.5)
        fig.tight_layout(); fig.savefig(out / 'D2_strata.png', dpi=160); plt.close(fig)
    keys = [k for k in ('prmbench_answer', 'processbench_answer') if k in R]
    if keys:
        fig, ax = plt.subplots(1, len(keys), figsize=(6 * len(keys), 5.6), sharey=True); ax = np.atleast_1d(ax)
        for a_, k in zip(ax, keys):
            r = R[k]; names = r['features']; y = np.arange(len(names))[::-1]
            a_.barh(y + .27, r['auc_real_wrong_vs_clean'], .26, color=S1, label='real wrong vs real clean (signal)')
            for off, vn, col in ((0, 'V2', S2), (-.27, 'V1', '#6250d6')):
                if vn in r: a_.barh(y + off, r[vn]['effect']['auc_syn_vs_realwrong'], .26, color=col, label=f'{vn} vs real wrong (realism gap)')
            a_.axvline(.5, color=INK2, lw=.8); a_.set_xlim(.2, 1); a_.set_yticks(y); a_.set_yticklabels(names)
            a_.set_xlabel('AUROC of one answer feature'); a_.set_title(k.replace('_answer', '') + ', dev answers', loc='left')
            a_.legend(loc='upper center', bbox_to_anchor=(.5, -.1), ncol=1, fontsize=8)
        fig.suptitle('D3 answer features (M = 1): real signal vs realism gap', x=.01, ha='left', fontsize=11.5)
        fig.tight_layout(); fig.savefig(out / 'D3_answer_features.png', dpi=160); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--W', required=True); ap.add_argument('--gold', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    pools, _ = D.load_all(a.W, a.gold); R = dict(created=time.strftime('%Y-%m-%d %H:%M'), W=str(a.W), side='dev only')
    for bench in ('prmbench', 'processbench'):
        if (bench, 'C') not in pools: continue
        C = pools[bench, 'C']; Cd = C.subset(C.side == 'dev'); R[bench] = {}; Vs = {}
        for vn in ('V2', 'V1'):
            if (bench, vn) in pools:
                V = pools[bench, vn]; Vd, nd = RA.aligned(V.subset(V.side == 'dev'), Cd); Vs[vn] = Vd
                R[bench][f'{vn}_dev_answers'] = dict(n=Vd.n, dropped=nd)
        if 'V2' in Vs:
            step_level(Cd, Vs['V2'], D._sidecar(a.W, bench, 'V2'), R[bench])
        answer_level(Cd, Vs, R, f'{bench}_answer')
        json.dump(RA.rnd(R), open(out / 'DIAG.json', 'w'), indent=1, default=str)
    R['seconds'] = round(time.time() - t0, 1)
    json.dump(RA.rnd(R), open(out / 'DIAG.json', 'w'), indent=1, default=str)
    figures(R, out)
    (out / 'diag.log').write_text('\n'.join(LOG)); log('done', R['seconds'], 's')


if __name__ == '__main__':
    main()
