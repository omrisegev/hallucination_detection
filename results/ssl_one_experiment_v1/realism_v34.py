"""Realism check of the new synthetic errors V3 (numeric edit) and V4 (on-topic donor), DEV only (Omri 2026-11-18,
decision card "Numeric edit + on-topic donor": one dev-only GPU pass, then only the realism check). CPU, minutes.

The check is the pre-registered stop rule of run_analysis.py, unchanged (same functions, same seed):
  step level    C2ST of the variant's step k vs the real wrong dev steps: R0 16 channels, top10 readout, balanced by
                step-length quintile, logistic regression, 5 folds by question group; plus the 16x16 correlation-pattern
                agreement. Pass: AUROC <= .65 and agreement >= .5.            (RA.c2st_steps)
  answer level  C2ST of the variant's answers vs the real wrong dev answers on the 17 stage-A features with M = 1,
                balanced by step-count quintile. Same pass rule.              (RA.c2st_answers)
V1 and V2 are recomputed in the same run as a code check (PRMBench must give step .8209, answer V2 .7232, V1 .806).
Explanation (no decision): "loudness" = how far the variant step moves each channel from the real clean steps,
divided by how far a real wrong step moves it (raw top10 readout, clean-step SDs; diag_realism.effect).
Caveat checks (gold labels as a check only, dev, never a fit):
  V3: share of the edited steps k that are already a labeled error step of the parent (PRMBench) or at / after the
      parent's first error (ProcessBench).
  V4: share of donor steps that are a labeled error step of the donor (PRMBench) or at / after the donor's first error
      (ProcessBench); and the mean out-of-fold P(synthetic) of the step-level C2ST in each of these strata.

    python realism_v34.py --W <dir with steps/ inputs/> --gold <processbench_gold.json> --out <OUT>
"""
import argparse, json, time
from pathlib import Path

import numpy as np

import diag_realism as DR
import run_analysis as RA
import ssl_data as D
import ssl_lib as L
import ssl_stages as S

N16 = list(L.N16)
VARS = ('V1', 'V2', 'V3', 'V4')
# code check: the same functions on the same pools must give the numbers of the analysis run / DIAG_REALISM.md
EXPECT = {('prmbench', 'step', 'V2'): .8209, ('prmbench', 'answer', 'V2'): .7232, ('prmbench', 'answer', 'V1'): .806,
          ('processbench', 'answer', 'V1'): .898}
LOG = []


def log(*a):
    s = ' '.join(str(x) for x in a); LOG.append(s); print(s, flush=True)


def median_loudness(e):
    x = np.array(e['loudness'], float); x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else None


def label_checks(bench, V, Vd, Cd, side, labels, gold):
    """Gold labels as a check only: is the edited step / the donor step already an error step?"""
    out = {}
    sc = [side[u] for u in Vd.uids]
    if bench == 'prmbench':
        lab = {r['idx']: r for r in labels}
        out['k_is_parent_error_step'] = float(np.mean([s['k'] in lab[s['parent_uid']]['error_steps'] for s in sc]))
        if V == 'V4':
            dz = [s['donor_step'] in lab[s['donor_uid']]['error_steps'] for s in sc]
            out['donor_step_is_labeled_error'] = float(np.mean(dz)); return out, np.array(dz)
    else:
        def after_err(uid, step1):           # PB labels only the first error (0-based; -1 = clean)
            g = gold[uid]['label']; return bool(g >= 0 and step1 - 1 >= g)
        out['k_at_or_after_parent_first_error'] = float(np.mean([after_err(s['parent_uid'], s['k']) for s in sc]))
        if V == 'V4':
            dz = [after_err(s['donor_uid'], s['donor_step']) for s in sc]
            out['donor_step_at_or_after_donor_first_error'] = float(np.mean(dz)); return out, np.array(dz)
    return out, None


def step_level(Cd, Vd, dz):
    """RA.c2st_steps (the stop rule) + loudness + out-of-fold P(synthetic) by the V4 donor-label stratum."""
    rawC, rawV = D.readout(Cd), D.readout(Vd)
    r = RA.c2st_steps(Cd, Vd, rawC, rawV)
    kV = Vd.off[:-1] + Vd.k; pj = RA.parent_index(Vd, Cd); kP = Cd.off[pj] + Vd.k
    e = DR.effect(rawV[kV], rawC[Cd.wrong_step], rawC[Cd.clean_step], rawC[kP])
    r.update(loudness_median=median_loudness(e), d_real=e['d_real'], d_syn=e['d_syn'], loudness=e['loudness'],
             auc_syn_vs_parent_step=e['auc_syn_vs_parent'])
    if dz is not None:                           # same balanced sets as RA.c2st_steps
        XC, XV = L.r0_cols(rawC, Cd.act, Cd.off), L.r0_cols(rawV, Vd.act, Vd.off); w = Cd.wrong_step
        gC = np.repeat(Cd.group, np.diff(Cd.off))[w]; ntC = Cd.ntok[w]; ntV = Vd.ntok[kV]
        edges = np.quantile(np.r_[ntC, ntV], [.2, .4, .6, .8])
        ia, ib = L.balance_by_strata(np.searchsorted(edges, ntV), np.searchsorted(edges, ntC), np.random.default_rng(RA.SEED))
        auc, p_syn, _ = DR.c2st_oof(XV[kV][ia], XC[w][ib], Vd.group[ia], gC[ib])
        assert abs(auc - r['auroc']) < 1e-9, (auc, r['auroc'])
        r['oof_p_synthetic_by_donor_label'] = {str(v): dict(n=int((dz[ia] == v).sum()), mean_p=float(p_syn[dz[ia] == v].mean()))
                                               for v in (True, False) if (dz[ia] == v).any()}
    return r


def figure(R, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    SURF, INK, INK2, GRID = '#fcfcfb', '#1f1e1c', '#52514e', '#e4e3df'
    COL = {'real wrong': '#2a78d6', 'V1': '#6250d6', 'V2': '#eb6834', 'V3': '#1baf7a', 'V4': '#d6a12a'}
    plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF, 'axes.labelcolor': INK2,
                         'xtick.color': INK2, 'ytick.color': INK2, 'axes.edgecolor': GRID, 'axes.spines.top': False,
                         'axes.spines.right': False, 'font.size': 9.5, 'axes.titlesize': 10.5, 'axes.titleweight': 'bold',
                         'legend.frameon': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .6})
    benches = [b for b in ('prmbench', 'processbench') if b in R]
    fig, ax = plt.subplots(len(benches), 2, figsize=(12.5, 5.4 * len(benches)), gridspec_kw=dict(width_ratios=[1, 1.35]), squeeze=False)
    for row, b in enumerate(benches):
        r = R[b]; a0, a1 = ax[row]
        vs = [v for v in VARS if v in r.get('answer', {})]
        x = np.arange(len(vs))
        st = [r['step'][v]['auroc'] if v in r.get('step', {}) else np.nan for v in vs]
        an = [r['answer'][v]['auroc'] for v in vs]
        a0.bar(x - .19, st, .36, color=[COL[v] for v in vs], edgecolor=INK2, linewidth=.4, hatch='', label='step level (step k)')
        a0.bar(x + .19, an, .36, color=[COL[v] for v in vs], alpha=.45, edgecolor=INK2, linewidth=.4, label='answer level')
        for xi, s_, a_ in zip(x, st, an):
            if np.isfinite(s_): a0.text(xi - .19, s_ + .01, f'{s_:.2f}', ha='center', fontsize=8, color=INK)
            a0.text(xi + .19, a_ + .01, f'{a_:.2f}', ha='center', fontsize=8, color=INK)
        a0.axhline(.65, color=INK, lw=1.2, ls='--'); a0.text(-.45, .655, 'pass limit .65', ha='left', va='bottom', fontsize=8, color=INK)
        a0.axhline(.5, color=INK2, lw=.6)
        a0.set_xticks(x); a0.set_xticklabels(vs); a0.set_ylim(.4, 1.06); a0.set_ylabel('classifier AUROC, variant vs real wrong')
        a0.set_title(f'{b}: realism check (dark = step level, light = answer level)', loc='left')
        ch = r.get('step', {}); y = np.arange(16)[::-1]
        if ch:
            first = next(iter(ch.values()))
            a1.scatter(first['d_real'], y, s=34, color=COL['real wrong'], label='real wrong steps', zorder=3, marker='s')
            for v in ('V2', 'V3', 'V4'):
                if v in ch: a1.scatter(ch[v]['d_syn'], y, s=26, color=COL[v], label=f'{v} step k', zorder=3)
            a1.axvline(0, color=INK2, lw=.8); a1.set_yticks(y); a1.set_yticklabels(N16)
            a1.set_xlabel('mean shift from real clean steps (clean-step SDs), raw top10 readout')
            a1.set_title(f'{b}: how far each error type moves each channel', loc='left')
            a1.legend(loc='lower right', fontsize=8)
    fig.suptitle('R1 dev only: do the new synthetic errors (V3 numeric edit, V4 on-topic donor) look like real errors?',
                 x=.01, ha='left', fontsize=11.5)
    fig.tight_layout(); fig.savefig(out / 'R1_realism_v34.png', dpi=160); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--W', required=True); ap.add_argument('--gold', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    W = Path(a.W); split = json.load(open(W / 'inputs' / 'SPLIT.json'))
    gold = {g['uid']: g for g in json.load(open(a.gold))}
    lp = W / 'inputs' / 'prmbench_C_labels.json'; labels = json.load(open(lp)) if lp.exists() else None
    R = dict(created=time.strftime('%Y-%m-%d %H:%M'), W=str(W), side='dev only', rule='AUROC <= .65 and agreement >= .5')
    for bench in ('prmbench', 'processbench'):
        if not (W / 'steps' / f'STEPS_{bench}_C.npz').exists(): continue
        C = D.load_pool(W, bench, 'C', split, gold, labels); Cd = C.subset(C.side == 'dev')
        rawC = D.readout(Cd); ev, sF, *_ = S.step_eval(Cd, rawC)
        AM = S.AnswerModel(Cd, rawC, sF); one = np.ones(16)
        FC = RA.feats(AM, Cd, rawC, one)[0]; nC = np.diff(Cd.off); edges = np.quantile(nC, [.2, .4, .6, .8])
        r = R[bench] = dict(dev=dict(answers=Cd.n, real_wrong_steps=int(Cd.wrong_step.sum()), real_wrong_answers=int(Cd.y_ans.sum())),
                            step={}, answer={}, label_checks={}, counts={})
        for V in VARS:
            if not (W / 'steps' / f'STEPS_{bench}_{V}.npz').exists(): continue
            P = D.load_pool(W, bench, V, split)
            Vd, nd = RA.aligned(P.subset(P.side == 'dev'), Cd)
            r['counts'][V] = dict(dev_answers=Vd.n, dropped_parent_mismatch=nd)
            dz = None
            if V in ('V3', 'V4'):
                side = D._sidecar(W, bench, V)
                r['label_checks'][V], dz = label_checks(bench, V, Vd, Cd, side, labels, gold)
            if V != 'V1':
                r['step'][V] = step_level(Cd, Vd, dz)
                log(bench, V, 'step C2ST', RA.rnd({k: r['step'][V][k] for k in ('auroc', 'agreement', 'n_each', 'pass_', 'loudness_median')}))
            FV = RA.feats(AM, Vd, D.readout(Vd), one)[0]
            r['answer'][V] = RA.c2st_answers(FC, Cd.y_ans, Cd.group, nC, FV, Vd.group, np.diff(Vd.off), edges)
            log(bench, V, 'answer C2ST', RA.rnd(r['answer'][V]))
        r['code_check'] = {f'{lv} {V}': dict(now=r[lv][V]['auroc'], expected=e, pass_=bool(abs(r[lv][V]['auroc'] - e) < .0015))
                           for (b, lv, V), e in EXPECT.items() if b == bench and V in r[lv]}
        log(bench, 'code check', RA.rnd(r['code_check']))
        r['verdict'] = {V: dict(step=r['step'].get(V, {}).get('pass_'), answer=r['answer'][V]['pass_']) for V in r['answer']}
        json.dump(RA.rnd(R), open(out / 'REALISM_V34.json', 'w'), indent=1, default=str)
    R['seconds'] = round(time.time() - t0, 1)
    json.dump(RA.rnd(R), open(out / 'REALISM_V34.json', 'w'), indent=1, default=str)
    figure(R, out)
    (out / 'realism_v34.log').write_text('\n'.join(LOG)); log('done', R['seconds'], 's')


if __name__ == '__main__':
    main()
