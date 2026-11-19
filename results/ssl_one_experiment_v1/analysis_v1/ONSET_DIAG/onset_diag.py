"""Onset diagnostic (2026-11-19, dev only, CPU): do real FIRST error steps differ from LATER error steps, and which
synthetic step k (V2-V5, V3+V4) looks like a first error? Gold labels are used for this check only, never for a fit.
  A  PRMBench C dev: C2ST (L.c2st, group folds) first vs later errors; position-only AUROC; relative-position-matched
     C2ST; first vs clean steps before it; later vs clean steps after the first error; per-channel AUROC.
  B  PRMBench per-channel mean of the R0 (per-answer z) readout: clean-before-first, first, later, V3/V4/V5 step k
     and after k  (CHANNEL_MEANS.csv).
  C  the unchanged stop-rule test RA.c2st_steps (step-length quintile balance, seed RA.SEED, AUROC <= .65 and
     agreement >= .5) with the real-wrong set = all wrong / first errors only / later errors only.
     ProcessBench labels only the first error, so there all = first.
    python onset_diag.py <code dir (commit 4d57e57)> <W> <processbench_gold.json> <out dir>
"""
import csv, json, sys, time
from pathlib import Path
import numpy as np

CODE, W, GOLD, OUT = sys.argv[1:5]
W = Path(W); OUT = Path(OUT); OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, CODE)
import ssl_data as D, ssl_lib as L, run_analysis as RA, mix_v34 as MX

t0 = time.time()
LOG = open(OUT / 'onset_diag.log', 'w', encoding='utf-8')


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + '\n'); LOG.flush()


split = json.load(open(W / 'inputs' / 'SPLIT.json'))
labels = json.load(open(W / 'inputs' / 'prmbench_C_labels.json'))
gold = {g['uid']: g for g in json.load(open(GOLD))}
names = list(L.N16)
R = dict(created=time.strftime('%Y-%m-%d %H:%M'), side='dev only', code_commit='4d57e57')


def masks(P):
    first = np.zeros(P.off[-1], bool); later = np.zeros(P.off[-1], bool); before = np.zeros(P.off[-1], bool)
    after_clean = np.zeros(P.off[-1], bool); n_err = []
    for a, b in zip(P.off[:-1], P.off[1:]):
        e = np.where(P.wrong_step[a:b])[0]; n_err.append(len(e))
        if len(e):
            first[a + e[0]] = True; later[a + e[1:]] = True; before[a:a + e[0]] = True
            m = np.zeros(b - a, bool); m[e[0] + 1:] = True; m &= ~P.wrong_step[a:b]; after_clean[a:b] = m
    return first, later, before, after_clean, np.array(n_err)


def with_wrong(Cd, m):
    P = Cd.subset(np.ones(Cd.n, bool)); P.wrong_step = m.copy(); return P


pools = {}
for bench in ('prmbench', 'processbench'):
    C = D.load_pool(W, bench, 'C', split, gold, labels); Cd = C.subset(C.side == 'dev'); rawC = D.readout(Cd)
    first, later, before, after_clean, n_err = masks(Cd)
    pv = {}
    for V in ('V2', 'V3', 'V4', 'V5'):
        if not (W / 'steps' / f'STEPS_{bench}_{V}.npz').exists():
            log(bench, V, 'missing: skipped'); continue
        P = D.load_pool(W, bench, V, split)
        pv[V], nd = RA.aligned(P.subset(P.side == 'dev'), Cd); assert nd == 0, (bench, V, nd)
    pv['V3+V4'] = MX.concat([pv['V3'], pv['V4']], 'V3+V4')
    pools[bench] = (Cd, rawC, first, later, before, after_clean, n_err, pv)

# ---------------- A (PRMBench)
Cd, rawC, first, later, before, after_clean, n_err, pv = pools['prmbench']
X = L.r0_cols(rawC, Cd.act, Cd.off); g = np.repeat(Cd.group, np.diff(Cd.off))
rel = Cd.pos / np.repeat(np.maximum(np.diff(Cd.off) - 1, 1), np.diff(Cd.off))


def c2(ma, mb, seed=0):
    rng = np.random.default_rng(seed); ia, ib = np.where(ma)[0], np.where(mb)[0]; n = min(len(ia), len(ib))
    ia, ib = rng.choice(ia, n, replace=False), rng.choice(ib, n, replace=False)
    return dict(auroc=round(float(L.c2st(X[ia], X[ib], g[ia], g[ib])), 4), n_each=int(n))


def pos_matched(ma, mb, seed=0):
    edges = np.quantile(rel[ma | mb], np.linspace(.1, .9, 9))
    ia, ib = L.balance_by_strata(np.searchsorted(edges, rel[ma]), np.searchsorted(edges, rel[mb]), np.random.default_rng(seed))
    A_, B_ = np.where(ma)[0][ia], np.where(mb)[0][ib]
    return dict(auroc=round(float(L.c2st(X[A_], X[B_], g[A_], g[B_])), 4), n_each=int(len(A_)))


def chan_auc(ma, mb):
    y = np.r_[np.ones(ma.sum()), np.zeros(mb.sum())].astype(bool)
    return {n: round(float(L.auroc(y, np.r_[X[ma, j], X[mb, j]])), 3) for j, n in enumerate(names)}


w = n_err > 0
yfl = np.r_[np.ones(first.sum()), np.zeros(later.sum())].astype(bool)
A = dict(answers=int(Cd.n), wrong_answers=int(w.sum()),
         errors_per_wrong_answer_p10_50_90=np.percentile(n_err[w], [10, 50, 90]).tolist(),
         share_wrong_answers_with_more_than_1_error=round(float((n_err[w] > 1).mean()), 3),
         n_first=int(first.sum()), n_later=int(later.sum()),
         rel_pos_mean=dict(first=round(float(rel[first].mean()), 3), later=round(float(rel[later].mean()), 3)),
         first_vs_later=dict(channels=c2(first, later), channels_position_matched=pos_matched(first, later),
                             position_only_auc=round(float(L.auroc(yfl, np.r_[rel[first], rel[later]])), 4)),
         first_vs_clean_before=dict(channels=c2(first, before), channels_position_matched=pos_matched(first, before)),
         later_vs_clean_after_first=dict(channels=c2(later, after_clean), channels_position_matched=pos_matched(later, after_clean)),
         chan_auc_first_vs_later=chan_auc(first, later), chan_auc_first_vs_clean_before=chan_auc(first, before),
         chan_auc_later_vs_clean_after=chan_auc(later, after_clean))
R['A_prmbench_first_vs_later'] = A
log('A prmbench first vs later', A['first_vs_later'], '| first vs clean before', A['first_vs_clean_before'],
    '| later vs clean after', A['later_vs_clean_after_first'])

# ---------------- B (PRMBench channel means)
rows = {'clean before first': X[before].mean(0), 'first error': X[first].mean(0), 'later errors': X[later].mean(0)}
for V in ('V3', 'V4', 'V5'):
    P = pv[V]; Xv = L.r0_cols(D.readout(P), P.act, P.off); rows[V + ' step k'] = Xv[P.off[:-1] + P.k].mean(0)
    aft = np.zeros(P.off[-1], bool)
    for j, (a, b) in enumerate(zip(P.off[:-1], P.off[1:])):
        aft[a + P.k[j] + 1:b] = True
    rows[V + ' after k'] = Xv[aft].mean(0)
with open(OUT / 'CHANNEL_MEANS.csv', 'w', newline='') as f:
    cw = csv.writer(f); cw.writerow(['row'] + names)
    for k, v in rows.items():
        cw.writerow([k] + [round(float(x), 4) for x in v])
R['B_prmbench_channel_means_r0'] = {k: dict(zip(names, np.round(v, 4).tolist())) for k, v in rows.items()}
log('B written CHANNEL_MEANS.csv')

# ---------------- C (stop-rule test vs all / first / later)
C2 = {}
for bench, (Cd, rawC, first, later, before, after_clean, n_err, pv) in pools.items():
    refs = {'all_wrong': Cd.wrong_step, 'first_only': first}
    if bench == 'prmbench':
        refs['later_only'] = later
    for V, P in pv.items():
        rawV = D.readout(P)
        for rn, m in refs.items():
            r = RA.c2st_steps(with_wrong(Cd, m), P, rawC, rawV)
            r = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
            C2[f'{bench} {V} vs {rn}'] = r; log('C', bench, V, 'vs', rn, r)
R['C_c2st_steps'] = C2
EXPECT = {'prmbench V2 vs all_wrong': .8209, 'prmbench V3 vs all_wrong': .6987, 'prmbench V4 vs all_wrong': .6539,
          'prmbench V5 vs all_wrong': .7303, 'prmbench V3+V4 vs all_wrong': .6494, 'processbench V3 vs all_wrong': .6761,
          'processbench V4 vs all_wrong': .7865, 'processbench V5 vs all_wrong': .7211, 'processbench V3+V4 vs all_wrong': .6691,
          'prmbench V3 vs first_only': .7425, 'prmbench V4 vs first_only': .6418, 'prmbench V5 vs first_only': .7453}
R['code_check'] = {k: dict(now=C2[k]['auroc'], expected=e, pass_=bool(abs(C2[k]['auroc'] - e) < .0015)) for k, e in EXPECT.items() if k in C2}
log('code check', R['code_check'])
R['seconds'] = round(time.time() - t0, 1)
(OUT / 'ONSET_DIAG.json').write_text(json.dumps(R, indent=1))
log('done', R['seconds'], 's')
