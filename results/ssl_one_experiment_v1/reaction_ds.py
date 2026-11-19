"""Dev-only check of Omri's idea (thread, 2026-11-19 10:45 and 10:54 Israel): inject several error types into the SAME
answer, measure the reaction of each type, and fuse the reactions label-free to get the gate threshold pi.

Reaction of answer i to error type j:  r_ij = sA(T_j(x_i)) - sA(x_i)
  sA = the answer score of stage G (17 answer features, M = 1, answer L-SML fitted label-free on the dev C pool), the
  same model as pihat_dev.py / mix_v34.py / parent_split.py. T_j(x_i) = the dev variant of type j whose parent is x_i.
Idea: a clean answer jumps when an error is injected; an answer that is already wrong reacts less (its extreme step is
already there). So each type is a "classifier" of "is this answer already wrong" (small reaction = wrong). If the errors
of these classifiers are conditionally independent given the label, Dawid-Skene identifies pi without labels and without
the realism (SCAR) assumption of BBE.

Reports per benchmark (types = the variant pools that exist and passed the GPU smoke check):
  1. per type: n, AUROC of -r (wrong vs clean parent); the same inside 5 strata of the parent sA (does the reaction add
     information beyond the answer score?); Spearman(r, parent sA) inside each class (headroom);
  2. conditional Spearman between the reactions of each pair of types, inside wrong and inside clean answers (key check);
  3. pi-hat, label-free, on the answers with all types present ("complete") and on all answers (missing allowed):
     - binary Dawid-Skene EM, vote "wrong" if r_j is below its median (also cuts at the 35% and 65% quantiles);
     - Gaussian latent-class EM (2 classes, conditionally independent normal views on rank-normal reactions);
  4. gate with each pi-hat: open the pi-hat share of answers with the highest sA (the stage-G rule); wrong opened, clean
     closed; PB F1 / SLA, PRMBench PRMScore (GAP flags of the top10 readout, as in GATE_DEV). References: no gate, true pi.
     Code check: the true-pi gate must equal GATE_DEV.json (PB F1 41.653, PRMBench PRMScore 67.7636).
Gold labels: AUROC values, conditional correlations, true pi, metrics. Never inside a fit.

    python reaction_ds.py --W <dir with steps/ inputs/> --gold <processbench_gold.json> --out <OUT>
"""
import argparse, json, time
from pathlib import Path

import numpy as np
from scipy.stats import norm, rankdata, spearmanr

import run_analysis as RA
import ssl_data as D
import ssl_lib as L
import ssl_stages as S

TYPES = ('V1', 'V2', 'V3', 'V4', 'V5')
EXPECT_TRUE_PI = {'processbench': ('F1', 41.653), 'prmbench': ('PRMScore', 67.7636)}
LOG = []


def log(*a):
    s = ' '.join(str(x) for x in a); LOG.append(s); print(s, flush=True)


def rank_normal(x):
    """Rank-normal transform of the finite values; NaN stays NaN."""
    out = np.full(len(x), np.nan); m = np.isfinite(x)
    out[m] = norm.ppf((rankdata(x[m]) - .5) / m.sum()); return out


def ds_binary(Vt, iters=1000, tol=1e-10):
    """Two-class Dawid-Skene EM. Vt: n x m votes in {0, 1} (1 = 'wrong'), NaN = missing.
    Returns pi (share of class 1), per-view sensitivity / specificity, posterior P(class 1). Init: mean vote."""
    A = np.isfinite(Vt); V = np.where(A, Vt, 0.)
    q = np.where(A.any(1), V.sum(1) / np.maximum(A.sum(1), 1), .5); q = np.clip(q, .05, .95)
    pi = q.mean(); ll_old = -np.inf
    for _ in range(iters):
        pi = float(np.clip(q.mean(), 1e-4, 1 - 1e-4))
        sens = (q[:, None] * V * A).sum(0) / np.maximum((q[:, None] * A).sum(0), 1e-12)
        spec = ((1 - q)[:, None] * (1 - V) * A).sum(0) / np.maximum(((1 - q)[:, None] * A).sum(0), 1e-12)
        sens = np.clip(sens, 1e-4, 1 - 1e-4); spec = np.clip(spec, 1e-4, 1 - 1e-4)
        l1 = np.log(pi) + (A * (V * np.log(sens) + (1 - V) * np.log(1 - sens))).sum(1)
        l0 = np.log(1 - pi) + (A * (V * np.log(1 - spec) + (1 - V) * np.log(spec))).sum(1)
        mx = np.maximum(l1, l0); ll = float(np.sum(mx + np.log(np.exp(l1 - mx) + np.exp(l0 - mx))))
        q = 1 / (1 + np.exp(l0 - l1))
        if abs(ll - ll_old) < tol: break
        ll_old = ll
    # class 1 must be the 'wrong' class: the votes are oriented so that 1 = small reaction = wrong
    if np.mean(sens + spec) < 1:
        pi, sens, spec, q = 1 - pi, 1 - spec, 1 - sens, 1 - q
    return dict(pi_hat=float(pi), sens=sens.tolist(), spec=spec.tolist(), loglik=ll), q


def gauss_lc(Z, q0, iters=1000, tol=1e-10):
    """Two-class latent-class EM with conditionally independent normal views (NaN = missing). Z oriented: higher = wrong.
    Init: posterior q0 (binary DS)."""
    A = np.isfinite(Z); Zf = np.where(A, Z, 0.); q = np.clip(q0, .02, .98); ll_old = -np.inf
    for _ in range(iters):
        pi = float(np.clip(q.mean(), 1e-4, 1 - 1e-4))
        w1, w0 = q[:, None] * A, (1 - q)[:, None] * A
        m1 = (w1 * Zf).sum(0) / np.maximum(w1.sum(0), 1e-12); m0 = (w0 * Zf).sum(0) / np.maximum(w0.sum(0), 1e-12)
        s1 = np.sqrt(np.maximum((w1 * (Zf - m1) ** 2).sum(0) / np.maximum(w1.sum(0), 1e-12), 1e-3))
        s0 = np.sqrt(np.maximum((w0 * (Zf - m0) ** 2).sum(0) / np.maximum(w0.sum(0), 1e-12), 1e-3))
        l1 = np.log(pi) + (A * norm.logpdf(Zf, m1, s1)).sum(1); l0 = np.log(1 - pi) + (A * norm.logpdf(Zf, m0, s0)).sum(1)
        mx = np.maximum(l1, l0); ll = float(np.sum(mx + np.log(np.exp(l1 - mx) + np.exp(l0 - mx))))
        q = 1 / (1 + np.exp(l0 - l1))
        if abs(ll - ll_old) < tol: break
        ll_old = ll
    if np.mean(m1 - m0) < 0:
        pi, q, m1, m0, s1, s0 = 1 - pi, 1 - q, m0, m1, s0, s1
    return dict(pi_hat=float(pi), mean_wrong=m1.tolist(), mean_clean=m0.tolist(), sd_wrong=s1.tolist(), sd_clean=s0.tolist(),
                loglik=ll), q


def gate_metrics(Cd, sC, f0, share, key):
    W = Cd.y_ans.astype(bool); op = RA.gate_open(sC, share)
    m = RA.strip(RA.score_flags(Cd, f0 & L.expand(op, Cd.off)))
    return dict(share=float(share), open_share=float(op.mean()), wrong_opened=float(op[W].mean()),
                clean_closed=float((~op[~W]).mean()), **{k: m[k] for k in (('F1', 'SLA') if key == 'F1' else ('PRMScore',))})


def strata_auroc(y, score, base, k=5):
    """Mean AUROC inside k quantile strata of `base` (weighted by stratum size; strata with one class skipped)."""
    e = np.quantile(base, np.linspace(0, 1, k + 1)[1:-1]); st = np.searchsorted(e, base); a, w = [], []
    for s in range(k):
        m = st == s
        if y[m].any() and (~y[m]).any(): a.append(L.auroc(y[m], score[m])); w.append(m.sum())
    return float(np.average(a, weights=w)) if a else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--W', required=True); ap.add_argument('--gold', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--bench', default='processbench,prmbench')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    W_ = Path(a.W); split = json.load(open(W_ / 'inputs' / 'SPLIT.json'))
    gold = {g['uid']: g for g in json.load(open(a.gold))}
    lp = W_ / 'inputs' / 'prmbench_C_labels.json'; labels = json.load(open(lp)) if lp.exists() else None
    R = dict(created=time.strftime('%Y-%m-%d %H:%M'), W=str(W_), side='dev only', score='answer L-SML score sA, M = 1',
             reaction='sA(variant) - sA(parent); small reaction = wrong', labels_use='reports only; never inside a fit')
    for bench in a.bench.split(','):
        C = D.load_pool(W_, bench, 'C', split, gold, labels); Cd = C.subset(C.side == 'dev')
        rawC = D.readout(Cd); ev, s0, f0, X0 = S.step_eval(Cd, rawC)
        AM = S.AnswerModel(Cd, rawC, s0); one = np.ones(16)
        FC = RA.feats(AM, Cd, rawC, one)[0]; AM.fit_fusion(FC); sC = AM.score(FC)
        Wm = Cd.y_ans.astype(bool); pt = float(Wm.mean()); key = 'F1' if bench == 'processbench' else 'PRMScore'
        r = R[bench] = dict(dev=dict(answers=Cd.n, wrong=int(Wm.sum()), pi_true=pt, sA_auroc=L.auroc(Wm, sC)), types={}, skipped={})
        Rx = {}
        for V in TYPES:
            try:
                P = D.load_pool(W_, bench, V, split)
            except (AssertionError, FileNotFoundError, KeyError) as e:
                r['skipped'][V] = f'{type(e).__name__}: {str(e)[:160]}'; continue
            Pd, nd = RA.aligned(P.subset(P.side == 'dev'), Cd)
            if Pd.n == 0:
                r['skipped'][V] = 'no dev variant aligned to a parent'; continue
            pj = RA.parent_index(Pd, Cd); sP = AM.score(RA.feats(AM, Pd, D.readout(Pd), one)[0])
            x = np.full(Cd.n, np.nan); dup = int(len(pj) - len(np.unique(pj)))
            for i, v in zip(pj, sP - sC[pj]):          # one variant per parent and type; a duplicate keeps the first one
                if np.isnan(x[i]): x[i] = v
            Rx[V] = x; m = np.isfinite(x)
            sp = lambda cls: float(spearmanr(x[m & cls], sC[m & cls])[0]) if (m & cls).sum() > 10 else None
            r['types'][V] = dict(n=int(m.sum()), coverage=float(m.mean()), dropped_parent_mismatch=nd, duplicate_parents=dup,
                                 reaction_median_wrong=float(np.median(x[m & Wm])), reaction_median_clean=float(np.median(x[m & ~Wm])),
                                 auroc=L.auroc(Wm[m], -x[m]), auroc_inside_sA_strata=strata_auroc(Wm[m], -x[m], sC[m]),
                                 spearman_with_parent_sA_wrong=sp(Wm), spearman_with_parent_sA_clean=sp(~Wm))
            log(bench, V, RA.rnd(r['types'][V]))
        names = list(Rx)
        if len(names) < 3:
            r['verdict'] = f'fewer than 3 error types ({names}): Dawid-Skene cannot identify pi'; log(bench, r['verdict']); continue
        X = np.column_stack([Rx[V] for V in names]); A = np.isfinite(X)
        cc = {}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                m = A[:, i] & A[:, j]
                cc[f'{names[i]}-{names[j]}'] = dict(
                    n=int(m.sum()), wrong=float(spearmanr(X[m & Wm, i], X[m & Wm, j])[0]) if (m & Wm).sum() > 10 else None,
                    clean=float(spearmanr(X[m & ~Wm, i], X[m & ~Wm, j])[0]) if (m & ~Wm).sum() > 10 else None)
        r['conditional_spearman'] = cc
        vals = [v[c] for v in cc.values() for c in ('wrong', 'clean') if v[c] is not None]
        r['conditional_spearman_mean_abs'] = float(np.mean(np.abs(vals)))
        log(bench, 'conditional Spearman', RA.rnd(cc), 'mean |.|', RA.rnd(r['conditional_spearman_mean_abs']))
        Z = np.column_stack([rank_normal(-X[:, j]) for j in range(X.shape[1])])     # higher = wrong
        r['gate_reference'] = dict(no_gate=gate_metrics(Cd, sC, f0, 1., key), true_pi=gate_metrics(Cd, sC, f0, pt, key))
        e = EXPECT_TRUE_PI[bench]; now = r['gate_reference']['true_pi'][e[0]]
        r['code_check_true_pi_gate'] = dict(metric=e[0], now=now, expected=e[1], pass_=bool(abs(now - e[1]) < .01))
        log(bench, 'code check (true-pi gate vs GATE_DEV)', RA.rnd(r['code_check_true_pi_gate']))
        fits = {}
        for scope, rows in (('complete', A.all(1)), ('missing_allowed', A.sum(1) >= 2)):
            for cut in (.35, .5, .65):
                Vt = np.where(A, (X <= np.nanquantile(X, cut, axis=0)).astype(float), np.nan)
                fd, qd = ds_binary(Vt[rows])
                fits[f'DS binary, cut q{int(cut * 100)}, {scope}'] = dict(n=int(rows.sum()), **fd,
                                                                          posterior_auroc=L.auroc(Wm[rows], qd))
                if cut == .5:
                    fg, qg = gauss_lc(Z[rows], qd)
                    fits[f'Gaussian latent class, {scope}'] = dict(n=int(rows.sum()), **fg, posterior_auroc=L.auroc(Wm[rows], qg))
        for k_, f in fits.items():
            sub = (A.all(1) if k_.endswith('complete') else A.sum(1) >= 2)
            f['pi_true_on_rows'] = float(Wm[sub].mean()); f['abs_err'] = abs(f['pi_hat'] - f['pi_true_on_rows'])
            f['pi_check_pass_10pts'] = bool(f['abs_err'] <= .10)
            f['gate'] = gate_metrics(Cd, sC, f0, f['pi_hat'], key)
            log(bench, k_, 'n', f['n'], 'pi-hat', RA.rnd(f['pi_hat']), 'true', RA.rnd(f['pi_true_on_rows']),
                'post AUROC', RA.rnd(f['posterior_auroc']), '| gate', RA.rnd(f['gate']))
        r['fits'] = fits
        # identification warning: when the views carry little independent signal, binary DS returns about the vote rate
        r['ds_pi_hat_vs_vote_cut'] = {scope: {f'q{int(c * 100)}': fits[f'DS binary, cut q{int(c * 100)}, {scope}']['pi_hat']
                                              for c in (.35, .5, .65)} for scope in ('complete', 'missing_allowed')}
        log(bench, 'DS pi-hat vs vote cut (pi-hat that follows the cut = not identified)', RA.rnd(r['ds_pi_hat_vs_vote_cut']))
        log(bench, 'references', RA.rnd(r['gate_reference']))
        json.dump(RA.rnd(R), open(out / 'REACTION_DS.json', 'w'), indent=1, default=str)
    R['seconds'] = round(time.time() - t0, 1)
    json.dump(RA.rnd(R), open(out / 'REACTION_DS.json', 'w'), indent=1, default=str)
    (out / 'reaction_ds.log').write_text('\n'.join(LOG)); log('done', R['seconds'], 's')


if __name__ == '__main__':
    main()
