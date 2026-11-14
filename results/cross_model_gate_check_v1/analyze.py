"""Cross-model answer gate, go/no-go check, stage 2 (CPU, any machine, 2026-11-14).

Input: steps/<bench>__<prm>.csv.gz from extract.py (label-free step score s per base model; labels for diagnostic only).

Definitions (per benchmark, per base model m):
  p_i      step p-value = share of ALL steps of the benchmark (dataset-wide ECDF, this model) with score >= s_i.
           Small p_i = the step is more error-like than most steps of the dataset. No per-answer normalization.
  Simes    T_simes = -log10( min_i n * p_(i) / i ), p_(1) <= ... <= p_(n) the sorted p-values of the n steps of
           one answer. Large = the answer has steps that are extreme for the dataset, corrected for n.
  HC       T_hc = max_i sqrt(n) (i/n - p_(i)) / sqrt(p_(i) (1 - p_(i)))  (Higher-Criticism, p clipped to the ECDF grid).
  A_mean   answer mean of s (the "whole-answer level" view of the same model; reference for within-model dependence).
  y        answer error = 1 if any step of the answer carries an error label. Diagnostic only.

Reports:
  (1) AUROC of each answer statistic for y: pooled and length-stratified (5 bins of n_steps; mean of within-bin AUROC
      weighted by bin size). n_steps alone is the length-only reference.
  (2) Conditional |Spearman| given y (weighted mean of the y=0 and y=1 values), and given y and length bin.
      Cross-model: T_simes of model a vs model b. Within-model reference: A_mean_m vs T_simes_m (A-S view pair).
  (3) Dawid-Skene with 3 binary raters (vote_m = T_simes_m > its own q-quantile, label-free), 2 latent classes, EM on
      the 8 vote patterns, 50 restarts. Error class = class with the higher mean vote rate. pi-hat vs true prevalence,
      bootstrap 90% interval (300 resamples of answers). Main q = .5; q = .3 and .7 as sensitivity.
Go criterion (fixed before the run): cross-model conditional |corr| clearly below .26 AND |pi-hat - pi| <= .10 on
ProcessBench (main triplet, q = .5).

    python analyze.py
"""
import csv, gzip, itertools, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
STEPS = HERE / 'steps'
MODELS = {'qwen_prm_7b': '7B-Inst', 'qwen_prm_72b': '72B-Inst', 'pure_prm': '7B-base', 'rlhflow_prm': 'Llama-8B'}
TRIPLETS = {'main (7B-Inst, 72B-Inst, Llama-8B)': ('qwen_prm_7b', 'qwen_prm_72b', 'rlhflow_prm'),
            'alt (7B-base, 72B-Inst, Llama-8B)': ('pure_prm', 'qwen_prm_72b', 'rlhflow_prm')}
BENCHES = ('processbench', 'prmbench', 'socratic', 'hard2verify', 'crv_arith', 'crv_gsm8k')
QS = (.5, .3, .7)
RNG = np.random.default_rng(20261114)


def read_cell(bench, prm):
    p = STEPS / f'{bench}__{prm}.csv.gz'
    if not p.exists():
        return None
    by = defaultdict(list)
    with gzip.open(p, 'rt', encoding='utf8') as f:
        for r in csv.DictReader(f):
            by[r['uid']].append((int(r['step']), int(r['y']), float(r['s'])))
    uids = list(by)
    s = np.concatenate([[t[2] for t in sorted(by[u])] for u in uids])
    pv = rankdata(-s, method='max') / len(s)                      # share of steps with score >= s_i
    lo = 1. / (len(s) + 1)
    out, k = {}, 0
    for u in uids:
        rows = sorted(by[u]); n = len(rows); pu = np.sort(pv[k:k + n]); su = s[k:k + n]; k += n
        i = np.arange(1, n + 1)
        simes = float(np.min(n * pu / i))
        pc = np.clip(pu, lo, 1 - lo)
        hc = float(np.max(np.sqrt(n) * (i / n - pc) / np.sqrt(pc * (1 - pc))))
        out[u] = dict(n=n, y=int(any(t[1] for t in rows)), simes=-np.log10(max(simes, 1e-300)), hc=hc,
                      amean=float(su.mean()))
    return out


def length_bins(n, k=5):
    r = rankdata(n, method='average') / len(n)
    return np.minimum((r * k).astype(int), k - 1) if len(np.unique(n)) > 1 else np.zeros(len(n), int)


def auc(y, x):
    return float(roc_auc_score(y, x)) if 0 < y.sum() < len(y) else np.nan


def strat_auc(y, x, b):
    v, w = [], []
    for k in np.unique(b):
        m = b == k; a = auc(y[m], x[m])
        if np.isfinite(a): v.append(a); w.append(m.sum())
    return float(np.average(v, weights=w)) if v else np.nan


def cond_corr(x1, x2, y, b=None):
    """weighted mean |Spearman| within y (and within length bin if b is given); also per-class signed values."""
    v, w, per = [], [], {}
    groups = [(c,) for c in (0, 1)] if b is None else [(c, k) for c in (0, 1) for k in np.unique(b)]
    for g in groups:
        m = y == g[0] if b is None else (y == g[0]) & (b == g[1])
        if m.sum() < 20 or np.std(x1[m]) == 0 or np.std(x2[m]) == 0: continue
        r = spearmanr(x1[m], x2[m]).statistic
        v.append(abs(r)); w.append(m.sum())
        if b is None: per[f'y{g[0]}'] = float(r)
    return (float(np.average(v, weights=w)) if v else np.nan), per


def ds_fit(counts, pats, restarts=50, iters=500):
    """2-class Dawid-Skene on binary votes. counts: weight of each pattern (8). -> pi(error), rates, loglik."""
    best = None; W = counts / counts.sum()
    for _ in range(restarts):
        pi = RNG.uniform(.05, .95); a = RNG.uniform(.05, .95, 3); b = RNG.uniform(.05, .95, 3)   # a: P(v=1|err), b: P(v=1|clean)
        for _ in range(iters):
            le = pi * np.prod(np.where(pats, a, 1 - a), 1); lc = (1 - pi) * np.prod(np.where(pats, b, 1 - b), 1)
            g = le / (le + lc)
            pi_n = float((W * g).sum())
            a_n = (W[:, None] * g[:, None] * pats).sum(0) / max(pi_n, 1e-12)
            b_n = (W[:, None] * (1 - g)[:, None] * pats).sum(0) / max(1 - pi_n, 1e-12)
            a_n, b_n = np.clip(a_n, 1e-6, 1 - 1e-6), np.clip(b_n, 1e-6, 1 - 1e-6)
            done = abs(pi_n - pi) < 1e-10 and np.abs(a_n - a).max() < 1e-10
            pi, a, b = pi_n, a_n, b_n
            if done: break
        ll = float((W * np.log(le + lc + 1e-300)).sum())
        if best is None or ll > best[3] + 1e-12: best = (pi, a.copy(), b.copy(), ll)
    pi, a, b, ll = best
    if a.mean() < b.mean(): pi, a, b = 1 - pi, b, a                 # error class = higher vote rate
    return pi, a, b, ll


PATS = np.array(list(itertools.product((0, 1), repeat=3)), bool)


def ds(votes, boot=300):
    idx = votes @ np.array([4, 2, 1])
    cnt = np.bincount(idx, minlength=8).astype(float)
    pi, a, b, ll = ds_fit(cnt, PATS)
    bs = []
    for _ in range(boot):
        c = RNG.multinomial(int(cnt.sum()), cnt / cnt.sum()).astype(float)
        bs.append(ds_fit(c, PATS, restarts=8, iters=300)[0])
    return dict(pi_hat=pi, lo=float(np.quantile(bs, .05)), hi=float(np.quantile(bs, .95)),
                tpr=a.tolist(), fpr=b.tolist(), loglik=ll,
                degenerate=bool(pi < .02 or pi > .98 or np.max(np.abs(a - b)) < .05))


def main():
    R1, R2, R3, info = [], [], [], {}
    for bench in BENCHES:
        cells = {m: read_cell(bench, m) for m in MODELS}
        cells = {m: v for m, v in cells.items() if v}
        if len(cells) < 3:
            info[bench] = f'only {len(cells)} models with steps; skipped'; continue
        common = sorted(set.intersection(*[set(v) for v in cells.values()]))
        ref = cells.get('qwen_prm_7b') or next(iter(cells.values()))
        y = np.array([ref[u]['y'] for u in common]); n = np.array([ref[u]['n'] for u in common])
        agree = {MODELS[m]: float(np.mean([v[u]['y'] == ref[u]['y'] for u in common])) for m, v in cells.items()}
        b = length_bins(n)
        info[bench] = dict(answers=len(common), prevalence=float(y.mean()), label_agreement=agree,
                           models=[MODELS[m] for m in cells], mean_steps=float(n.mean()))
        T = {m: {k: np.array([v[u][k] for u in common]) for k in ('simes', 'hc', 'amean')} for m, v in cells.items()}
        # (1)
        R1.append(dict(bench=bench, model='(length only)', stat='n_steps', auc_pooled=auc(y, n), auc_len_strat=np.nan))
        for m in cells:
            for k in ('simes', 'hc', 'amean'):
                R1.append(dict(bench=bench, model=MODELS[m], stat=k, auc_pooled=auc(y, T[m][k]),
                               auc_len_strat=strat_auc(y, T[m][k], b)))
        # (2)
        for m1, m2 in itertools.combinations(cells, 2):
            c, per = cond_corr(T[m1]['simes'], T[m2]['simes'], y); cl, _ = cond_corr(T[m1]['simes'], T[m2]['simes'], y, b)
            R2.append(dict(bench=bench, kind='cross-model', pair=f'{MODELS[m1]} vs {MODELS[m2]}', cond_abs_corr=c,
                           cond_abs_corr_len=cl, corr_y0=per.get('y0'), corr_y1=per.get('y1')))
        for m in cells:
            c, per = cond_corr(T[m]['amean'], T[m]['simes'], y); cl, _ = cond_corr(T[m]['amean'], T[m]['simes'], y, b)
            R2.append(dict(bench=bench, kind='within-model A-S', pair=f'{MODELS[m]}: A_mean vs Simes', cond_abs_corr=c,
                           cond_abs_corr_len=cl, corr_y0=per.get('y0'), corr_y1=per.get('y1')))
        # (3)
        for tname, trip in TRIPLETS.items():
            if not all(m in cells for m in trip): continue
            for q in QS:
                V = np.column_stack([T[m]['simes'] > np.quantile(T[m]['simes'], q) for m in trip])
                d = ds(V)
                R3.append(dict(bench=bench, triplet=tname, q=q, pi_true=float(y.mean()), pi_hat=d['pi_hat'],
                               abs_err=abs(d['pi_hat'] - float(y.mean())), lo=d['lo'], hi=d['hi'],
                               tpr=' '.join(f'{x:.2f}' for x in d['tpr']), fpr=' '.join(f'{x:.2f}' for x in d['fpr']),
                               degenerate=d['degenerate']))
        print(bench, info[bench], flush=True)
    for name, rows in (('AUROC', R1), ('COND_CORR', R2), ('DS_PI', R3)):
        with open(HERE / f'{name}.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
            for r in rows: w.writerow({k: (f'{v:.4f}' if isinstance(v, float) else v) for k, v in r.items()})
    pb = [r for r in R3 if r['bench'] == 'processbench' and r['triplet'].startswith('main') and r['q'] == .5]
    cc = [r['cond_abs_corr'] for r in R2 if r['bench'] == 'processbench' and r['kind'] == 'cross-model'
          and 'base' not in r['pair']]
    verdict = dict(pb_pi_abs_err=pb[0]['abs_err'] if pb else None,
                   pb_cross_model_cond_corr_max=max(cc) if cc else None,
                   go=bool(pb and cc and pb[0]['abs_err'] <= .10 and max(cc) < .20),
                   rule='GO if PB |pi_hat - pi| <= .10 (main triplet, q=.5) AND all PB cross-model cond |corr| of the main '
                        'triplet < .20 ("clearly below .26")')
    json.dump(dict(info=info, verdict=verdict), open(HERE / 'VERDICT.json', 'w'), indent=1)
    print(json.dumps(verdict, indent=1))


if __name__ == '__main__':
    main()
