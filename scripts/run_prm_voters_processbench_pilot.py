"""Pilot: four ProcessBench step scorers as label-free voters.

Voters (all already scored on the same 3,400 ProcessBench items, joined on id):
  prm7b   Qwen2.5-Math-PRM-7B, per-step P(valid)
  j8b     Qwen3-8B LLM-judge, log-distribution over first-error position
  j14b    Qwen2.5-14B uPRM judge (paper-exact L1), same format as j8b
  c72b    Qwen2.5-72B critic, first-error prediction only

Two label-free uses of the voters, labels touched only in evaluation:
  1. MoM (spectral_utils.error_count.fit_mom) on binary votes derived from each
     voter's own native first-error prediction -> estimated number of erroneous
     answers (answer level) and erroneous prefix-steps (step level).
  2. Continuous L-SML (spectral_utils.fusion_utils.lsml_continuous) on z-scored
     continuous views -> answer AUROC, step AUROC, and ProcessBench F1 with a
     label-free threshold taken from the MoM prevalence estimate.

Step target is cumulative: y_k = 1 iff the gold first error is at or before
step k. It is fully defined by ProcessBench's first-error label (no undefined
steps after the first error).

Orientation is semantic (every view is built so higher = more error); no label
is used to orient, select or threshold.
"""
import json
import os
import pickle
import sys
from glob import glob

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from spectral_utils.error_count import fit_mom, posterior  # noqa: E402
from spectral_utils.fusion_utils import lsml_continuous  # noqa: E402

DATA = '/Users/osegev/Desktop/hallucination_detection'
SUBSETS = ['gsm8k', 'math', 'olympiadbench', 'omnimath']
VOTERS = ['prm7b', 'j8b', 'j14b', 'c72b']
OUT = os.path.join(HERE, 'results', 'prm_voters_processbench_pilot_2026-10-16')
N_BOOT = 2000
RNG = np.random.default_rng(20261016)


def load():
    fl = f'{DATA}/dataset_cache/four_localization'
    j14 = {}
    for p in sorted(glob(f'{DATA}/local_cache/fair_paper_exact_comparisons_v1/'
                         'l1_uprm_judge_full/shards/*.pkl')):
        for r in pickle.load(open(p, 'rb')):
            j14[r['trace_key']] = r
    items = []
    for s in SUBSETS:
        prm = pickle.load(open(f'{fl}/pb_prm_qwen25math7b_full/pb_prm_{s}.pkl', 'rb'))
        j8 = pickle.load(open(f'{fl}/pb_uprm_baseline_qwen3_8b_full/pb_uprm_base_{s}.pkl', 'rb'))
        cr = pickle.load(open(f'{fl}/pb_critic_qwen72b_full/pb_critic_{s}.pkl', 'rb'))
        by = lambda d: {r['id']: r for r in d.values()}
        prm, j8, cr = by(prm), by(j8), by(cr)
        for i, r in prm.items():
            items.append(dict(subset=s, id=i, n=len(r['steps']), label=int(r['label']),
                              prm=r, j8=j8.get(i), j14=j14.get(f'{s}:{i}'), cr=cr.get(i)))
    return items


def judge_dist(rec, n):
    """Normalized first-error distribution: p[k] for steps 0..n-1, p[n] = no error."""
    if rec is None or rec.get('failed') or not rec.get('scores') or len(rec['scores']) != n + 1:
        return None
    lp = np.array([rec['scores'][k] for k in range(1, n + 2)], float)
    p = np.exp(lp - lp.max())
    return p / p.sum()


def build(items):
    """Per voter: native first-error prediction and per-step cumulative error score."""
    rows, drop = [], dict(j8b=0, j14b=0, c72b=0)
    for it in items:
        n = it['n']
        rw = np.asarray(it['prm']['rewards'], float)
        pm = 1 - np.minimum.accumulate(rw)                       # prefix risk
        bad = np.flatnonzero(rw < .5)
        pred = dict(prm7b=int(bad[0]) if len(bad) else -1)
        cum = dict(prm7b=pm)
        for name, key in (('j8b', 'j8'), ('j14b', 'j14')):
            p = judge_dist(it[key], n)
            if p is None:
                drop[name] += 1
                break
            pred[name] = int(np.argmax(p)) if np.argmax(p) < n else -1
            cum[name] = np.cumsum(p[:n])                          # P(first error <= k)
        else:
            c = it['cr']
            if c is None or c.get('prediction') is None or c.get('truncated'):
                drop['c72b'] += 1
                continue
            cp = int(c['prediction'])
            pred['c72b'] = cp if cp < n else -1
            cum['c72b'] = (np.arange(n) >= pred['c72b']).astype(float) if pred['c72b'] >= 0 else np.zeros(n)
            rows.append(dict(subset=it['subset'], id=it['id'], n=n, label=it['label'],
                             pred=pred, cum=cum))
    return rows, drop


def pb_f1(pred, label):
    pred, label = np.asarray(pred), np.asarray(label)
    err, ok = label != -1, label == -1
    a_err = float((pred[err] == label[err]).mean())
    a_ok = float((pred[ok] == -1).mean())
    f1 = 0. if a_err + a_ok == 0 else 2 * a_err * a_ok / (a_err + a_ok)
    return dict(acc_err=a_err, acc_ok=a_ok, f1=f1)


def z(x):
    s = x.std()
    return (x - x.mean()) / (s if s > 0 else 1.)


def first_cross(cum_score, offsets, thr):
    out = []
    for a, b in zip(offsets[:-1], offsets[1:]):
        hit = np.flatnonzero(cum_score[a:b] > thr)
        out.append(int(hit[0]) if len(hit) else -1)
    return np.array(out)


def lsml(views):
    fused, meta = lsml_continuous(*[z(v) for v in views])
    # sml_fuse_signed is sign-ambiguous; orient to agree with the mean of the
    # semantically oriented views (label-free).
    if np.corrcoef(fused, np.mean([z(v) for v in views], 0))[0, 1] < 0:
        fused = -fused
    return fused, dict(K=int(meta['K']), groups=[int(g) for g in meta['c']],
                       degenerate=bool(meta['degenerate']))


def analyse(rows, tag):
    lab = np.array([r['label'] for r in rows])
    yA = (lab != -1).astype(int)
    off = np.concatenate([[0], np.cumsum([r['n'] for r in rows])])
    yS = np.concatenate([(np.arange(r['n']) >= r['label']).astype(int) if r['label'] >= 0
                         else np.zeros(r['n'], int) for r in rows])
    res = dict(tag=tag, n_answers=len(rows), n_steps=int(off[-1]),
               true_err_answers=int(yA.sum()), true_err_prefix_steps=int(yS.sum()))

    # ---- votes from each voter's native prediction (+1 = error) ----
    predA = {v: np.array([r['pred'][v] for r in rows]) for v in VOTERS}
    VA = np.column_stack([np.where(predA[v] >= 0, 1., -1.) for v in VOTERS])
    VS = np.column_stack([np.concatenate([np.where((np.arange(r['n']) >= r['pred'][v]) & (r['pred'][v] >= 0), 1., -1.)
                                          for r in rows]) for v in VOTERS])
    res['voter_single'] = {}
    for i, v in enumerate(VOTERS):
        res['voter_single'][v] = dict(
            vote_rate_answers=float((VA[:, i] > 0).mean()),
            answer_sens=float((VA[yA == 1, i] > 0).mean()),
            answer_spec=float((VA[yA == 0, i] < 0).mean()),
            **{f'pb_{k}': x for k, x in pb_f1(predA[v], lab).items()})

    # ---- 1. MoM count estimation ----
    mom = {}
    for level, V, y in (('answer', VA, yA), ('step', VS, yS)):
        m = fit_mom(V)
        d = dict(status=m['status'], true_prevalence=float(y.mean()), true_count=int(y.sum()),
                 majority_vote_rate=float(((V > 0).sum(1) >= 3).mean()),
                 mean_vote_rate=float((V > 0).mean()))
        if m['status'] == 'PASS':
            post = posterior(V, m)
            d.update(prevalence=m['prevalence'], est_count=float(m['prevalence'] * len(y)),
                     posterior_sum=float(post.sum()),
                     psi_sens=[float(x) for x in m['psi']], eta_spec=[float(x) for x in m['eta']],
                     true_sens=[float((V[y == 1, i] > 0).mean()) for i in range(4)],
                     true_spec=[float((V[y == 0, i] < 0).mean()) for i in range(4)],
                     posterior_auc=float(roc_auc_score(y, post)))
            if level == 'step':
                d['posterior_first_error_pb'] = pb_f1(first_cross(post, off, .5), lab)
                # per-answer count of erroneous prefix steps (= n - first error)
                est_c = np.array([post[a:b].sum() for a, b in zip(off[:-1], off[1:])])
                true_c = np.array([yS[a:b].sum() for a, b in zip(off[:-1], off[1:])])
                d['per_answer_count_mae'] = float(np.abs(est_c - true_c).mean())
                d['per_answer_count_mae_zero_baseline'] = float(true_c.mean())
            else:
                d['posterior_pred_acc'] = float(((post > .5) == yA).mean())
        else:
            d['failure'] = m.get('failure_reason')
        mom[level] = d
    res['mom'] = mom

    # ---- 2. Continuous L-SML ----
    viewsA = dict(prm7b=np.array([r['cum']['prm7b'][-1] for r in rows]),
                  j8b=np.array([r['cum']['j8b'][-1] for r in rows]),
                  j14b=np.array([r['cum']['j14b'][-1] for r in rows]),
                  c72b=(predA['c72b'] >= 0).astype(float))
    viewsS = {v: np.concatenate([r['cum'][v] for r in rows]) for v in VOTERS}
    lm = {}
    for level, views, y in (('answer', viewsA, yA), ('step', viewsS, yS)):
        fused, meta = lsml([views[v] for v in VOTERS])
        avg = np.mean([z(views[v]) for v in VOTERS], 0)
        three = lsml([views[v] for v in ('prm7b', 'j8b', 'j14b')])[0]
        aucs = {v: float(roc_auc_score(y, views[v])) for v in VOTERS}
        aucs.update(LSML_4=float(roc_auc_score(y, fused)), MEAN_Z_4=float(roc_auc_score(y, avg)),
                    LSML_3_no_critic=float(roc_auc_score(y, three)))
        best = max(VOTERS, key=lambda v: aucs[v])   # oracle choice of best single, for reference
        # paired bootstrap over answers: L-SML minus best single voter (scores held fixed)
        ids = np.arange(len(rows))
        if level == 'answer':
            grp = [np.array([i]) for i in ids]
        else:
            grp = [np.arange(a, b) for a, b in zip(off[:-1], off[1:])]
        deltas = []
        for _ in range(N_BOOT):
            pick = np.concatenate([grp[i] for i in RNG.choice(ids, len(ids))])
            yy = y[pick]
            if yy.min() == yy.max():
                continue
            deltas.append(roc_auc_score(yy, fused[pick]) - roc_auc_score(yy, views[best][pick]))
        lm[level] = dict(auc=aucs, meta=meta, best_single=best,
                         delta_vs_best=float(aucs['LSML_4'] - aucs[best]),
                         delta_ci95=[float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))])
        if level == 'step' and mom['step']['status'] == 'PASS':
            # label-free threshold: flag the top pi_hat fraction of steps
            thr = np.quantile(fused, 1 - mom['step']['prevalence'])
            lm[level]['pb_lsml_mom_quota'] = pb_f1(first_cross(fused, off, thr), lab)
    res['lsml'] = lm
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, drop = build(load())
    out = dict(dropped=drop, n_joined=len(rows), results=[])
    for tag, sub in [('pooled', rows)] + [(s, [r for r in rows if r['subset'] == s]) for s in SUBSETS]:
        out['results'].append(analyse(sub, tag))
        print(f'done {tag}', flush=True)
    json.dump(out, open(os.path.join(OUT, 'RESULTS.json'), 'w'), indent=1)
    print(json.dumps(dict(dropped=drop, n_joined=len(rows))))


if __name__ == '__main__':
    main()
