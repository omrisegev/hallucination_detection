"""Follow-up to the ProcessBench PRM-voter pilot: per-answer error counts and PRMScore.

Counts. ProcessBench labels only the first error, so the true number of
erroneous steps is not observed. The count used here is the cumulative one:
steps from the first error to the end (0 for a correct answer). Each voter's
count is the same quantity from its native first-error prediction; the MoM
count is the per-answer sum of the step posterior.

PRMScore = (F1_error + F1_valid) / 2 over steps (as in
spectral_utils.error_count_evaluation.summarize), evaluated on the ProcessBench
step domain: every step of a correct answer, and steps up to and including the
gold first error of an erroneous answer (the only steps with known labels).

Thresholds (flags are computed on all steps; labels touch evaluation only):
  NATIVE    the voter's own first-error prediction, cumulative (singles only)
  Q80       canonical ANSWER_Z_Q80: within-answer z-score, global 80th percentile
  MOM_COUNT canonical MoM count arm: per-answer count = round(sum posterior),
            flag that many top-scoring steps in the answer
  MOM_QUOTA global quota: flag the top pi_hat fraction of steps
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_prm_voters_processbench_pilot as P  # noqa: E402

OUT = P.OUT
N_BOOT = 1000
RNG = np.random.default_rng(20261016)


def answer_z(score, off):
    out = np.zeros(len(score))
    for a, b in zip(off[:-1], off[1:]):
        v = score[a:b]
        sd = v.std()
        out[a:b] = (v - v.mean()) / sd if sd > 1e-12 else 0.
    return out


def top_flags(score, off, counts):
    out = np.zeros(len(score), bool)
    for i, (a, b) in enumerate(zip(off[:-1], off[1:])):
        k = int(min(max(counts[i], 0), b - a))
        if k:
            out[a + np.argsort(-score[a:b], kind='stable')[:k]] = True
    return out


def f1(tp, fp, fn):
    d = 2 * tp + fp + fn
    return np.where(d > 0, 2 * tp / np.where(d > 0, d, 1), np.nan)


def confusion_by_answer(y, flags, dom, off):
    g = np.repeat(np.arange(len(off) - 1), np.diff(off))
    m = dom
    return np.column_stack([np.bincount(g[m & k], minlength=len(off) - 1)
                            for k in (y & flags, ~y & flags, y & ~flags, ~y & ~flags)])


def prmscore_from(c):
    tp, fp, fn, tn = c.T if c.ndim == 2 else c
    return (f1(tp, fp, fn) + f1(tn, fn, fp)) / 2


def main():
    rows, _ = P.build(P.load())
    n = np.array([r['n'] for r in rows])
    lab = np.array([r['label'] for r in rows])
    off = np.concatenate([[0], np.cumsum(n)])
    pos = np.concatenate([np.arange(k) for k in n])
    labs = np.repeat(lab, n)
    y = (labs >= 0) & (pos >= labs)
    dom = (labs < 0) | (pos <= labs)
    true_count = np.where(lab >= 0, n - lab, 0)

    pred = {v: np.array([r['pred'][v] for r in rows]) for v in P.VOTERS}
    VS = np.column_stack([np.where((pred[v][np.repeat(np.arange(len(rows)), n)] >= 0) &
                                   (pos >= pred[v][np.repeat(np.arange(len(rows)), n)]), 1., -1.)
                          for v in P.VOTERS])
    mom = P.fit_mom(VS)
    assert mom['status'] == 'PASS', mom.get('failure_reason')
    post = P.posterior(VS, mom)
    mom_count = np.floor(np.add.reduceat(post, off[:-1]) + .5).astype(int)

    views = {v: np.concatenate([r['cum'][v] for r in rows]) for v in P.VOTERS}
    lsml, meta = P.lsml([views[v] for v in P.VOTERS])
    scores = dict(views, LSML_4=lsml, MOM_POSTERIOR=post)

    counts = {'TRUTH': true_count}
    for v in P.VOTERS:
        counts[v] = np.where(pred[v] >= 0, n - pred[v], 0)
    counts['MOM'] = mom_count
    counts['MOM_raw_sum'] = np.add.reduceat(post, off[:-1])
    count_table = {}
    err = lab >= 0
    for k, c in counts.items():
        count_table[k] = dict(mean_all=float(c.mean()), mean_error_answers=float(c[err].mean()),
                              mean_correct_answers=float(c[~err].mean()),
                              mae=float(np.abs(c - true_count).mean()),
                              bias=float((c - true_count).mean()))

    flags = {}
    for name, s in scores.items():
        az = answer_z(s, off)
        flags[(name, 'Q80')] = az >= np.quantile(az, .8)
        flags[(name, 'MOM_COUNT')] = top_flags(s, off, mom_count)
        flags[(name, 'MOM_QUOTA')] = s > np.quantile(s, 1 - mom['prevalence'])
    for v in P.VOTERS:
        flags[(v, 'NATIVE')] = VS[:, P.VOTERS.index(v)] > 0
    flags[('MOM_POSTERIOR', 'NATIVE')] = post > .5

    conf = {k: confusion_by_answer(y, f, dom, off) for k, f in flags.items()}
    table = {}
    for (name, thr), c in conf.items():
        tot = c.sum(0)
        tp, fp, fn, tn = tot
        table.setdefault(name, {})[thr] = dict(
            prm_score=float(prmscore_from(tot)), f1_error=float(f1(tp, fp, fn)), f1_valid=float(f1(tn, fn, fp)),
            flag_rate=float((tp + fp) / tot.sum()), error_recall=float(tp / (tp + fn)))
    # paired answer bootstrap: LSML_4 minus prm7b under the same threshold
    draws = RNG.multinomial(len(rows), np.full(len(rows), 1 / len(rows)), size=N_BOOT).astype(float)
    delta = {}
    for thr in ('Q80', 'MOM_COUNT', 'MOM_QUOTA'):
        d = prmscore_from(draws @ conf[('LSML_4', thr)]) - prmscore_from(draws @ conf[('prm7b', thr)])
        delta[thr] = dict(point=table['LSML_4'][thr]['prm_score'] - table['prm7b'][thr]['prm_score'],
                          ci95=[float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))])
    out = dict(n_answers=len(rows), n_domain_steps=int(dom.sum()), domain_error_rate=float(y[dom].mean()),
               mom_prevalence=mom['prevalence'], lsml_meta=meta, counts=count_table,
               prmscore=table, lsml_minus_prm7b=delta)
    json.dump(out, open(os.path.join(OUT, 'COUNTS_PRMSCORE.json'), 'w'), indent=1)
    print(json.dumps(dict(counts=count_table, delta=delta), indent=1))
    for name, t in table.items():
        print(name, {k: round(v['prm_score'], 4) for k, v in t.items()})


if __name__ == '__main__':
    main()
