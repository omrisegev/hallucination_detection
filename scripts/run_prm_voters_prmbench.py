"""PRMBench: step scorers as label-free voters — MoM error counts and continuous L-SML.

PRMBench counterpart of scripts/run_prm_voters_processbench_pilot.py. PRMBench labels every
step, so the true per-answer error count (len(error_steps)) and the official PRMScore are both
available here; ProcessBench had only the first error.

Inputs (all joined on PRMBench `idx`; rows missing or failed in any voter are dropped and counted):
  --prm7b PATH           Qwen2.5-Math-PRM-7B cache from cluster/run_prmbench_prm.py (int-keyed,
                         `rewards`, `labels`); also supplies idx/classification/error_steps/n_steps.
  --prm NAME=PATH        extra PRM, prmbench_telemetry.pkl from run_prmbench_prm_telemetry.py
                         (`rewards`, `native_labels`; rows with `failure` are dropped). Repeatable.
  --judge NAME=PATH      uPRM judge output of cluster/run_prmbench_step_voters.py (file or glob
                         over shards; `step_logp_err`). Repeatable.
  --critic NAME=PATH     critic output of the same driver (file or shard glob; `prediction`).

Per-voter step views (semantic orientation, higher = more likely erroneous; no label is used):
  PRM     score 1 - reward; native flag = its own native label rule (reward > .5 is valid).
  judge   score P(step incorrect) from the {+,-} marker pair; native flag = P > .5.
  critic  only a first-error index: flag that step and every later one (cumulative), score =
          the same indicator. This voter cannot express "valid after an error".

Arms (flags computed on all 6,969 rows, including the `correct` controls; evaluation follows):
  NATIVE     each voter's own decision; MoM posterior > .5
  Q80        canonical ANSWER_Z_Q80: within-answer z-score, flag >= global 80th percentile
  MOM_COUNT  canonical MoM count arm: per-answer round(sum posterior), flag top-k steps by score
PRMScore = official (f1 + negative_f1) / 2 from spectral_utils.prmbench.prmbench_evaluate
(noncontrol rows pooled, exactly as upstream). Paired source-group bootstrap (1,000 draws) for
L-SML minus the best single voter under each arm; the best single voter is picked post hoc and
is a reference row, not a method.

    .venv/bin/python scripts/run_prm_voters_prmbench.py --prm7b <pkl> \\
        --judge j8b=<dir>/prmbench_uprm_judge.pkl --judge j14b=... \\
        --critic c72b='<dir>/prmbench_critic_shard*of6.pkl' \\
        --prm prm72b=<dir>/prmbench_telemetry.pkl ... --out results/prm_voters_prmbench_v1
"""
import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from spectral_utils.error_count import fit_mom, posterior  # noqa: E402
from spectral_utils.fusion_utils import lsml_continuous  # noqa: E402
from spectral_utils.prmbench import eval_on_hallucination_step, prmbench_evaluate  # noqa: E402

CORRECT = 'correct'
N_BOOT = 1000


def _load(path):
    files = sorted(glob.glob(path))
    if not files:
        raise FileNotFoundError(path)
    out = {}
    for f in files:
        d = pickle.load(open(f, 'rb'))
        for r in d.values():
            if r['idx'] in out:
                raise ValueError(f'duplicate idx {r["idx"]} across {files}')
            out[r['idx']] = r
    return out, files


def recover_step_logp_err(scores):
    """Per-step log P(step incorrect) from uPRM candidate scores (keys 1..T+1), as
    cluster/run_prmbench_step_voters.py:step_error_logprobs, returning the reconstruction gap
    |sum log P(correct) - S(T+1)| instead of raising on it."""
    sc = {int(k): float(v) for k, v in scores.items()}
    T = len(sc) - 1
    out, cum = [], 0.0
    for j in range(1, T + 1):
        lpn = sc[j] - cum
        out.append(lpn)
        cum += np.log1p(-np.exp(min(lpn, -1e-12)))
    return out, abs(cum - sc[T + 1])


def _pair(s):
    name, path = s.split('=', 1)
    return name, path


def load_voters(cfg):
    base, f7 = _load(cfg.prm7b)
    meta = [dict(idx=r['idx'], source_idx=r['source_idx'], classification=r['classification'],
                 error_steps=list(r['error_steps']), n_steps=r['n_steps']) for r in base.values()]
    voters, prov, drops = {}, {'prm7b': f7}, {}

    def prm_view(recs, name, label_key):
        v = {}
        for i, r in recs.items():
            if r.get('failure') or r.get('rewards') is None:
                continue
            rw = np.asarray(r['rewards'], float)
            valid = np.asarray(r[label_key], int)
            v[i] = (1 - rw, valid == 0)
        return v

    voters['prm7b'] = prm_view(base, 'prm7b', 'labels')
    for name, path in map(_pair, cfg.prm):
        recs, prov[name] = _load(path)
        voters[name] = prm_view(recs, name, 'native_labels')
    for name, path in map(_pair, cfg.judge):
        recs, prov[name] = _load(path)
        v, n_rec = {}, 0
        for i, r in recs.items():
            if r.get('failed'):
                continue
            lp = r.get('step_logp_err')
            if lp is None and r.get('scores'):
                # The driver's per-step recovery check uses a 1e-4 relative tolerance, which bf16
                # candidate scores miss by ~1e-3 (2026-10-18: 6,593 / 6,509 of 6,969 judge rows).
                # The candidate scores are intact; recover the per-step view here and bound the gap.
                lp, gap = recover_step_logp_err(r['scores'])
                if gap > cfg.recovery_tol:
                    continue
                n_rec += 1
            if lp is not None:
                v[i] = (np.exp(lp), np.exp(lp) > .5)
        voters[name] = v
        prov[name + '_recovered_rows'] = n_rec
    for name, path in map(_pair, cfg.critic):
        recs, prov[name] = _load(path)
        v = {}
        for i, r in recs.items():
            p = r.get('prediction')
            if p is None or r.get('truncated'):
                continue
            n = r['n_steps']
            flag = (np.arange(n) >= p) if 0 <= p < n else np.zeros(n, bool)
            v[i] = (flag.astype(float), flag)
        voters[name] = v
    keep = []
    for m in meta:
        ok = all(m['idx'] in v and len(v[m['idx']][0]) == m['n_steps'] for v in voters.values())
        if ok:
            keep.append(m)
    for name, v in voters.items():
        drops[name] = sum(m['idx'] not in v or len(v[m['idx']][0]) != m['n_steps'] for m in meta)
    return meta, keep, voters, prov, drops


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


def z(x):
    s = x.std()
    return (x - x.mean()) / (s if s > 0 else 1.)


def lsml(views):
    fused, meta = lsml_continuous(*[z(v) for v in views])
    if np.corrcoef(fused, np.mean([z(v) for v in views], 0))[0, 1] < 0:
        fused = -fused        # label-free orientation: agree with the mean of oriented views
    return fused, dict(K=int(meta['K']), groups=[int(g) for g in meta['c']],
                       degenerate=bool(meta['degenerate']))


def official(rows, off, flags):
    preds = [{'idx': m['idx'], 'labels': (~flags[a:b]).astype(int).tolist()}
             for m, a, b in zip(rows, off[:-1], off[1:])]
    t = prmbench_evaluate(preds, rows)['total']
    return dict(prm_score=(t['f1'] + t['negative_f1']) / 2, f1_valid=t['f1'], f1_error=t['negative_f1'],
                first_error_acc=t['first_error_acc'])


def row_confusion(rows, off, flags):
    """Per-row TP/FP/TN/FN from the official per-row routine (noncontrol rows only)."""
    c = np.zeros((len(rows), 4))
    for i, (m, a, b) in enumerate(zip(rows, off[:-1], off[1:])):
        if m['classification'] == CORRECT:
            continue
        f = eval_on_hallucination_step(m['error_steps'], (~flags[a:b]).astype(int).tolist())['f1_matrix']
        c[i] = [f['TP'], f['FP'], f['TN'], f['FN']]
    return c


def f1(tp, fp, fn):
    d = 2 * tp + fp + fn
    return np.divide(2 * tp, d, out=np.full_like(d, np.nan, dtype=float), where=d > 0)


def score_from_conf(c):
    tp, fp, tn, fn = c.T
    return (f1(tp, fp, fn) + f1(tn, fn, fp)) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prm7b', required=True)
    ap.add_argument('--prm', action='append', default=[])
    ap.add_argument('--judge', action='append', default=[])
    ap.add_argument('--critic', action='append', default=[])
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=20261016)
    ap.add_argument('--recovery-tol', type=float, default=0.02,
                    help='max |log| gap when recovering judge per-step scores from candidate scores')
    cfg = ap.parse_args()
    os.makedirs(cfg.out, exist_ok=True)

    meta, rows, voters, prov, drops = load_voters(cfg)
    names = list(voters)
    if len(names) < 3:
        raise SystemExit('need at least 3 voters (MoM needs m >= 3)')
    n = np.array([m['n_steps'] for m in rows])
    off = np.concatenate([[0], np.cumsum(n)])
    cls = np.array([m['classification'] for m in rows])
    nonctl = cls != CORRECT
    y = np.concatenate([np.isin(np.arange(1, k + 1), m['error_steps']) for m, k in zip(rows, n)])
    true_count = np.array([len(m['error_steps']) for m in rows])

    S = {v: np.concatenate([voters[v][m['idx']][0] for m in rows]) for v in names}
    F = {v: np.concatenate([voters[v][m['idx']][1] for m in rows]).astype(bool) for v in names}
    V = np.column_stack([np.where(F[v], 1., -1.) for v in names])

    mom = fit_mom(V)
    res = dict(n_meta=len(meta), n_joined=len(rows), dropped_by_voter=drops, voters=names,
               provenance=prov, mom_status=mom['status'])
    scores = dict(S)
    fused, lmeta = lsml([S[v] for v in names])
    scores['LSML_all'] = fused
    res['lsml_meta'] = lmeta
    if mom['status'] == 'PASS':
        post = posterior(V, mom)
        mom_count = np.floor(np.add.reduceat(post, off[:-1]) + .5).astype(int)
        scores['MOM_POSTERIOR'] = post
        res['mom'] = dict(prevalence=mom['prevalence'], true_step_prevalence_all=float(y.mean()),
                          psi_sens=[float(x) for x in mom['psi']], eta_spec=[float(x) for x in mom['eta']],
                          true_sens=[float(F[v][y].mean()) for v in names],
                          true_spec=[float((~F[v][~y]).mean()) for v in names])
    else:
        mom_count = None
        res['mom'] = dict(failure=mom.get('failure_reason'))

    # ---- counts ----
    counts = {'TRUTH': true_count}
    for v in names:
        counts[v] = np.add.reduceat(F[v].astype(int), off[:-1])
    if mom_count is not None:
        counts['MOM'] = mom_count
    res['counts'] = {k: dict(mean_noncontrol=float(c[nonctl].mean()), mean_controls=float(c[~nonctl].mean()),
                             mae_noncontrol=float(np.abs(c - true_count)[nonctl].mean()),
                             bias_noncontrol=float((c - true_count)[nonctl].mean()),
                             exact_noncontrol=float((c == true_count)[nonctl].mean()))
                     for k, c in counts.items()}

    # ---- PRMScore ----
    flags = {}
    for k, s in scores.items():
        az = answer_z(s, off)
        flags[(k, 'Q80')] = az >= np.quantile(az, .8)
        if mom_count is not None:
            flags[(k, 'MOM_COUNT')] = top_flags(s, off, mom_count)
    for v in names:
        flags[(v, 'NATIVE')] = F[v]
    if mom_count is not None:
        flags[('MOM_POSTERIOR', 'NATIVE')] = scores['MOM_POSTERIOR'] > .5
    step_nc = np.repeat(nonctl, n)
    table = {}
    for (k, arm), f in flags.items():
        table.setdefault(k, {})[arm] = official(rows, off, f)
    for k, s in scores.items():
        table[k]['step_auc_noncontrol'] = float(roc_auc_score(y[step_nc], s[step_nc]))
    res['prmscore'] = table

    # ---- paired bootstrap: L-SML minus best single, per arm ----
    groups = np.unique([m['source_idx'] for m in rows], return_inverse=True)[1]
    rng = np.random.default_rng(cfg.seed)
    G = groups.max() + 1
    W = rng.multinomial(G, np.full(G, 1 / G), size=N_BOOT).astype(float)[:, groups]
    res['lsml_minus_best_single'] = {}
    for arm in ('Q80', 'MOM_COUNT'):
        if ('LSML_all', arm) not in flags:
            continue
        best = max(names, key=lambda v: table[v][arm]['prm_score'])
        cl = row_confusion(rows, off, flags[('LSML_all', arm)])
        cb = row_confusion(rows, off, flags[(best, arm)])
        d = score_from_conf(W @ cl) - score_from_conf(W @ cb)
        res['lsml_minus_best_single'][arm] = dict(
            best_single=best, point=table['LSML_all'][arm]['prm_score'] - table[best][arm]['prm_score'],
            ci95=[float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))])

    json.dump(res, open(os.path.join(cfg.out, 'RESULTS.json'), 'w'), indent=1, default=float)
    print(f"joined {len(rows)}/{len(meta)}  dropped {drops}  MoM {mom['status']}")
    print('counts (noncontrol): ' + ', '.join(f"{k} {c['mean_noncontrol']:.2f} (MAE {c['mae_noncontrol']:.2f})"
                                               for k, c in res['counts'].items()))
    for k, t in table.items():
        print(f"{k:14s} " + ' '.join(f"{a}={t[a]['prm_score']:.4f}" for a in ('NATIVE', 'Q80', 'MOM_COUNT') if a in t)
              + f"  stepAUC={t['step_auc_noncontrol']:.3f}")
    print(json.dumps(res['lsml_minus_best_single']))


if __name__ == '__main__':
    main()
