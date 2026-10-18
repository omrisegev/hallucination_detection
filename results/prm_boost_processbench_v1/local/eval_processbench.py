"""Evaluate PRM boosting on ProcessBench (local only: the gold labels never leave this machine).

Inputs: a directory with, per PRM adapter <a>, the label-free score files written on AIRCC
  SCORES_<a>.npz        score_external.py   arms PRM, HALF_N16, LSML_N16_PRMG, LSML_CONT (frozen arm list)
  SCORES_CHAIN_<a>.npz  score_chain_pb.py   arms TELE/PRMG/CHAIN for N13 and N16 (post-registration arms)
Both are merged per adapter. score__PRM must agree exactly, and the CHAIN file's LSML_N16_PRMG must agree with the
frozen file's to 1e-9 (parity). Every score is risk-oriented on the non-empty steps.

Decision rules (as in ../../prm_boost_external_v1/eval_external.py):
  <arm>_Q80   answer-z of the arm on the non-empty steps, pooled 80th percentile -> flagged erroneous;
              empty-token steps are flagged erroneous (external convention; their count is reported)
  PRM_native  the PRM's own 0.5 threshold, ProcessBench tie rule: reward < 0.5 -> erroneous (from score__PRM > 0)
  PRM_native_adapter  the adapters' native_valid (reward > 0.5 is valid, so 0.5 is erroneous), as on PRMBench/Socratic
  (all 5 adapters use the 0.5 reward threshold; the two rules differ only on rewards of exactly 0.5)
Predicted first error = first flagged step, -1 if none. Metrics: see pb_metrics.py (official ProcessBench F1 per
subset + average, PRISM TPR/TNR names, prefix-domain step confusion, within-answer AUC, pooled step AUROC with three
declared targets for CROP).

Coverage: answers the PRM could not score (counted context-overflow failures, no truncation) are absent from its
SCORES file. Rows are given for population `scored` (that PRM's answers) and `common` (answers scored by every PRM
in the directory); `avg_f1_failures_wrong` counts every unscored answer as a miss on the full 3,400.
Paired bootstrap over answers, stratified by subset, 1,000 draws, seed 20261018: every arm/rule minus PRM_native and
minus PRM_Q80, for average F1, within-AUC and pooled AUROC (prefix and first_only targets). It runs on `common` when coverage differs between
PRMs, otherwise on `scored`.

    .venv/bin/python results/prm_boost_processbench_v1/local/eval_processbench.py <scores dir> <out dir> \
        [--gold results/prm_boost_processbench_v1/local/processbench_gold.json]
"""
import argparse, csv, hashlib, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pb_metrics as PM  # noqa: E402

DRAWS, SEED, Q = 1000, 20261018, .8
FROZEN = ('PRM', 'HALF_N16', 'LSML_N16_PRMG', 'LSML_CONT')


def bank_note(arm):
    if arm.endswith('N16') or arm in ('HALF_N16', 'LSML_N16_PRMG'):
        return 'N16 (3 digit channels)'
    if arm.endswith('N13'):
        return 'N13 (digit-free)'
    if arm == 'LSML_CONT':
        return 'S3 (3 label-chosen PRMBench channels, digit-free)'
    return 'none'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def az_q80(s, off, q=Q):
    az = np.zeros_like(s)
    for a, b in zip(off[:-1], off[1:]):
        v = s[a:b]; az[a:b] = (v - v.mean()) / max(v.std(), 1e-8)
    return az >= np.quantile(az, q)


def load_model(files):
    """Merge the score files of one adapter into {arm: risk on non-empty rows} + layout."""
    base, arms = None, {}
    for f in files:
        S = dict(np.load(f, allow_pickle=False))
        lay = {k: S[k] for k in ('uids', 'nonempty', 'nonempty_lengths', 'native_valid')}
        if base is None:
            base = lay
        else:
            for k in lay:
                if not np.array_equal(base[k], lay[k]):
                    raise SystemExit(f'{f}: {k} differs from the other score file of this adapter')
        for k in [k for k in S if k.startswith('score__')]:
            arm, v = k[len('score__'):], np.asarray(S[k], float)
            if arm in arms:
                if arm == 'PRM' and not np.array_equal(arms[arm], v):
                    raise SystemExit(f'{f}: score__PRM parity FAIL')
                if not np.allclose(arms[arm], v, atol=1e-9, rtol=0):
                    raise SystemExit(f'{f}: {arm} parity FAIL, max gap {np.max(np.abs(arms[arm] - v))}')
                continue
            arms[arm] = v
    return base, arms


def expand(lay, s_ne):
    """Non-empty-row scores -> all steps (NaN on empty steps)."""
    ne = lay['nonempty'].astype(bool); full = np.full(len(ne), np.nan); full[ne] = s_ne
    return full


def decisions(lay, arms):
    """{rule: flags over all steps} for every arm and the PRM's native rule."""
    ne = lay['nonempty'].astype(bool); lens = lay['nonempty_lengths']; starts = np.r_[0, np.cumsum(lens)]
    off = np.r_[0, np.cumsum(np.add.reduceat(ne.astype(int), starts[:-1]))] if len(lens) else np.r_[0]
    out = {}
    for arm, s in arms.items():
        f = np.ones(len(ne), bool); f[ne] = az_q80(s, off); out[f'{arm}_Q80'] = (arm, f)
    # ProcessBench's PRM rule flags reward < 0.5 (prm_scorer.localize_first_error, job 177776, PRISM's baseline).
    # risk = log((1-r)/r) is > 0 exactly when r < 0.5; the adapters' native_valid (r > 0.5) also flags r == 0.5,
    # which bf16 rewards hit (4 of 3,400 answers for PRM-7B). Empty-token steps fall back to native_valid.
    f = lay['native_valid'].astype(int) == 0; f[ne] = arms['PRM'] > 0
    out['PRM_native'] = ('PRM', f)
    out['PRM_native_adapter'] = ('PRM', lay['native_valid'].astype(int) == 0)
    return out


def subset_view(lay, arms, keep_uids):
    """Restrict a model's layout and scores to answers in keep_uids (order kept)."""
    uids = [str(u) for u in lay['uids']]; lens = lay['nonempty_lengths']; starts = np.r_[0, np.cumsum(lens)]
    ne = lay['nonempty'].astype(bool); ne_idx = np.cumsum(ne) - 1
    sel = [i for i, u in enumerate(uids) if u in keep_uids]
    rows = np.concatenate([np.arange(starts[i], starts[i + 1]) for i in sel]) if sel else np.array([], int)
    lay2 = {'uids': np.array([uids[i] for i in sel]), 'nonempty': ne[rows], 'nonempty_lengths': lens[sel],
            'native_valid': lay['native_valid'][rows]}
    rne = rows[ne[rows]]
    return lay2, {k: v[ne_idx[rne]] for k, v in arms.items()}


def per_answer_preds(lens, flags):
    starts = np.r_[0, np.cumsum(lens)]
    return np.array([PM.first_flag(flags[a:b]) for a, b in zip(starts[:-1], starts[1:])])


def stratified_draws(subsets, rng, draws=DRAWS):
    cols = [np.flatnonzero(subsets == s) for s in PM.SUBSETS if (subsets == s).any()]
    return np.column_stack([rng.choice(c, size=(draws, len(c))) for c in cols]), np.concatenate(cols)


def avg_f1_draws(W, labels, preds, subsets):
    """Average subset F1 for every row of the index matrix W (D x n)."""
    hit = (preds == labels); err = labels != PM.NO_ERROR; f1 = []
    for s in PM.SUBSETS:
        m = subsets[W[0]] == s   # stratified: a column's subset is fixed across draws
        if not m.any(): continue
        Wi = W[:, m]; e = err[Wi]; h = hit[Wi]
        ea = 100 * (h & e).sum(1) / np.maximum(e.sum(1), 1); ca = 100 * (h & ~e).sum(1) / np.maximum((~e).sum(1), 1)
        f1.append(np.where(ea + ca > 0, 2 * ea * ca / np.maximum(ea + ca, 1e-12), 0.))
    return np.mean(f1, axis=0)


def weighted_auc_draws(y, s, step_answer, counts, chunk=100):
    """Pooled AUROC on resampled answers: each step weighted by its answer's multiplicity in the draw."""
    order = np.argsort(s, kind='stable'); s, y, step_answer = s[order], y[order], step_answer[order]
    starts = np.r_[0, np.flatnonzero(np.diff(s) != 0) + 1]; out = []
    for c0 in range(0, len(counts), chunk):
        w = counts[c0:c0 + chunk][:, step_answer]
        gp = np.add.reduceat(w * y, starts, axis=1); gn = np.add.reduceat(w * ~y, starts, axis=1)
        below = np.cumsum(gn, axis=1) - gn
        out.append((gp * (below + .5 * gn)).sum(1) / (gp.sum(1) * gn.sum(1)))
    return np.concatenate(out)


def evaluate(lay, arms, gold, population, all_uids, rng, boot):
    uids = [str(u) for u in lay['uids']]; lens = lay['nonempty_lengths'].astype(int)
    G = [gold[u] for u in uids]
    if any(g['n_steps'] != n for g, n in zip(G, lens)):
        raise SystemExit('step count differs from gold')
    labels = np.array([g['label'] for g in G]); subsets = np.array([g['subset'] for g in G])
    full = {arm: expand(lay, s) for arm, s in arms.items()}
    within = {arm: PM.within_auc_per_answer(labels, lens, full[arm]) for arm in arms}
    rows, sub_rows, stats = [], [], {}
    n_missing = len(all_uids) - len(uids)
    for name, (arm, flags) in decisions(lay, arms).items():
        preds = per_answer_preds(lens, flags)
        off = PM.official(labels, preds, subsets)
        # counted failures: every answer this PRM did not score is a miss, on the full population
        miss = {}
        for s in PM.SUBSETS:
            gl = [g for g in gold.values() if g['subset'] == s]
            e_all = sum(g['label'] != PM.NO_ERROR for g in gl); c_all = len(gl) - e_all
            o = off.get(s)
            if o is None or not e_all or not c_all: continue
            ea = (o['error_acc'] or 0.) * o['n_error'] / e_all; ca = (o['correct_acc'] or 0.) * o['n_correct'] / c_all
            miss[s] = 2 * ea * ca / (ea + ca) if ea + ca else 0.
        conf = PM.step_confusion(labels, lens, flags)
        row = dict(population=population, arm=name, score=arm, bank=bank_note(arm),
                   frozen_arm=arm in FROZEN, n_answers=len(uids), n_unscored=n_missing,
                   avg_f1=off['average']['f1'],
                   avg_f1_failures_wrong=float(np.mean(list(miss.values()))) if len(miss) == 4 else None,
                   **{f'{s}_{k}': off[s][k] for s in PM.SUBSETS if s in off for k in ('error_acc', 'correct_acc', 'f1')},
                   within_auc=float(np.nanmean(within[arm])), within_n=int(np.isfinite(within[arm]).sum()),
                   **{f'auroc_{t}': PM.pooled_auroc(labels, lens, full[arm], t) for t in PM.TARGETS},
                   **{f'auroc_prefix_{s}': PM.pooled_auroc(labels[subsets == s], lens[subsets == s],
                                                           full[arm][np.repeat(subsets == s, lens)], 'prefix')
                      for s in PM.SUBSETS},
                   step_tpr=conf['tpr'], step_tnr=conf['tnr'], step_f1_error=conf['f1_error'],
                   n_empty_steps=int((~lay['nonempty'].astype(bool)).sum()))
        rows.append(row)
        for s in PM.SUBSETS:
            if s in off:
                sub_rows.append(dict(population=population, arm=name, subset=s, **off[s],
                                     prism_tpr=off[s]['error_acc'], prism_tnr=off[s]['correct_acc'],
                                     prism_f1=off[s]['f1']))
        stats[name] = (arm, preds)
    if not boot:
        return rows, sub_rows, []
    W, _ = stratified_draws(subsets, rng)
    counts = np.zeros((DRAWS, len(uids)))
    for d in range(DRAWS):
        counts[d] = np.bincount(W[d], minlength=len(uids))
    step_answer = np.repeat(np.arange(len(uids)), lens); tg = PM.step_arrays(labels, lens); auroc_d = {}
    for t in ('prefix', 'first_only'):
        y, inc = tg[t]
        for arm in arms:
            k = inc & np.isfinite(full[arm])
            auroc_d[t, arm] = weighted_auc_draws(y[k], full[arm][k], step_answer[k], counts)
    f1_d = {n: avg_f1_draws(W, labels, p, subsets) for n, (_, p) in stats.items()}
    wa_d = {arm: np.nanmean(within[arm][W], axis=1) for arm in arms}
    point = {r['arm']: r for r in rows}; boot_rows = []
    for name, (arm, _) in stats.items():
        for ref in ('PRM_native', 'PRM_Q80'):
            if name == ref: continue
            rarm = stats[ref][0]
            def gap(k): return np.nan if None in (point[name][k], point[ref][k]) else point[name][k] - point[ref][k]
            for mt, d, p in (('avg_f1', f1_d[name] - f1_d[ref], gap('avg_f1')),
                             ('within_auc', wa_d[arm] - wa_d[rarm], gap('within_auc')),
                             ('auroc_prefix', auroc_d['prefix', arm] - auroc_d['prefix', rarm], gap('auroc_prefix')),
                             ('auroc_first_only', auroc_d['first_only', arm] - auroc_d['first_only', rarm],
                              gap('auroc_first_only'))):
                boot_rows.append(dict(population=population, contrast=f'{name} - {ref}', metric=mt, delta=float(p),
                                      ci_low=float(np.quantile(d, .025)), ci_high=float(np.quantile(d, .975))))
    return rows, sub_rows, boot_rows


def write_csv(path, rows):
    if not rows: return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, keys); w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('scores', type=Path); ap.add_argument('out', type=Path)
    ap.add_argument('--gold', type=Path, default=Path(__file__).resolve().parent / 'processbench_gold.json')
    ap.add_argument('--no-bootstrap', action='store_true')
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    gold = {g['uid']: g for g in json.loads(a.gold.read_text(encoding='utf8'))}
    models = {}
    for f in sorted(a.scores.glob('SCORES_*.npz')):
        stem = f.stem[len('SCORES_'):]
        models.setdefault(stem[len('CHAIN_'):] if stem.startswith('CHAIN_') else stem, []).append(f)
    if not models: raise SystemExit('no SCORES_*.npz in ' + str(a.scores))
    loaded = {m: load_model(fs) for m, fs in models.items()}
    for m, (lay, _) in loaded.items():
        extra = set(map(str, lay['uids'])) - set(gold)
        if extra: raise SystemExit(f'{m}: {len(extra)} uids not in gold')
    common = set.intersection(*[set(map(str, lay['uids'])) for lay, _ in loaded.values()])
    rng = np.random.default_rng(SEED); R, SR, B = [], [], []
    for m, (lay, arms) in loaded.items():
        pops = [('scored', lay, arms)]
        if len(common) < len(lay['uids']):
            pops.append(('common', *subset_view(lay, arms, common)))
        for pop, l2, a2 in pops:
            r, sr, b = evaluate(l2, a2, gold, pop, gold, rng, not a.no_bootstrap and pop == pops[-1][0])
            for x in r + sr + b: x['model'] = m
            R += r; SR += sr; B += b
        print(m, {r['arm']: round(r['avg_f1'], 2) for r in R if r['model'] == m and r['population'] == 'scored'})
    write_csv(a.out / 'RESULTS_processbench.csv', R)
    write_csv(a.out / 'SUBSETS_processbench.csv', SR)
    write_csv(a.out / 'BOOTSTRAP_processbench.csv', B)
    (a.out / 'EVAL_MANIFEST.json').write_text(json.dumps({
        'gold_sha256': sha(a.gold), 'scores': {f.name: sha(f) for fs in models.values() for f in fs},
        'n_common': len(common), 'n_scored': {m: int(len(lay['uids'])) for m, (lay, _) in loaded.items()},
        'draws': 0 if a.no_bootstrap else DRAWS, 'seed': SEED, 'q': Q}, indent=1))


if __name__ == '__main__':
    main()
