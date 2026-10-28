"""Evaluate PRM boosting on ProcessBench (local only: the gold labels never leave this machine).

Inputs: a directory with, per PRM adapter <a>, the label-free score files written on AIRCC
  SCORES_<a>.npz        score_external_v2.py  arms PRM, HALF_N16, LSML_N16_PRMG, LSML_CONT (frozen arm list, guarded L-SML)
  SCORES_CHAIN_<a>.npz  score_chain_pb.py     arms TELE/PRMG/CHAIN and the MEAN/HALF/EQ controls for N13 and N16,
                                              POS_ONLY, LEN_ONLY
Both are merged per adapter. score__PRM must agree exactly, and the CHAIN file's LSML_N16_PRMG must agree with the
frozen file's to 1e-9 (parity). Every score is risk-oriented on the non-empty steps.

Decision rules (as in ../../prm_boost_external_v1/eval_external.py):
  <arm>_Q80   answer-z of the arm on the non-empty steps, pooled 80th percentile -> flagged erroneous;
              empty-token steps are flagged erroneous (external convention; their count is reported)
  PRM_native  the PRM's own 0.5 threshold, ProcessBench tie rule: reward < 0.5 -> erroneous (from score__PRM > 0)
  PRM_native_adapter  the adapters' native_valid (reward > 0.5 is valid, so 0.5 is erroneous), as on PRMBench/Socratic
  (all 5 adapters use the 0.5 reward threshold; the two rules differ only on rewards of exactly 0.5)
  <arm>_NC    pre-registered rule (PREREGISTRATION_HE.md section 3, pb_metrics.nc_flags): gate = PRM_native fired,
              k = PRM_native flags on non-empty steps, flag the arm's top-k non-empty steps (ties -> earlier step);
              k = 0 with the gate fired -> PRM_native's flags (n_nc_k0 counts these answers)
Predicted first error = first flagged step, -1 if none. Metrics: see pb_metrics.py (official ProcessBench F1 per
subset + average, PRISM TPR/TNR names, prefix-domain step confusion, within-answer AUC, pooled step AUROC with three
declared targets for CROP).

Coverage: answers the PRM could not score (counted context-overflow failures, no truncation) are absent from its
SCORES file. Rows are given for population `scored` (that PRM's answers) and `common` (answers scored by every PRM
in the directory); `avg_f1_failures_wrong` counts every unscored answer as a miss on the full 3,400.
Paired bootstrap (section 4): 1,000 draws, seed 20261018 (a fresh generator per PRM and population, so PRMs scored on
the same answers share the draws), stratified by subset and clustered by question (whitespace-normalised question
text from inputs/answers.json; 988 of the 3,400 answers share a question). Contrasts: every arm/rule minus PRM_native
and minus PRM_Q80, plus CHAIN_N16 minus HALF_N16 / EQ_N16 under NC and Q80 (the separate L-SML claim). Metrics:
average F1, within-AUC, pooled AUROC (prefix, first_only), and on `scored` also avg_f1_failures_wrong (unscored
answers enter the draws as misses). It runs on `scored`, and also on `common` when coverage differs between PRMs.
Verdict (section 4), written to VERDICT_processbench.json: per PRM, CHAIN_N16_NC - PRM_native on `scored`;
E1 = within-AUC CI low > 0, E2 = avg_f1_failures_wrong CI low > -1.0; success = E1 and E2; overall SUCCESS if at
least 3 of the 5 PRMs succeed and one of them is qwen_prm or qwen_prm_72b.

    .venv/bin/python results/prm_boost_processbench_v1/local/eval_processbench.py <scores dir> <out dir> \
        [--gold results/prm_boost_processbench_v1/local/processbench_gold.json] \
        [--answers results/prm_boost_processbench_v1/inputs/answers.json]
"""
import argparse, collections, csv, hashlib, json, sys
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
    if arm in ('POS_ONLY', 'LEN_ONLY'):
        return 'none (position / length only)'
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
    """{rule: (arm, flags over all steps, info)} for every arm under Q80 and NC, and the PRM's native rules."""
    ne = lay['nonempty'].astype(bool); lens = lay['nonempty_lengths']; starts = np.r_[0, np.cumsum(lens)]
    off = np.r_[0, np.cumsum(np.add.reduceat(ne.astype(int), starts[:-1]))] if len(lens) else np.r_[0]
    out = {}
    for arm, s in arms.items():
        f = np.ones(len(ne), bool); f[ne] = az_q80(s, off); out[f'{arm}_Q80'] = (arm, f, {})
    # ProcessBench's PRM rule flags reward < 0.5 (prm_scorer.localize_first_error, job 177776, PRISM's baseline).
    # risk = log((1-r)/r) is > 0 exactly when r < 0.5; the adapters' native_valid (r > 0.5) also flags r == 0.5,
    # which bf16 rewards hit (4 of 3,400 answers for PRM-7B). Empty-token steps fall back to native_valid.
    f = lay['native_valid'].astype(int) == 0; f[ne] = arms['PRM'] > 0
    out['PRM_native'] = ('PRM', f, {})
    out['PRM_native_adapter'] = ('PRM', lay['native_valid'].astype(int) == 0, {})
    for arm, s in arms.items():   # NC: the native rule's gate and count, the arm's order
        g, info = PM.nc_flags(lens, ne, f, s); out[f'{arm}_NC'] = (arm, g, info)
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


def avg_f1_counts(counts, labels, preds, subsets):
    """Average subset F1 for every row of the multiplicity matrix counts (D x n); avg_f1_draws for index draws."""
    hit = (preds == labels); err = labels != PM.NO_ERROR; f1 = []
    for s in PM.SUBSETS:
        m = subsets == s
        if not m.any(): continue
        w = counts[:, m]; e = err[m]; h = hit[m]
        ea = 100 * (w @ (h & e)) / np.maximum(w @ e, 1); ca = 100 * (w @ (h & ~e)) / np.maximum(w @ ~e, 1)
        f1.append(np.where(ea + ca > 0, 2 * ea * ca / np.maximum(ea + ca, 1e-12), 0.))
    return np.mean(f1, axis=0)


def weighted_nanmean(counts, v):
    """Mean of v over the resampled answers of every draw, NaN entries left out (np.nanmean of v[W] per row)."""
    ok = np.isfinite(v)
    return (counts @ np.where(ok, v, 0.)) / (counts @ ok)


def contrast_pairs(names):
    pairs = [(n, r) for n in names for r in ('PRM_native', 'PRM_Q80') if n != r and r in names]
    for rule in ('NC', 'Q80'):   # the separate claim "L-SML adds beyond the mean" (section 4); controls only
        pairs += [(f'CHAIN_N16_{rule}', f'{c}_{rule}') for c in ('HALF_N16', 'EQ_N16')
                  if f'CHAIN_N16_{rule}' in names and f'{c}_{rule}' in names]
    return pairs


def evaluate(lay, arms, gold, population, all_uids, boot, clusters=None, phantom=False):
    """Rows for one PRM and population. boot: run the bootstrap. clusters: {uid: question key} (singletons if None).
    phantom: the gold answers this PRM did not score enter the bootstrap as misses (avg_f1_failures_wrong)."""
    uids = [str(u) for u in lay['uids']]; lens = lay['nonempty_lengths'].astype(int)
    G = [gold[u] for u in uids]
    if any(g['n_steps'] != n for g, n in zip(G, lens)):
        raise SystemExit('step count differs from gold')
    labels = np.array([g['label'] for g in G]); subsets = np.array([g['subset'] for g in G])
    full = {arm: expand(lay, s) for arm, s in arms.items()}
    within = {arm: PM.within_auc_per_answer(labels, lens, full[arm]) for arm in arms}
    rows, sub_rows, stats = [], [], {}
    n_missing = len(all_uids) - len(uids)
    native_preds = None
    for name, (arm, flags, info) in decisions(lay, arms).items():
        preds = per_answer_preds(lens, flags)
        if name == 'PRM_native': native_preds = preds
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
                   n_empty_steps=int((~lay['nonempty'].astype(bool)).sum()),
                   rule='NC' if name.endswith('_NC') else 'Q80' if name.endswith('_Q80') else 'native',
                   n_nc_gate=info.get('n_gate'), n_nc_k0=info.get('n_k0'))
        rows.append(row)
        for s in PM.SUBSETS:
            if s in off:
                sub_rows.append(dict(population=population, arm=name, subset=s, **off[s],
                                     prism_tpr=off[s]['error_acc'], prism_tnr=off[s]['correct_acc'],
                                     prism_f1=off[s]['f1']))
        stats[name] = (arm, preds)
    for r in rows:   # answers whose predicted first error differs from PRM_native's
        r['n_pred_diff_vs_native'] = int((stats[r['arm']][1] != native_preds).sum())
    if not boot:
        return rows, sub_rows, []
    rng = np.random.default_rng(SEED)
    keep = set(uids); ph = [u for u in all_uids if u not in keep] if phantom else []
    lab_x = np.r_[labels, [gold[u]['label'] for u in ph]].astype(int)
    sub_x = np.r_[subsets, [gold[u]['subset'] for u in ph]]
    cl_x = np.array([u if clusters is None else clusters[u] for u in uids + ph], dtype=object)
    counts_x = PM.clustered_counts(sub_x, cl_x, rng, DRAWS); counts = counts_x[:, :len(uids)]
    step_answer = np.repeat(np.arange(len(uids)), lens); tg = PM.step_arrays(labels, lens); auroc_d = {}
    for t in ('prefix', 'first_only'):
        y, inc = tg[t]
        for arm in arms:
            k = inc & np.isfinite(full[arm])
            auroc_d[t, arm] = weighted_auc_draws(y[k], full[arm][k], step_answer[k], counts)
    f1_d = {n: avg_f1_counts(counts, labels, p, subsets) for n, (_, p) in stats.items()}
    fw_d = {n: avg_f1_counts(counts_x, lab_x, np.r_[p, np.full(len(ph), -2)], sub_x)   # -2: never a correct prediction
            for n, (_, p) in stats.items()} if phantom else {}
    wa_d = {arm: weighted_nanmean(counts, within[arm]) for arm in arms}
    point = {r['arm']: r for r in rows}; boot_rows = []
    for name, ref in contrast_pairs(list(stats)):
        arm, rarm = stats[name][0], stats[ref][0]
        def gap(k): return np.nan if None in (point[name][k], point[ref][k]) else point[name][k] - point[ref][k]
        mets = [('avg_f1', f1_d[name] - f1_d[ref], gap('avg_f1'))]
        if phantom: mets.append(('avg_f1_failures_wrong', fw_d[name] - fw_d[ref], gap('avg_f1_failures_wrong')))
        mets += [('within_auc', wa_d[arm] - wa_d[rarm], gap('within_auc')),
                 ('auroc_prefix', auroc_d['prefix', arm] - auroc_d['prefix', rarm], gap('auroc_prefix')),
                 ('auroc_first_only', auroc_d['first_only', arm] - auroc_d['first_only', rarm], gap('auroc_first_only'))]
        for mt, d, p in mets:
            boot_rows.append(dict(population=population, contrast=f'{name} - {ref}', metric=mt, delta=float(p),
                                  ci_low=float(np.quantile(d, .025)), ci_high=float(np.quantile(d, .975)),
                                  n_answers_drawn=len(uids) + len(ph), n_unscored_as_miss=len(ph)))
    return rows, sub_rows, boot_rows


def load_clusters(answers_path, gold):
    """{uid: whitespace-normalised question} for every gold answer, from the label-free answers file. If the file has
    none of the gold uids (a fixture with no question text) every answer is its own cluster, and the manifest says
    so; a partial match is an error."""
    A = json.loads(answers_path.read_text(encoding='utf8')) if answers_path.exists() else []
    q = {str(x['uid']): PM.norm_question(x['question']) for x in A}
    hit = [u for u in gold if u in q]
    if not hit:
        print(f'WARNING: no gold uid in {answers_path}; bootstrap clusters are single answers')
        return None, {'source': None, 'mode': 'singleton', 'n_clusters': len(gold)}
    if len(hit) < len(gold):
        raise SystemExit(f'{len(gold) - len(hit)} gold uids have no question in {answers_path}')
    keys = [q[u] for u in gold]
    return {u: q[u] for u in gold}, {'source': str(answers_path), 'sha256': sha(answers_path), 'mode': 'question',
                                     'n_clusters': len(set(keys)), 'n_answers_in_shared_questions':
                                     sum(c for c in collections.Counter(keys).values() if c > 1)}


def verdict_report(R, B, loaded):
    """Section 4: E1 / E2 per PRM, the overall verdict, the separate L-SML claim and the NC edge-case counts."""
    per = {m: PM.prm_success(B, m) for m in loaded}
    nc = {m: {k: r[k] for k in ('n_nc_gate', 'n_nc_k0', 'n_pred_diff_vs_native', 'avg_f1', 'avg_f1_failures_wrong',
                                'within_auc', 'n_answers', 'n_unscored')}
          for m in loaded for r in R if r['model'] == m and r['population'] == 'scored' and r['arm'] == PM.PRIMARY[0]}
    claim = {m: {r['contrast']: {k: r[k] for k in ('delta', 'ci_low', 'ci_high')} for r in B
                 if r['model'] == m and r['population'] == 'scored' and r['metric'] == 'within_auc'
                 and r['contrast'] in ('CHAIN_N16_NC - HALF_N16_NC', 'CHAIN_N16_NC - EQ_N16_NC')} for m in loaded}
    return {'registered': {'document': 'results/prm_boost_processbench_v1/PREREGISTRATION_HE.md',
                           'contrast': ' - '.join(PM.PRIMARY), 'population': 'scored',
                           'E1': 'within_auc CI low > 0', 'E2': f'avg_f1_failures_wrong CI low > {PM.E2_MARGIN}',
                           'bootstrap': f'paired, {DRAWS} draws, seed {SEED}, stratified by subset, clustered by question'},
            'per_prm': per, 'overall': PM.verdict(per), 'primary_arm_rows': nc,
            'lsml_claim_within_auc (not part of the verdict)': claim}


def write_csv(path, rows):
    if not rows: return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, keys); w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('scores', type=Path); ap.add_argument('out', type=Path)
    ap.add_argument('--gold', type=Path, default=Path(__file__).resolve().parent / 'processbench_gold.json')
    ap.add_argument('--answers', type=Path, default=Path(__file__).resolve().parents[1] / 'inputs/answers.json',
                    help='label-free answers with the question text (bootstrap clusters)')
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
    clusters, cluster_source = load_clusters(a.answers, gold)
    R, SR, B = [], [], []
    for m, (lay, arms) in loaded.items():
        pops = [('scored', lay, arms)]
        if len(common) < len(lay['uids']):
            pops.append(('common', *subset_view(lay, arms, common)))
        for pop, l2, a2 in pops:
            r, sr, b = evaluate(l2, a2, gold, pop, list(gold), not a.no_bootstrap, clusters, phantom=pop == 'scored')
            for x in r + sr + b: x['model'] = m
            R += r; SR += sr; B += b
        print(m, {r['arm']: round(r['avg_f1'], 2) for r in R if r['model'] == m and r['population'] == 'scored'})
    write_csv(a.out / 'RESULTS_processbench.csv', R)
    write_csv(a.out / 'SUBSETS_processbench.csv', SR)
    write_csv(a.out / 'BOOTSTRAP_processbench.csv', B)
    if not a.no_bootstrap:
        V = verdict_report(R, B, loaded)
        (a.out / 'VERDICT_processbench.json').write_text(json.dumps(V, indent=1))
        print('VERDICT', V['overall']['verdict'], V['overall']['successes'], 'missing', V['overall']['missing_prms'])
    (a.out / 'EVAL_MANIFEST.json').write_text(json.dumps({
        'gold_sha256': sha(a.gold), 'scores': {f.name: sha(f) for fs in models.values() for f in fs},
        'n_common': len(common), 'n_scored': {m: int(len(lay['uids'])) for m, (lay, _) in loaded.items()},
        'draws': 0 if a.no_bootstrap else DRAWS, 'seed': SEED, 'q': Q, 'clusters': cluster_source}, indent=1))


if __name__ == '__main__':
    main()
