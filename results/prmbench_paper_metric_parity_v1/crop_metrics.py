"""CROP (arXiv 2605.30085) Appendix Table 6, PRMBench row, recomputed for our arms on their protocol.

Their protocol (paper Appendix B + github.com/matthewyccheung/crop, read 2026-10-18):
  * traces: the FIRST 5,000 rows of hitsmy/PRMBench_Preview (split train) in dataset order, using
    modified_process; no shuffling, no dedup (crop/scripts/import_external_process.py, --prmbench_limit 5000).
    Table 4: 5,000 traces, 67,445 steps, 15.9% error steps.
  * labels: step t is an error iff (t+1) in error_steps (1-based -> 0-based, values < 1 clamped to 0).
  * splits: 10 seeds 2806..2815, trace-level 60/20/20 (train / calibration / test), stratified by domain and
    by has-error (crop/splits.py: rng = default_rng(seed); shuffle within strata).
  * learned proxies: sklearn pipeline median-impute + StandardScaler + LogisticRegression(L2, C=1, lbfgs,
    max_iter=500, class_weight='balanced') fitted on the train split's per-step labels.
  * grid: fixed lambda in {0, 0.01, ..., 1.00} (101 points) for the additional datasets incl. PRMBench.
  * CROP: largest lambda with (1 + #contaminated calibration traces) / (n_cal + 1) <= alpha (alpha = 0.05);
    retained prefix = steps before the first score > lambda; infeasible -> smallest lambda (their code).
  * metrics on the test split: step AUROC pooled over all test steps (per-step error label); prefix kept =
    mean over test traces of M_i / T_i; prefix risk = share of test traces whose kept prefix has an error.
    Reported as mean +- 1.96 * sd / sqrt(10) over splits (their caption: 95% normal intervals).

Our reproduction, and every place it differs:
  * trace subset: first 5,000 non-control rows of our loader (dataset_cache prmbench_prm.pkl key order, which
    is the Hub order). Our loader dropped 5 duplicate multi_solutions rows that CROP keeps, so the subset is
    APPROXIMATE (67,465 steps vs their 67,445; error rate 15.88% vs 15.9%).
  * unbounded scores (LSML_CONT, CHAIN_*, position/length is a probability so not affected) cannot use the
    fixed [0,1] grid; they use 201 empirical quantiles of the TRAIN-split step scores (CROP's own target-domain
    grid rule, label-free). Direct PRM is run on both grids as a bridge.
  * our fused arms are label-free and were fitted ONCE on all 6,969 PRMBench answers' steps (transductive,
    no labels); they do not use the 60% train split at all. CROP's threshold uses calibration LABELS for
    every arm (that is CROP's access level, not ours).
  * position/length baseline = LR on step index, total steps, relative position, step token length, prefix
    token length, answer token length (Qwen3-8B token counts from population.npz). CROP's token/format LR also
    uses character/digit/punctuation counts of the step text; we have no local step text, so it is a SUBSET.
  * TELE13+PRM LR = supervised analog of their Trace+PRM row on our own features (PRM risk + 13 digit-free
    channels). Not their trace features (Llama-3.1-8B likelihood/hidden-state summaries).
"""
import csv, json
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from common import HERE, load_population, get_scores, rewards_by_id, N13, within_auc

SEEDS = list(range(2806, 2816)); ALPHA = .05; N_TRACES = 5000


def split_traces(has_err, seed):
    rng = np.random.default_rng(seed)
    tr, ca, te = [], [], []
    for key in (False, True):
        idx = np.flatnonzero(has_err == key).tolist(); rng.shuffle(idx)
        n = len(idx); n_tr = int(round(.6 * n)); n_ca = int(round(.2 * n))
        tr += idx[:n_tr]; ca += idx[n_tr:n_tr + n_ca]; te += idx[n_tr + n_ca:]
    for part in (tr, ca, te): rng.shuffle(part)
    return np.array(tr), np.array(ca), np.array(te)


def prefix_len(s, lam):
    hit = np.flatnonzero(s > lam)
    return int(hit[0]) if len(hit) else len(s)


def crop(scores, labels, cal, test, grid):
    # contamination loss per calibration trace for every lambda (vectorized via each trace's running max)
    def first_err(y): f = np.flatnonzero(y); return f[0] if len(f) else None
    losses = np.zeros((len(grid), len(cal)))
    for k, i in enumerate(cal):
        fe = first_err(labels[i])
        if fe is None: continue
        need = np.max(scores[i][:fe + 1])         # prefix contains the first error iff lambda >= max score up to it
        losses[:, k] = grid >= need
    risk = (1 + losses.sum(1)) / (len(cal) + 1)
    ok = np.flatnonzero(risk <= ALPHA)
    lam = grid[ok[np.argmax(grid[ok])]] if len(ok) else grid.min()
    M = np.array([prefix_len(scores[i], lam) for i in test]); T = np.array([len(scores[i]) for i in test])
    cont = np.array([bool(m > 0 and labels[i][:m].any()) for m, i in zip(M, test)])
    return dict(kept=100 * float(np.mean(M / np.maximum(T, 1))), kept_stepweighted=100 * M.sum() / T.sum(),
                risk=100 * float(cont.mean()), lam=float(lam), feasible=bool(len(ok)))


def main():
    P = load_population(); S = get_scores(P)
    off = P['off']; pos = {i: j for j, i in enumerate(P['ids'])}
    rew, order, klass = rewards_by_id()
    sub = [i for i in order if klass[i] != 'correct'][:N_TRACES]
    J = np.array([pos[i] for i in sub])
    meta = {r['idx']: r for r in json.loads(open(HERE.parents[1] / 'results/advisor_completion_2026-10-11/inputs/prmbench_meta.json').read())['rows']}
    sl = [slice(off[j], off[j + 1]) for j in J]
    labels = []
    for i, s in zip(sub, sl):                       # CROP's label rule (independent of population y)
        es = meta[i]['error_steps']; es = json.loads(es) if isinstance(es, str) else es
        e0 = {max(int(e) - 1, 0) for e in es}; n = s.stop - s.start
        labels.append(np.array([t in e0 for t in range(n)]))
    n_lab_diff = sum(int((labels[k] != P['y'][sl[k]]).sum()) for k in range(len(sub)))
    has_err = np.array([len(json.loads(meta[i]['error_steps']) if isinstance(meta[i]['error_steps'], str) else meta[i]['error_steps']) > 0 for i in sub])
    steps = int(sum(s.stop - s.start for s in sl)); err = int(sum(l.sum() for l in labels))
    for i, s in zip(sub, sl):
        assert len(rew[i]) == s.stop - s.start
    # per-step features for the learned proxies
    names = P['names']; X = P['X']
    step_idx = np.concatenate([np.arange(1, b - a + 1) for a, b in zip(off[:-1], off[1:])]).astype(float)
    tot = np.repeat(np.diff(off), np.diff(off)).astype(float)
    pref = np.concatenate([np.r_[0, np.cumsum(P['tok'][a:b])[:-1]] for a, b in zip(off[:-1], off[1:])])
    F_pos = np.column_stack([step_idx, tot, step_idx / tot, P['tok'], pref, np.repeat(P['atok'], np.diff(off))])
    F_tele = np.column_stack([P['prm']] + [X[:, names.index(c)] for c in N13])
    fixed = np.linspace(0, 1, 101)
    arms = {}
    rng0 = np.random.default_rng(0)
    for seed in SEEDS:
        tr, ca, te = split_traces(has_err, seed)
        rows_tr = np.concatenate([np.arange(sl[k].start, sl[k].stop) for k in tr])
        ytr = np.concatenate([labels[k] for k in tr])
        def per_trace(v): return [v[sl[k]] for k in range(len(sub))]
        def qgrid(v): return np.quantile(v[rows_tr], np.linspace(0, 1, 201))
        cand = {}
        rr = np.random.default_rng(seed); cand['Random'] = ([rr.random(s.stop - s.start) for s in sl], fixed, 'fixed101')
        risk1 = [1 - rew[i] for i in sub]
        cand['Direct PRM (1-reward)'] = (risk1, fixed, 'fixed101')
        cand['Direct PRM (1-reward), quantile grid'] = (risk1, np.quantile(np.concatenate([risk1[k] for k in tr]), np.linspace(0, 1, 201)), 'train-quantile201')
        for nm, F in (('Position/length LR (supervised)', F_pos), ('TELE13+PRM LR (supervised)', F_tele)):
            mdl = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                                LogisticRegression(C=1.0, solver='lbfgs', max_iter=500, class_weight='balanced', random_state=seed))
            mdl.fit(F[rows_tr], ytr)
            p = np.zeros(len(F)); allrows = np.concatenate([np.arange(s.start, s.stop) for s in sl])
            p[allrows] = mdl.predict_proba(F[allrows])[:, 1]
            cand[nm] = (per_trace(p), fixed, 'fixed101')
        for a in ('LSML_CONT', 'CHAIN_N16', 'CHAIN_N13'):
            cand[a] = (per_trace(S[a]), qgrid(S[a]), 'train-quantile201')
        cand['PRM_native (logit risk), quantile grid'] = (per_trace(S['PRM_native']), qgrid(S['PRM_native']), 'train-quantile201')
        yte = np.concatenate([labels[k] for k in te])
        for nm, (sc, grid, gname) in cand.items():
            r = crop(sc, labels, ca, te, grid)
            r['auroc'] = roc_auc_score(yte, np.concatenate([sc[k] for k in te]))
            r['grid'] = gname
            arms.setdefault(nm, []).append(r)
    out = {'subset': {'n_traces': len(sub), 'n_steps': steps, 'error_rate': err / steps,
                      'n_label_cells_differing_from_population_y': n_lab_diff,
                      'n_has_error_traces': int(has_err.sum())}, 'arms': {}}
    for nm, rs in arms.items():
        agg = {}
        for k in ('auroc', 'kept', 'kept_stepweighted', 'risk', 'lam'):
            v = np.array([r[k] for r in rs]); agg[k] = float(v.mean()); agg[k + '_ci95'] = float(1.96 * v.std(ddof=1) / np.sqrt(len(v)))
        agg['n_infeasible'] = int(sum(not r['feasible'] for r in rs)); agg['grid'] = rs[0]['grid']
        out['arms'][nm] = agg
    # within-answer AUC on the same 5,000 traces (our usual metric) for reference
    msk = np.zeros(len(P['ids']), bool); msk[J] = True
    out['within_auc_subset'] = {a: within_auc(P['y'], S[a], off, msk)[0] for a in S}
    json.dump(out, open(HERE / 'CROP_RESULTS.json', 'w'), indent=1)
    print(json.dumps(out['subset']))
    for nm, a in out['arms'].items():
        print(f"{nm:42s} AUROC {a['auroc']:.4f}+-{a['auroc_ci95']:.4f}  kept {a['kept']:.1f}+-{a['kept_ci95']:.1f}  risk {a['risk']:.1f}+-{a['risk_ci95']:.1f}  infeas {a['n_infeasible']}")
    print(out['within_auc_subset'])


if __name__ == '__main__':
    main()
