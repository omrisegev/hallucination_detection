"""Supervised LR transfer: PRMBench N16 (labels) -> Socratic-PRMBench and Hard2Verify (no target labels).

Recipe mirrors results/advisor_completion_2026-10-11/lr (branch codex/advisor-v11-completion):
  * features = bank N16 columns, per-answer z-scored (digit channels z-scored over their active
    positions only), exactly prepare_population.py's adapter;
  * spectral_utils.advisor_lr.fit_lr: StandardScaler + balanced LogisticRegression (lbfgs);
  * C chosen from C_GRID by mean class-balanced log-loss over the 5 FOLDS_V2 PRMBench source-group
    folds (training data only), then refit on all training steps.
Training population: PRMBench noncontrol answers (classification != 'correct'), all their steps.
Scoring: decision function (higher = error) on each external N16 step; ANSWER_Z_Q80 decisions,
the formula of monday_cont_baselines_v1/score_ext.py. Evaluation = the frozen external evaluator.
Check first: reproduce the ALL_EQ / N16 / ANSWER_Z_Q80 external rows of external_nested.csv.
"""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '4'
import hashlib, json, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parent
MAIN = OUT.parents[1]
SSL = MAIN / '.worktrees/ssl-pseudolabel-residual-v1'
sys.path.insert(0, str(SSL))
from spectral_utils.external_generalization.evaluation import confusion, within_auc  # noqa: E402
from spectral_utils.family_external_metrics import summary  # noqa: E402

INP = MAIN / 'scratch/external_lr_prm_v1/inputs'
GOLD = MAIN / 'scratch/external_generalization_private/inputs/evaluator_only'
META = MAIN / 'results/thinkfix_rescore_v1/thinkfix_mask1/prmscore_inputs/prmbench_meta.json'
BANK8 = MAIN / 'results/thinkfix_rescore_v1/prmscore_inputs/bank_N16.npz'  # = Drive thinkfix_mask1/prmscore_inputs (10900709 B)
GROUPS = MAIN / 'results/localization_source_group_audit_v1/CANONICAL_GROUPS.json'
FOLDS = MAIN / 'results/localization_source_group_audit_v1/FOLDS_V2.json'
N16 = json.loads((MAIN / 'results/monday_nested_banks_v2/BANKS.json').read_text())['N16']
DIG = ['digit_alternative', 'digit_spread', 'digit_alternative_innovation']
C_GRID = (.001, .01, .1, 1., 10., 100., 1000.)
CELLS = {'socratic': 'socratic_qwen3_8b', 'hard2verify': 'hard2verify_qwen3_8b'}
REF = MAIN / 'results/thinkfix_rescore_4b_v1/share_qwen3_4b_vs_8b/qwen3_4b_vs_8b__external_nested.csv'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def answer_std(x, off):
    out = np.zeros_like(x, dtype=float)
    for a, b in zip(off[:-1], off[1:]):
        v = x[a:b]; sd = v.std(0)
        out[a:b] = np.divide(v - v.mean(0), sd, out=np.zeros_like(v), where=sd > 1e-12)
    return out


def bank_V(F, answers):
    """prepare_population.py adapter on a subset of answers (all answers if None)."""
    nm = [str(c) for c in F['names']]; X = F['values'].astype(float); fo = F['offsets'].astype(np.int64)
    da = F['digit_active'].astype(bool)
    Vall = answer_std(X, fo); vd = np.zeros((len(X), 3))
    for a, b in zip(fo[:-1], fo[1:]):
        for j, c in enumerate(DIG):
            act = da[a:b, j]; v = X[a:b, nm.index(c)][act]
            if len(v) and v.std() > 1e-12: vd[a:b, j][act] = (v - v.mean()) / v.std()
    if answers is None: answers = np.arange(len(fo) - 1)
    rows = np.concatenate([np.arange(fo[a], fo[a + 1]) for a in answers])
    V = np.column_stack([vd[rows, DIG.index(c)] if c in DIG else Vall[rows, nm.index(c)] for c in N16])
    off = np.r_[0, np.cumsum(np.diff(fo)[answers])]
    assert np.isfinite(V).all()
    return V, off


def fit_lr(X, y, C):
    sc = StandardScaler().fit(X)
    est = LogisticRegression(C=float(C), class_weight='balanced', solver='lbfgs', max_iter=3000, random_state=20261003)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        est.fit(sc.transform(X), y)
    w = est.coef_[0] / sc.scale_; b = float(est.intercept_[0] - sc.mean_ @ w)
    assert np.max(np.abs(X @ w + b - est.decision_function(sc.transform(X)))) < 1e-9
    return w, b, est.coef_[0], int(est.n_iter_[0])


def balanced_log_loss(y, s):
    cnt = np.bincount(y, minlength=2); loss = np.logaddexp(0., s) - y * s
    return float(np.mean(loss * (len(y) / (2. * cnt[y]))))


def az_q80(s, off):
    az = np.zeros_like(s)
    for a, b in zip(off[:-1], off[1:]):
        v = s[a:b]; az[a:b] = (v - v.mean()) / max(v.std(), 1e-8)
    return az >= np.quantile(az, .8)


def evaluate(bench, F, score, flag):
    """Same code path as monday_cont_baselines_v1/evaluate_ext.py."""
    uids = [str(u) for u in F['uids']]; lens = F['nonempty_lengths']; ne = F['nonempty'].astype(bool)
    full_s = np.full(int(lens.sum()), np.nan); full_s[np.flatnonzero(ne)] = score
    full_p = np.zeros(int(lens.sum()), int); full_p[np.flatnonzero(ne)] = (~flag).astype(int)
    starts = np.r_[0, np.cumsum(lens)]
    gold = {g['uid']: g for g in json.loads((GOLD / f'{bench}.json').read_text(encoding='utf8'))}
    counts, aucs = [], []
    for i, u in enumerate(uids):
        g = gold[u]; a, b = starts[i], starts[i + 1]; inc = np.asarray(g['include'], bool)
        counts.append(confusion(g['correct'], full_p[a:b], inc))
        v = within_auc(g['correct'], full_s[a:b], inc & ne[a:b]); aucs.append(np.nan if v is None else v)
    return summary(np.array(counts), bench)['metric'], float(np.nanmean(aucs)), len(uids)


def prmbench_train(F, meta_rows, groups, folds, include_controls):
    meta = {r['idx']: r for r in meta_rows}
    rid = [str(u) for u in F['row_ids']]; cells = np.asarray(F['cells']).astype(str)
    pos = {u: i for i, u in enumerate(rid)}
    idx = [r['idx'] for r in meta_rows]
    keep = [u for u in idx if include_controls or meta[u]['classification'] != 'correct']
    ans = np.array([pos[u] for u in keep])
    assert set(cells[ans]) == {'prmbench_qwen3_8b'} or len(set(cells[ans])) == 1, set(cells[ans])
    V, off = bank_V(F, ans)
    y = np.zeros(off[-1], int)
    for i, u in enumerate(keep):
        n = off[i + 1] - off[i]; assert n == meta[u]['n_steps'] if 'n_steps' in meta[u] else True
        y[off[i]:off[i + 1]] = np.isin(np.arange(1, n + 1), meta[u]['error_steps'])
    fold = np.repeat(np.array([folds[groups[u]] for u in keep]), np.diff(off))
    return V, y, off, fold, keep, ans


def within_prmb(score, y, off):
    out = []
    for a, b in zip(off[:-1], off[1:]):
        yy = y[a:b].astype(bool)
        if yy.any() and (~yy).any():
            from scipy.stats import rankdata
            r = rankdata(score[a:b]); out.append((r[yy].sum() - yy.sum() * (yy.sum() + 1) / 2) / (yy.sum() * (~yy).sum()))
    return float(np.mean(out)), len(out)


def main():
    t0 = time.time(); rows, log = [], {'inputs': {}}
    meta_rows = json.loads(META.read_text())['rows']
    groups = {r['row_id']: r['canonical_group_id'] for r in json.loads(GROUPS.read_text())['rows']}
    folds = json.loads(FOLDS.read_text())['prmbench']['outer']
    ref = pd.read_csv(REF)
    for p in sorted(INP.glob('*.npz')): log['inputs'][p.name] = sha(p)

    # ---- Check 1: PRMBench 8B adapter parity with the frozen ALL_EQ export.
    F8 = np.load(INP / 'src8b_FEATURES.npz', allow_pickle=True)
    b8 = np.load(BANK8, allow_pickle=True)
    rid = [str(u) for u in F8['row_ids']]; pos = {u: i for i, u in enumerate(rid)}
    V, off = bank_V(F8, np.array([pos[str(u)] for u in b8['idx']]))
    assert np.array_equal(off, b8['offsets'])
    err = float(np.max(np.abs(answer_std(V.mean(1)[:, None], off).ravel() - b8['S__ALL_EQ'])))
    log['check_prmbench_ALL_EQ_max_abs_8b'] = err; print('PRMBench 8B ALL_EQ parity', err, flush=True)
    assert err < 1e-9

    for model, src in (('8B', 'src8b'), ('4B', 'src4b')):
        F = F8 if model == '8B' else np.load(INP / f'{src}_FEATURES.npz', allow_pickle=True)
        ext = {b: np.load(INP / f'{c}_{model.lower()}_FEATURES.npz', allow_pickle=True) for b, c in CELLS.items()}
        Vext = {b: bank_V(E, None) for b, E in ext.items()}
        # ---- Check 2: reproduce ALL_EQ N16 ANSWER_Z_Q80 external rows.
        for bench, E in ext.items():
            Ve, oe = Vext[bench]; s = Ve.mean(1)
            m, a, n = evaluate(bench, E, s, az_q80(s, oe))
            r = ref[(ref.cell == CELLS[bench]) & (ref.bank == 'N16') & (ref.scorer == 'ALL_EQ') & (ref.rule == 'ANSWER_Z_Q80')].iloc[0]
            rm, ra = r[f'official_metric_{model.lower()}'], r[f'within_auc_{model.lower()}']
            log[f'check_ALL_EQ_{bench}_{model}'] = {'ours': [m, a], 'reference': [rm, ra], 'abs_diff': [abs(m - rm), abs(a - ra)]}
            print(f'check ALL_EQ {bench} {model}: ours {m:.6f}/{a:.6f} ref {rm:.6f}/{ra:.6f}', flush=True)
            assert abs(m - rm) < 1e-9 and abs(a - ra) < 1e-9
            rows.append(dict(benchmark=bench, model=model, method='ALL_EQ (plain average, reproduction check)', rule='ANSWER_Z_Q80',
                             official_metric=m, within_auc=a, n_answers=n))
        for include_controls in (False, True):
            tag = 'noncontrol' if not include_controls else 'all6969'
            X, y, offp, fold, keep, _ = prmbench_train(F, meta_rows, groups, folds, include_controls)
            losses = []
            for C in C_GRID:
                fl = []
                for k in range(5):
                    tr, va = fold != k, fold == k
                    w, b, _, _ = fit_lr(X[tr], y[tr], C); fl.append(balanced_log_loss(y[va], X[va] @ w + b))
                losses.append(float(np.mean(fl)))
            best = min(range(len(C_GRID)), key=lambda i: (losses[i], C_GRID[i])); C = C_GRID[best]
            w, b, coef_std, nit = fit_lr(X, y, C)
            # in-sample PRMBench within-AUC (training data; reference only)
            wa, na = within_prmb(X @ w + b, y, offp)
            log[f'fit_{model}_{tag}'] = {'n_answers': len(keep), 'n_steps': int(len(y)), 'pos_rate': float(y.mean()),
                                         'C_losses': dict(zip(map(str, C_GRID), losses)), 'C': C, 'n_iter': nit,
                                         'columns': N16, 'weight_on_answer_z': w.tolist(), 'intercept': b,
                                         'coef_standardized': coef_std.tolist(), 'prmbench_in_sample_within_auc': wa}
            print(model, tag, 'C', C, 'in-sample PRMB within', round(wa, 4), flush=True)
            for bench, E in ext.items():
                Ve, oe = Vext[bench]; s = Ve @ w + b
                m, a, n = evaluate(bench, E, s, az_q80(s, oe))
                rows.append(dict(benchmark=bench, model=model, method=f'LR transfer (trained on PRMBench {tag} steps)',
                                 rule='ANSWER_Z_Q80', official_metric=m, within_auc=a, n_answers=n))
                print(f'  {bench}: official {m:.4f} within {a:.4f}', flush=True)
                np.save(OUT / 'scores' / f'LR_{tag}_{model}_{bench}.npy', s) if (OUT / 'scores').exists() else None
    log['seconds'] = round(time.time() - t0, 1)
    pd.DataFrame(rows).to_csv(OUT / 'LR_TRANSFER.csv', index=False)
    (OUT / 'LR_TRANSFER_LOG.json').write_text(json.dumps(log, indent=1), encoding='utf8')
    print('done', log['seconds'], 's')


if __name__ == '__main__':
    main()
