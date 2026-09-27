"""Before/after verification for the 2026-09-27 fold-role fix of six runners that wrote each fold-k model's scores
into one array for both its evaluation fold k and its calibration fold (k+1)%5 (LESSONS 2026-09-27):
bank20_lsml_run.py, indbank_lsml_run.py, declared_joint_run.py (Steps 438-440), error_cluster_lsml_run.py,
core_virtual_lsml_run.py (Step 441) and partition_ceiling_run.py (Step 447-series partition ceiling).

Mechanism (confirmed by an independent toy replay): with k = 0..4 in order, the last write to fold 0 came from the
fold-4 model; folds 1-4 kept their own model. The threshold of fold k was read from the rows of fold (k+1)%5 of that
array: for k = 0..3 the fold-(k+1) model's evaluation scores, for k = 4 the fold-4 model's calibration scores on fold 0.

Checks per stage (old run beside the new run_20260927_calfix):
  1. identical input hashes; identical fit / selection records (weights as logged, rounded to 4-5 decimals);
  2. when the old run saved step scores: evaluation scores on folds 1-4 identical for every method; on fold 0 the old
     scores equal the new fold-4 model's calibration scores (CAL_SCORES.npz) bit for bit, i.e. the old array is
     rebuilt exactly from the new per-fold outputs;
  3. when the old run saved thresholds: old thresholds recomputed from the old array, new thresholds recomputed from
     CAL_SCORES, both equal to what the runs saved;
  4. fixed-weight arms (equal, declared_equal, energy_level_alone, ct7) must not change at all; the B11 fit-on-4 replay
     must keep its scores (its threshold now comes from a fold its own model was fitted on - label-free, disclosed);
  5. metric and contrast tables before/after, flagging any sign flip or change in whether the 95% interval excludes 0.

    python -B scripts/experiments/old_runners_calfix_verify.py [stage ...]
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
NEW = 'run_20260927_calfix'
STAGES = {'bank20_lsml_prmbench_v1': 'run_20260924', 'indbank_lsml_prmbench_v1': 'run_20260924',
          'declared_joint_prmbench_v1': 'run_20260924', 'error_cluster_lsml_v1': 'run_20260924',
          'core_virtual_lsml_v1': 'run_20260924', 'partition_ceiling_prmbench_v1': 'run_20260927'}
METRIC = {'within_auc': 'within_auc', 'prm_within_auc': 'within_auc', 'prmscore_answer_z_q80': 'prmscore', 'prmscore': 'prmscore',
          'sla': 'pb_sla', 'pb_sla_macro8': 'pb_sla'}


def is_fixed(m):
    return m in ('ct7', 'equal', 'energy_level_alone') or (re.search(r'_equal$', m) is not None and not m.endswith('group_equal'))


def zt(s):
    return (s - s.mean()) / max(s.std(), 1e-8)


def records(d):
    for name in ('FIT_MANIFEST.jsonl', 'SELECTION_LOG.jsonl'):
        if (d / name).exists():
            return name, [json.loads(line) for line in open(d / name, encoding='utf8')]
    return None, None


def metrics(d):
    M = pd.read_csv(d / 'METRICS.csv')
    if 'stratum' in M: M = M[M.stratum.isin(['all', 'macro8_context'])]
    M = M[M.metric.isin(METRIC)].assign(metric=lambda x: x.metric.map(METRIC))
    return M.pivot_table(index='method', columns='metric', values='estimate', aggfunc='first')


def contrasts(d):
    C = pd.read_csv(d / 'CONTRASTS.csv').rename(columns={'contrast': 'contrast_id'})
    dup = C[C.duplicated(['contrast_id', 'endpoint'], keep=False)]
    if len(dup):   # declared_joint_run lists some primary contrasts twice; they must be identical before one is dropped
        num = dup.drop(columns=['primary'], errors='ignore').groupby(['contrast_id', 'endpoint']).nunique(dropna=False)
        assert (num <= 1).all().all(), f'non-identical duplicate contrasts in {d}'
    return C.drop_duplicates(['contrast_id', 'endpoint'])


def verify(stage, old_run):
    o, n = ROOT / 'results' / stage / old_run, ROOT / 'results' / stage / NEW
    out, fail = {'stage': stage, 'old_run': old_run}, []
    im_o = json.loads((o / 'INPUT_MANIFEST.json').read_text(encoding='utf8'))
    im_n = json.loads((n / 'INPUT_MANIFEST.json').read_text(encoding='utf8'))
    keys = [k for k, v in im_o.items() if isinstance(v, dict) and 'sha256' in v]
    out['inputs_identical'] = all(im_o[k]['sha256'] == im_n[k]['sha256'] for k in keys); out['inputs_checked'] = len(keys)
    rn, ro = records(o); _, rw = records(n)
    out['fit_records_identical'] = ro == rw; out['fit_records_checked'] = f'{len(ro)} ({rn})'
    fail += [c for c in ('inputs_identical', 'fit_records_identical') if not out[c]]

    ans = pd.read_csv(im_o['oof_answers']['path'], encoding='utf-8-sig')
    folds, prm = ans.fold.to_numpy(), ~ans.cell.str.startswith('pb_').to_numpy()
    sn, cn = np.load(n / 'STEP_SCORES.npz'), np.load(n / 'CAL_SCORES.npz')
    off = sn['offsets']; step_fold = np.repeat(folds, np.diff(off))

    def q80(arr, cal):
        return float(np.quantile(np.concatenate([zt(arr[off[i]:off[i + 1]]) for i in np.flatnonzero(prm & (folds == cal))]), .8))

    methods = [m for m in sn.files if m != 'offsets']
    if (o / 'STEP_SCORES.npz').exists():
        so = np.load(o / 'STEP_SCORES.npz'); assert np.array_equal(off, so['offsets'])
        assert set(methods) == set(so.files) - {'offsets'}, 'method sets differ'
        rows = []
        for m in methods:
            a, b = so[m], sn[m]
            r = {'method': m, 'fixed_arm': is_fixed(m), 'max_diff_folds1_4': float(np.max(np.abs(a[step_fold > 0] - b[step_fold > 0]))),
                 'max_diff_fold0': float(np.max(np.abs(a[step_fold == 0] - b[step_fold == 0])))}
            if f'{m}__fold4' in cn.files:   # the fold-4 model's calibration fold is fold 0
                r['old_fold0_minus_new_fold4_model'] = float(np.max(np.abs(a[step_fold == 0] - cn[f'{m}__fold4'][step_fold == 0])))
            r['note'] = 'fit-on-4 replay: same scores, threshold now from its own model (in-sample fold, label-free)' if m.endswith('replay4') else ''
            rows.append(r)
        S = pd.DataFrame(rows); S.to_csv(n / 'SCORE_DIFF_BY_FOLD.csv', index=False)
        out['folds1_4_identical_all_methods'] = bool((S.max_diff_folds1_4 == 0).all())
        changed = S[~(S.max_diff_fold0 == 0)]   # NaN counts as changed
        out['methods_changed_on_fold0'] = changed.method.tolist()
        # the fit-on-4 replay wrote only evaluation rows in the old code too, so the fold-0 rebuild does not apply to it
        dbl = S[~S.method.str.endswith('replay4') & S.old_fold0_minus_new_fold4_model.notna()]
        out['old_array_rebuilt_exactly'] = bool((dbl.old_fold0_minus_new_fold4_model == 0).all()); out['rebuilt_methods'] = len(dbl)
        out['fixed_arms_scores_identical'] = bool((S[S.fixed_arm | S.method.str.endswith('replay4')][['max_diff_folds1_4', 'max_diff_fold0']] == 0).all().all())
        fail += [c for c in ('folds1_4_identical_all_methods', 'old_array_rebuilt_exactly', 'fixed_arms_scores_identical') if not out[c]]
        diag_o = json.loads((o / 'DIAGNOSTICS.json').read_text(encoding='utf8')) if (o / 'DIAGNOSTICS.json').exists() else {}
        diag_n = json.loads((n / 'DIAGNOSTICS.json').read_text(encoding='utf8')) if (n / 'DIAGNOSTICS.json').exists() else {}
        if 'calibration_thresholds' in diag_o:
            T = []
            for m in methods:
                for k in range(5):
                    cal = (k + 1) % 5
                    new_src = sn[m] if m == 'ct7' else cn[f'{m}__fold{k}']
                    T.append({'method': m, 'fold': k, 'old_saved': diag_o['calibration_thresholds'][m][str(k)], 'old_recomputed_from_old_array': q80(so[m], cal),
                              'new_saved': diag_n['calibration_thresholds'][m][str(k)], 'new_recomputed_from_same_model': q80(new_src, cal)})
            T = pd.DataFrame(T); T.to_csv(n / 'THRESHOLDS_BEFORE_AFTER.csv', index=False)
            out['thresholds_checked'] = len(T)
            out['old_thresholds_read_from_overwritten_array'] = bool(np.allclose(T.old_saved, T.old_recomputed_from_old_array, rtol=0, atol=1e-12))
            out['new_thresholds_from_same_model'] = bool(np.allclose(T.new_saved, T.new_recomputed_from_same_model, rtol=0, atol=1e-12))
            out['thresholds_changed'] = int((T.old_saved != T.new_saved).sum())
            fail += [c for c in ('old_thresholds_read_from_overwritten_array', 'new_thresholds_from_same_model') if not out[c]]
    else:
        out['score_level_checks'] = 'old run saved no step scores; metric and contrast level only'

    mo, mn = metrics(o), metrics(n)
    T = mo.join(mn, lsuffix='_old', rsuffix='_new', how='outer')
    for c in mo.columns: T[f'{c}_delta'] = T[f'{c}_new'] - T[f'{c}_old']
    T.index.name = 'method'; T.to_csv(n / 'BEFORE_AFTER.csv')
    dcols = [c for c in T.columns if c.endswith('_delta')]
    fixed = T[[is_fixed(m) for m in T.index]]
    out['fixed_arms_metrics_identical'] = bool((fixed[dcols].abs() < 1e-12).all().all())
    fail += [] if out['fixed_arms_metrics_identical'] else ['fixed_arms_metrics_identical']
    out['max_abs_delta'] = {c: float(T[c].abs().max()) for c in dcols}

    co, cw = contrasts(o), contrasts(n)
    C = co.merge(cw, on=['contrast_id', 'endpoint'], suffixes=('_old', '_new'), how='outer')
    excl = lambda lo, hi: np.where(lo > 0, 'above0', np.where(hi < 0, 'below0', 'includes0'))
    C['ci_old'] = excl(C.ci95_lo_old, C.ci95_hi_old); C['ci_new'] = excl(C.ci95_lo_new, C.ci95_hi_new)
    if 'ci_adj_lo_old' in C:
        C['adj_old'] = np.where(C.ci_adj_lo_old.notna(), excl(C.ci_adj_lo_old, C.ci_adj_hi_old), '')
        C['adj_new'] = np.where(C.ci_adj_lo_new.notna(), excl(C.ci_adj_lo_new, C.ci_adj_hi_new), '')
    else:
        C['adj_old'] = C['adj_new'] = ''
    C['sign_flip'] = np.sign(C.delta_old) != np.sign(C.delta_new)
    C['verdict_changed'] = (C.ci_old != C.ci_new) | (C.adj_old != C.adj_new) | C.sign_flip
    keep = ['contrast_id', 'endpoint'] + (['primary_old'] if 'primary_old' in C else []) + ['delta_old', 'delta_new', 'ci95_lo_new', 'ci95_hi_new',
                                                                                          'ci_old', 'ci_new', 'adj_old', 'adj_new', 'sign_flip', 'verdict_changed']
    C[keep].to_csv(n / 'CONTRASTS_BEFORE_AFTER.csv', index=False)
    out['contrasts'] = int(len(C)); out['contrast_verdicts_changed'] = C.loc[C.verdict_changed, ['contrast_id', 'endpoint']].values.tolist()
    out['failed_checks'] = fail; out['pass'] = not fail
    (n / 'CALFIX_VERIFY.json').write_text(json.dumps(out, indent=1), encoding='utf8')
    return out, T, C


if __name__ == '__main__':
    pd.set_option('display.width', 250)
    ok = True
    for st in (sys.argv[1:] or list(STAGES)):
        out, T, C = verify(st, STAGES[st])
        print(json.dumps(out, indent=1)); print(T.round(4).to_string())
        ok &= out['pass']
    print('ALL CHECKS PASS' if ok else 'CHECK FAILED')
    sys.exit(0 if ok else 1)
