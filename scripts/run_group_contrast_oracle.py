"""Full matched group-level oracle; historical results are read-only."""
import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '2'
import sys
sys.dont_write_bytecode = True
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
import run_contrast_span_oracle as infra

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/group_contrast_oracle_20261005'
PREVIOUS = ROOT / 'results/contrast_span_oracle_20261005/run_v3'
spec = importlib.util.spec_from_file_location('group_oracle', ROOT / 'spectral_utils/group_contrast_oracle.py')
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
import ds_group_weights as GW
from spectral_utils.lsml_gate_locator_research import answer_standardize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    out = BASE / ('smoke' if args.smoke else 'run_v1')
    if out.exists():
        raise RuntimeError('Output directory already exists; preserve it and choose a new registered run')
    out.mkdir(parents=True)
    if shutil.disk_usage(ROOT).free < 200_000_000:
        raise RuntimeError('Insufficient disk space')
    print('Free GB', round(shutil.disk_usage(ROOT).free / 1e9, 2), flush=True)
    old_seal = json.loads((PREVIOUS / 'INPUT_CODE_SEAL.json').read_text(encoding='utf8'))
    for section in ['inputs', 'code']:
        for path, digest in old_seal[section].items():
            if infra.sha(path) != digest:
                raise AssertionError('Prior source/code changed: ' + path)
    sources = dict(old_seal['inputs'])
    code = dict(old_seal['code'])
    for path in [Path(__file__), ROOT / 'spectral_utils/group_contrast_oracle.py',
                 ROOT / 'tests/test_group_contrast_oracle.py', BASE / 'PROTOCOL.json']:
        code[str(path)] = infra.sha(path)
    pop = infra.CC.Population()
    assert pop.n == 13769
    banks = ['B13'] if args.smoke else ['B13', 'B32']
    cells = ['pb_gsm8k_q4'] if args.smoke else sorted(set(pop.cells))
    old_complete = json.loads((PREVIOUS / 'COMPLETE.json').read_text(encoding='utf8'))
    for bk in banks:
        for cell in cells:
            path = PREVIOUS / (bk + '__' + cell + '.scores.npz')
            digest = infra.sha(path)
            assert digest == old_complete['artifacts'][path.name]
            sources[str(path)] = digest
    infra.dump(out / 'INPUT_CODE_SEAL.json', {'inputs': sources, 'code': code})
    manifest = json.loads((ROOT / 'results/level_reduction_v1/run_20261003/INPUT_MANIFEST.json').read_text(encoding='utf8'))
    rows, fits, checks, metrics_checks = [], [], [], []
    start = time.time()
    for bk in banks:
        data = infra.HistoricalData(manifest, pop, banks=[bk])
        for cell in cells:
            state = infra.restore_content(data, bk, cell, 'original')
            bank, x = state['bank'], state['X']
            groups = GW.canon(state['groups'])
            z = state['recipes']['MEAN_EQ']['Z']
            names = [bank['names'][i] for i in state['surv']]
            assert not any('digit' in name for name in names)
            rebuilt_z = answer_standardize(x @ g.group_means_matrix(groups), bank['off'])
            z_error = float(np.max(np.abs(z - rebuilt_z)))
            assert z_error < 1e-10
            for method in ['MEAN_DS', 'MEAN_SML', 'MEAN_MOM']:
                np.testing.assert_array_equal(z, state['recipes'][method]['Z'])
            old_stats = np.load(infra.RUNS['level_reduction_v1'] / 'chunks' / (cell + '__' + bk + '__original.stats.npz'))
            scores = {'plain': state['scores']['DS_FILTER_EQ']}
            baseline_errors = {}
            for method in ['MEAN_EQ', 'MEAN_DS', 'MEAN_SML', 'MEAN_MOM']:
                scores[method] = state['scores'][method]
                baseline_errors[method] = float(np.max(np.abs(scores[method] - old_stats[method + '__step_score'])))
                assert baseline_errors[method] <= 1e-10
            previous = np.load(PREVIOUS / (bk + '__' + cell + '.scores.npz'))
            np.testing.assert_array_equal(previous['answers'], bank['answers'])
            np.testing.assert_array_equal(previous['offsets'], bank['off'])
            assert np.max(np.abs(previous['plain'] - scores['plain'])) <= 1e-10
            for method in ['LR_all_X', 'LR_PC1_within_group_linear']:
                scores[method] = previous[method]
            y = infra.training_labels(pop, bank)
            folds = np.repeat(pop.fold[bank['answers']], np.diff(bank['off']))
            for fold in range(5):
                te, predicted, info = g.fold_fit(x, z, y, folds, fold, groups)
                error = float(np.max(np.abs(predicted['LR_RAW_GROUPS_PLUS_WITHIN'] - scores['LR_all_X'][te])))
                assert error <= 1e-5, ('Complete feature-span replay', bk, cell, fold, error)
                for method, values in predicted.items():
                    if method not in scores:
                        scores[method] = np.full(len(x), np.nan)
                    scores[method][te] = values
                info.update(bank=bk, cell=cell, full_feature_replay_error=error,
                            group_members=[[names[i] for i in np.flatnonzero(groups == h)] for h in range(z.shape[1])])
                fits.append(info)
            for method, score in scores.items():
                assert np.isfinite(score).all()
                actual = g.cs.per_answer(score, bank, pop)
                independent = g.independent_metrics(score, bank, pop)
                for aa, bb in zip(actual, independent):
                    np.testing.assert_allclose(aa, bb, atol=1e-12, rtol=0, equal_nan=True)
                panel, values = ('PRMB', actual[0]) if cell.startswith('prm') else ('PB', actual[1])
                metrics_checks.append(dict(bank=bk, cell=cell, method=method, n_checked=int(np.isfinite(values).sum())))
                for i, value in zip(bank['answers'], values):
                    rows.append(dict(bank=bk, cell=cell, panel=panel, method=method, answer=int(i),
                                     uid=str(pop.uids[i]), source_group=str(pop.groups[i]), fold=int(pop.fold[i]), value=float(value)))
            np.savez_compressed(out / (bk + '__' + cell + '.scores.npz'), answers=bank['answers'], offsets=bank['off'], **scores)
            checks.append(dict(bank=bk, cell=cell, n_answers=len(bank['answers']), n_steps=len(x),
                               n_channels=x.shape[1], n_groups=z.shape[1], group_rebuild_error=z_error,
                               baseline_errors=baseline_errors, n_unknown_post_error=int((y < 0).sum())))
            pd.DataFrame(rows).to_csv(out / 'PER_ANSWER.csv', index=False)
            infra.dump(out / 'FITS.json', fits)
            infra.dump(out / 'GATES.json', checks)
            print(bk, cell, 'complete and independently checked', z.shape[1], 'groups,', round(time.time() - start, 1), 's', flush=True)
        del data
    frame = pd.DataFrame(rows)
    cell_summary = frame.groupby(['bank', 'cell', 'panel', 'method']).value.agg(['mean', 'count']).reset_index()
    cell_summary.to_csv(out / 'PER_CELL.csv', index=False)
    summary = cell_summary.groupby(['bank', 'panel', 'method']).agg(value=('mean', 'mean'), n_checked=('count', 'sum'), n_cells=('cell', 'nunique')).reset_index()
    summary.to_csv(out / 'SUMMARY.csv', index=False)
    if not args.smoke:
        pairs = [('LR_GROUPS', b) for b in ['plain', 'MEAN_EQ', 'MEAN_DS', 'MEAN_SML', 'MEAN_MOM']]
        pairs += [('LR_GROUPS', 'LR_GROUPS_NONNEG'), ('LR_GROUPS_PLUS_WITHIN', 'LR_GROUPS')]
        pairs += [('LR_all_X', b) for b in ['LR_GROUPS', 'LR_GROUPS_PLUS_WITHIN', 'LR_RAW_GROUP_SPAN']]
        pairs += [('LR_RAW_GROUP_SPAN', 'LR_GROUPS'), ('LR_GROUPS_NONNEG', 'MEAN_EQ')]
        g.cs.bootstrap_differences(frame, pairs, draws=5000, seed=20261006).to_csv(out / 'PAIRED_CONTRASTS.csv', index=False)
    coef_rows = []
    for info in fits:
        for fit in info['fits']:
            if fit['method'] in ['LR_GROUPS', 'LR_GROUPS_NONNEG']:
                for j, weight in enumerate(fit['coef']):
                    coef_rows.append(dict(bank=info['bank'], cell=info['cell'], fold=info['fold'], method=fit['method'],
                                          group=j, weight=weight, members=' | '.join(info['group_members'][j])))
    pd.DataFrame(coef_rows).to_csv(out / 'GROUP_WEIGHTS.csv', index=False)
    for section in [sources, code]:
        changed = [path for path, digest in section.items() if infra.sha(path) != digest]
        assert not changed, changed
    infra.dump(out / 'REVIEW.json', dict(status='PASS', n_unit_tests=4, n_cells=len(checks), n_fold_fits=len(fits),
                                        metric_checks=metrics_checks, n_metric_rows=len(metrics_checks),
                                        max_rotation_error=max(i['rotation_error'] for i in fits),
                                        max_full_feature_replay_error=max(i['full_feature_replay_error'] for i in fits),
                                        review_scope='Same-agent independent rankdata versus pair-counting metrics; no external review'))
    infra.dump(out / 'COMPLETE.json', dict(status='PASS', scope='FEASIBILITY' if args.smoke else 'FULL_DEVELOPMENT',
                                           runtime_s=time.time() - start, n_cells=len(checks), all_sources_unchanged=True,
                                           artifacts={p.name: infra.sha(p) for p in out.glob('*') if p.is_file() and p.name != 'COMPLETE.json'}))
    print(summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
