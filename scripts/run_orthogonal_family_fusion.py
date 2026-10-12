"""Frozen label-free within-family weighting; evaluate only after all scores sealed."""
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
import run_group_contrast_oracle as prior

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/orthogonal_family_fusion_20261005'
PREVIOUS = ROOT / 'results/contrast_span_oracle_20261005/run_v3'
spec = importlib.util.spec_from_file_location('orthogonal_family', ROOT / 'spectral_utils/orthogonal_family_fusion.py')
f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--output-name')
    args = parser.parse_args()
    out = BASE / (args.output_name or ('smoke' if args.smoke else 'run_v1'))
    if out.exists():
        raise RuntimeError('Preserve previous outputs; directory already exists')
    out.mkdir(parents=True)
    assert shutil.disk_usage(ROOT).free > 200_000_000
    seal = json.loads((PREVIOUS / 'INPUT_CODE_SEAL.json').read_text(encoding='utf8'))
    for section in [seal['inputs'], seal['code']]:
        for path, digest in section.items():
            assert infra.sha(path) == digest, path
    for path in [Path(__file__), ROOT / 'spectral_utils/orthogonal_family_fusion.py',
                 ROOT / 'tests/test_orthogonal_family_fusion.py', BASE / 'PROTOCOL.json',
                 ROOT / 'scripts/run_group_contrast_oracle.py', ROOT / 'spectral_utils/group_contrast_oracle.py']:
        seal['code'][str(path)] = infra.sha(path)
    for path in (ROOT / '.worktrees/cumulative-vote-fusion-v2/scripts/experiments/cvf_v2').rglob('*.py'):
        seal['code'][str(path)] = infra.sha(path)
    pop = infra.CC.Population()
    assert pop.n == 13769
    banks = ['B13'] if args.smoke else ['B13', 'B32']
    cells = ['pb_gsm8k_q4'] if args.smoke else sorted(set(pop.cells))
    complete = json.loads((PREVIOUS / 'COMPLETE.json').read_text(encoding='utf8'))
    for bk in banks:
        for cell in cells:
            path = PREVIOUS / (bk + '__' + cell + '.scores.npz')
            digest = infra.sha(path)
            assert digest == complete['artifacts'][path.name]
            seal['inputs'][str(path)] = digest
    infra.dump(out / 'INPUT_CODE_SEAL.json', seal)
    manifest = json.loads((ROOT / 'results/level_reduction_v1/run_20261003/INPUT_MANIFEST.json').read_text(encoding='utf8'))
    gates, failures, weights, metric_checks = [], [], [], []
    start = time.time()
    for bk in banks:
        data = infra.HistoricalData(manifest, pop, banks=[bk])
        for cell in cells:
            state = infra.restore_content(data, bk, cell, 'original')
            bank, x = state['bank'], state['X']
            names = [bank['names'][i] for i in state['surv']]
            assert not any('digit' in name for name in names)
            ng = len(np.unique(state['groups']))
            key = data.env['KEYG'][bank['rows'], :ng]
            print('FIT', bk, cell, x.shape, 'groups', ng, flush=True)
            scores, info, zs = f.fit_cell(x, bank['marks'][:, state['surv']], state['groups'], names,
                                          bank['off'], key, int(state['anchor']), bk + '|' + cell)
            replay = {}
            for new, old in [('RAW_EQ_EQ', 'MEAN_EQ'), ('RAW_EQ_MOM', 'MEAN_MOM'),
                             ('RAW_HEM_EQ', 'HEM_EQ'), ('RAW_HEM_MOM', 'HEM_MOM'),
                             ('COMMON_EQ', 'MEAN_EQ'), ('COMMON_MOM', 'MEAN_MOM')]:
                assert scores[new] is not None, (new, info['failures'])
                replay[new] = float(np.max(np.abs(scores[new] - state['scores'][old])))
                assert replay[new] <= 1e-9, (bk, cell, new, replay[new])
            previous = np.load(PREVIOUS / (bk + '__' + cell + '.scores.npz'))
            np.testing.assert_array_equal(previous['answers'], bank['answers'])
            np.testing.assert_array_equal(previous['offsets'], bank['off'])
            scores['plain'] = previous['plain']
            scores['LR_all_X'] = previous['LR_all_X']
            assert np.max(np.abs(scores['plain'] - state['scores']['DS_FILTER_EQ'])) <= 1e-10
            for item in info['failures']:
                failures.append(dict(bank=bk, cell=cell, **item))
            for record in info['records']:
                for group in record['details']['groups']:
                    weights.append(dict(bank=bk, cell=cell, representation=record['representation'], within=record['within'],
                                        **{k:v for k,v in group.items() if k != 'estimate'}))
            arrays = {name:score for name, score in scores.items() if score is not None}
            assert all(np.isfinite(s).all() for s in arrays.values())
            np.savez_compressed(out / (bk + '__' + cell + '.scores.npz'), answers=bank['answers'], offsets=bank['off'], **arrays)
            infra.dump(out / (bk + '__' + cell + '.fits.json'), info)
            gates.append(dict(bank=bk, cell=cell, n_answers=len(bank['answers']), n_steps=len(x),
                              n_channels=x.shape[1], n_groups=ng, reconstruction_error=info['basis']['reconstruction_error'],
                              replay_errors=replay, n_scored=len(arrays), n_failed=len(info['failures'])))
            print('SEALED CELL', bk, cell, len(arrays), 'scored', len(info['failures']), 'failed', round(time.time()-start,1), 's', flush=True)
        del data
    score_seal = {p.name:infra.sha(p) for p in out.glob('*.scores.npz')}
    infra.dump(out / 'SCORES_SEALED_BEFORE_EVALUATION.json', dict(scores=score_seal, n_cells=len(gates)))
    infra.dump(out / 'GATES.json', gates)
    infra.dump(out / 'FAILURES.json', failures)
    pd.DataFrame(weights).to_csv(out / 'WITHIN_WEIGHTS.csv', index=False)
    rows = []
    for bk in banks:
        for cell in cells:
            path = out / (bk + '__' + cell + '.scores.npz')
            assert infra.sha(path) == score_seal[path.name]
            pack = np.load(path)
            bank = dict(answers=pack['answers'], off=pack['offsets'])
            for method in pack.files:
                if method in ['answers', 'offsets']:
                    continue
                actual = infra.cs.per_answer(pack[method], bank, pop)
                independent = prior.g.independent_metrics(pack[method], bank, pop)
                for a, b in zip(actual, independent):
                    np.testing.assert_allclose(a, b, atol=1e-12, rtol=0, equal_nan=True)
                panel, values = ('PRMB', actual[0]) if cell.startswith('prm') else ('PB', actual[1])
                metric_checks.append(dict(bank=bk, cell=cell, method=method, n_checked=int(np.isfinite(values).sum())))
                for i, value in zip(bank['answers'], values):
                    rows.append(dict(bank=bk, cell=cell, panel=panel, method=method, answer=int(i),
                                     source_group=str(pop.groups[i]), value=float(value)))
            print('EVALUATED', bk, cell, flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'PER_ANSWER.csv', index=False)
    cell_summary = frame.groupby(['bank', 'cell', 'panel', 'method']).value.agg(['mean', 'count']).reset_index()
    cell_summary.to_csv(out / 'PER_CELL.csv', index=False)
    summary = cell_summary.groupby(['bank', 'panel', 'method']).agg(value=('mean','mean'), n_checked=('count','sum'), n_cells=('cell','nunique')).reset_index()
    summary['complete_population'] = [n == (1 if panel == 'PRMB' or args.smoke else 8) for panel,n in zip(summary.panel,summary.n_cells)]
    summary.loc[~summary.complete_population, 'value'] = np.nan
    summary.to_csv(out / 'SUMMARY.csv', index=False)
    if not args.smoke:
        pairs = []
        for within in ['EQ','SML','MOM','DS','HEM']:
            for between in ['EQ','MOM']:
                arm = f'ORTH_{within}_{between}'
                pairs += [(arm, f'RAW_{within}_{between}'), (arm, f'COMMON_{between}'), (arm, 'plain')]
        infra.cs.bootstrap_differences(frame, pairs, draws=5000, seed=20261007).to_csv(out / 'PAIRED_CONTRASTS.csv', index=False)
    for section in [seal['inputs'], seal['code']]:
        for path, digest in section.items():
            assert infra.sha(path) == digest, path
    infra.dump(out / 'REVIEW.json', dict(status='PASS', unit_tests=4, metric_checks=metric_checks,
                                        n_metric_rows=len(metric_checks), n_expected_failures=len(failures),
                                        review_scope='Same-agent independent rankdata versus pair-counting; no external review'))
    infra.dump(out / 'COMPLETE.json', dict(status='PASS', scope='FEASIBILITY' if args.smoke else 'FULL_DEVELOPMENT',
                                          runtime_s=time.time()-start, all_sources_unchanged=True, n_cells=len(gates),
                                          artifacts={p.name:infra.sha(p) for p in out.glob('*') if p.is_file() and p.name != 'COMPLETE.json'}))
    print(summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
