"""Run the registered CPU-only contrast-span diagnostic on the complete roster."""
import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '2'
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/contrast_span_oracle_20261005'
CODE = ROOT / 'results/monday_fusion_analysis_v1/code'
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / 'scripts/experiments'))
spec = importlib.util.spec_from_file_location('contrast_span', ROOT / 'spectral_utils/contrast_span_oracle.py')
cs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cs)
from spectral_utils.monday_fusion_replay import HistoricalData
from spectral_utils.monday_fusion_execution import restore_content
from spectral_utils.monday_fusion_lr import training_labels
from spectral_utils.monday_fusion_protocol import RUNS
import calfix_common as CC


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False,
                              default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist()) + '\n', encoding='utf8')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    out = OUT / 'smoke_v3' if args.smoke else OUT / 'run_v3'
    out.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(ROOT).free < 150_000_000:
        raise RuntimeError('Insufficient free space')
    print('Python', sys.executable, 'free_GB', round(shutil.disk_usage(ROOT).free / 1e9, 2), flush=True)
    manifest_path = ROOT / 'results/level_reduction_v1/run_20261003/INPUT_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    pop = CC.Population()
    banks = ['B13'] if args.smoke else ['B13', 'B32']
    cells = ['pb_gsm8k_q4'] if args.smoke else sorted(set(pop.cells))
    sources = {str(manifest_path): sha(manifest_path)}
    for rec in manifest['inputs'].values():
        path = Path(rec['path'])
        actual = sha(path)
        expected = rec.get('sha256')
        if expected and actual != expected:
            raise AssertionError('Input hash mismatch: ' + str(path))
        sources[str(path)] = actual
    for path in pop.inputs.values():
        sources[str(path)] = sha(path)
    for bank in banks:
        for cell in cells:
            directory = RUNS['level_reduction_v1'] / 'chunks'
            stem = cell + '__' + bank + '__original'
            for suffix in ['.diagnostics.json', '.stats.npz', '.complete.json']:
                path = directory / (stem + suffix)
                sources[str(path)] = sha(path)
    code_files = [Path(__file__), ROOT / 'spectral_utils/contrast_span_oracle.py', OUT / 'PROTOCOL.json', OUT / 'PROTOCOL_AMENDMENT_V2.json', OUT / 'NUMERICAL_REFINEMENT_V3.json']
    code_files += list(CODE.rglob('*.py'))
    code_files += list((ROOT / 'spectral_utils/external_generalization').rglob('*.py'))
    initial = {str(p): sha(p) for p in code_files}
    dump(out / 'INPUT_CODE_SEAL.json', {'inputs': sources, 'code': initial, 'scope': 'FEASIBILITY' if args.smoke else 'FULL_DEVELOPMENT'})
    records, fits, gates, availability = [], [], [], []
    started = time.time()
    for bk in banks:
        data = HistoricalData(manifest, pop, banks=[bk])
        for cell in cells:
            stem = bk + '__' + cell
            state = restore_content(data, bk, cell, 'original')
            bank = state['bank']
            x, off = state['X'], bank['off']
            names = [bank['names'][i] for i in state['surv']]
            assert not any('digit' in n for n in names)
            missing = [name for name in ['realized_drv', 'chosen_surprisal'] if name not in names]
            availability.append(dict(bank=bk, cell=cell, single_contrast_available=not missing,
                                     missing=missing, actual_names=names))
            dump(out / 'AVAILABILITY.json', availability)
            y = training_labels(pop, bank)
            folds = np.repeat(pop.fold[bank['answers']], np.diff(off))
            old = np.load(RUNS['level_reduction_v1'] / 'chunks' / (cell + '__' + bk + '__original.stats.npz'))
            err = float(np.max(np.abs(x.mean(axis=1) - old['DS_FILTER_EQ__step_score'])))
            if err > 1e-10:
                raise AssertionError('Baseline replay mismatch')
            scores = {'plain': x.mean(axis=1)}
            for fold in range(5):
                te, pred, info = cs.fold_fit(x, y, folds, fold, state['groups'], names, off)
                for method, value in pred.items():
                    if method not in scores:
                        scores[method] = np.full(len(x), np.nan)
                    scores[method][te] = value
                info.update(bank=bk, cell=cell, names=names)
                fits.append(info)
                print(stem, 'fold', fold, 'complete', round(time.time() - started, 1), 's', flush=True)
            for method, value in scores.items():
                assert np.isfinite(value).all()
                auc, hit = cs.per_answer(value, bank, pop)
                panel, metric = ('PRMB', auc) if cell.startswith('prm') else ('PB', hit)
                for j, i in enumerate(bank['answers']):
                    records.append(dict(bank=bk, cell=cell, panel=panel, method=method,
                                        answer=int(i), uid=str(pop.uids[i]), source_group=str(pop.groups[i]),
                                        fold=int(pop.fold[i]), value=float(metric[j])))
            np.savez_compressed(out / (stem + '.scores.npz'), answers=bank['answers'], offsets=off, **scores)
            gates.append(dict(bank=bk, cell=cell, n_answers=len(bank['answers']), n_steps=len(x),
                              n_channels=x.shape[1], baseline_max_error=err,
                              n_unknown_post_error=int((y < 0).sum())))
            pd.DataFrame(records).to_csv(out / 'PER_ANSWER.csv', index=False)
            dump(out / 'FITS.json', fits)
            dump(out / 'GATES.json', gates)
        del data
    frame = pd.DataFrame(records)
    per_cell = frame.groupby(['bank', 'cell', 'panel', 'method']).value.agg(['mean', 'count']).reset_index()
    per_cell.to_csv(out / 'PER_CELL.csv', index=False)
    summary = per_cell.groupby(['bank', 'panel', 'method']).agg(value=('mean', 'mean'), n_checked=('count', 'sum'), n_cells=('cell', 'nunique')).reset_index()
    summary.to_csv(out / 'SUMMARY.csv', index=False)
    if not args.smoke:
        methods = ['LR_PC1_single_linear', 'LR_PC1_within_group_linear', 'LR_PC1_single_answer_std', 'LR_PC1_within_group_answer_std', 'LR_all_X']
        pairs = [(m, r) for m in methods for r in ['plain', 'LR_PC1']]
        pairs += [('LR_PC1_within_group_linear', 'LR_PC1_single_linear'), ('LR_PC1_within_group_answer_std', 'LR_PC1_single_answer_std')]
        pairs += [('LR_all_X', m) for m in methods[:-1]]
        cs.bootstrap_differences(frame, pairs).to_csv(out / 'PAIRED_CONTRASTS.csv', index=False)
        historical = pd.read_csv(ROOT / 'results/mom_filter_within_diagnostic_20261004/CONTRAST_STRUCT_RESULTS.csv')
        replay = []
        for bk in banks:
            for panel, field in [('PRMB', 'PRMB_within_auc'), ('PB', 'PB_argmax')]:
                for ours, old in [('plain', 'PLAIN'), ('LR_legacy_replay', 'LR_OOF')]:
                    actual = float(summary[(summary.bank == bk) & (summary.panel == panel) & (summary.method == ours)].value.iloc[0])
                    expected = float(historical[(historical.bank == bk) & (historical.variant == old)][field].iloc[0])
                    difference = abs(actual - expected)
                    replay.append(dict(bank=bk, panel=panel, method=ours, actual=actual, expected=expected, error=difference))
                    if difference > 1e-9:
                        raise AssertionError('Historical metric replay mismatch: ' + str(replay[-1]))
        dump(out / 'HISTORICAL_REPLAY.json', replay)
    changed = [p for p, digest in initial.items() if sha(p) != digest]
    changed += [p for p, digest in sources.items() if sha(p) != digest]
    if changed:
        raise AssertionError('Inputs/code changed during run: ' + str(changed))
    dump(out / 'COMPLETE.json', {'status': 'PASS', 'n_cells': len(gates), 'n_banks': len(banks),
                                'runtime_s': time.time() - started, 'all_inputs_code_unchanged': True,
                                'scope': 'FEASIBILITY' if args.smoke else 'FULL_DEVELOPMENT',
                                'artifacts': {p.name: sha(p) for p in out.glob('*') if p.is_file() and p.name != 'COMPLETE.json'}})
    print(summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
