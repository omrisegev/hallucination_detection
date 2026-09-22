"""Read-only evidence audit; writes only a new review artifact, never source results."""
from pathlib import Path
import csv
import hashlib
import json
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import test_renyi_locator_feature_bank_v1 as bank_test
from scripts import test_fusion_input_normalization_ablation_v1 as norm_test
from spectral_utils import renyi_locator_feature_bank as bank
from spectral_utils.math_gate_selection import percentile_by_cell


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    suites = ['fusion_input_normalization_ablation_v1',
              'renyi_locator_feature_bank_v1', 'renyi_locator_integrated_replay_v1']
    audit = {'reviewed_commit': 'cf01849a7', 'new_benchmark_experiment': False,
             'full_score_replay_performed': False, 'suites': {}, 'source_hashes': {}}
    for suite in suites:
        path = ROOT / 'results' / suite / 'METRICS.json'
        result = json.loads(path.read_text(encoding='utf8'))
        rows = result['metrics']
        issues = []
        for name, row in rows.items():
            cells = row['pb_cells']
            macro = float(np.mean([value['f1'] for value in cells.values()]))
            expected_suppressed = row['pb_exact_count'] - round(
                row['pb_error_exact_accuracy'] * row['pb_error_count'])
            if abs(macro - row['pb_all8']) > 1e-12 or expected_suppressed != row['pb_correct_peaks_suppressed']:
                issues.append({'method': name, 'headline_pb': row['pb_all8'],
                               'embedded_cell_macro': macro,
                               'expected_suppressed_from_headline': expected_suppressed,
                               'stored_suppressed': row['pb_correct_peaks_suppressed']})
        csv_path = path.with_name('COMPARISON.csv')
        compared, csv_mismatches = 0, []
        if csv_path.exists():
            for csv_row in csv.DictReader(csv_path.open(encoding='utf-8-sig')):
                name = csv_row['method']
                for key, value in csv_row.items():
                    if key in rows[name] and isinstance(rows[name][key], (int, float)):
                        try:
                            number = float(value)
                        except ValueError:
                            continue
                        compared += 1
                        if abs(number - rows[name][key]) > 1e-12:
                            csv_mismatches.append([name, key])
        audit['suites'][suite] = {'method_count': len(rows), 'inconsistent_pb_bundles': issues,
                                 'csv_numeric_fields_compared': compared,
                                 'csv_mismatches': csv_mismatches}
        audit['source_hashes'][str(path.relative_to(ROOT))] = digest(path)

    audit['existing_unit_checks'] = {'bank': bank_test.run(), 'normalization': norm_test.run()}
    rng = np.random.default_rng(20260915)
    matrix = rng.normal(size=(60, 7)) * np.arange(1, 8)
    spans = np.array([[0, 20], [20, 40], [40, 60]])
    spec = bank.BANK_BY_NAME['ve1q15__h10__hinf0']
    before, _ = bank.score_bank({'matrix': matrix}, spans, spec, 'raw_step_equal')
    permuted = matrix.copy()
    for lo, hi in spans:
        for j in range(7):
            permuted[lo:hi, j] = rng.permutation(matrix[lo:hi, j])
    after, _ = bank.score_bank({'matrix': permuted}, spans, spec, 'raw_step_equal')
    audit['within_step_independent_view_permutation'] = {
        'max_abs': float(np.max(np.abs(before - after))),
        'scope': 'fixed already-oriented views; scoring only, not re-estimating orientation'}
    # A deployment-contract demonstration, not a performance experiment.
    p1 = percentile_by_cell(np.array([1., 2., 3.]), np.array(['a'] * 3))[1]
    p2 = percentile_by_cell(np.array([1., 2., 3., 4., 5., 6., 7.]), np.array(['a'] * 7))[1]
    audit['gate_cohort_dependence_fixture'] = {
        'unchanged_raw_score': 2., 'percentile_initial': float(p1),
        'percentile_with_more_high_scores': float(p2),
        'decision_at_q033_initial': bool(p1 >= .33),
        'decision_at_q033_enlarged': bool(p2 >= .33)}

    raw_root = ROOT.parents[1]
    cache = raw_root / 'dataset_cache/repgrid/pb_qwen3_4b/processbench_gsm8k.pkl'
    manifest = json.loads((ROOT / 'results/renyi_position_temporal_fusion_v1/MANIFEST.json').read_text())
    expected_hash = next(value for key, value in manifest['hashes'].items()
                         if key.endswith('/dataset_cache/repgrid/pb_qwen3_4b/processbench_gsm8k.pkl'))
    actual_hash = digest(cache)
    if actual_hash != expected_hash:
        raise ValueError('audit cache does not match remote source manifest')
    with cache.open('rb') as handle:
        data = pickle.load(handle)
    delta, tokens = [], 0
    for row in data.values():
        lp = np.asarray(row['top_k_logprobs']['logprobs'], dtype=float)[:, :15]
        p = np.exp(lp)
        q = p / (p.sum(axis=1, keepdims=True) + 1e-12)
        h = -(q * np.log(q + 1e-12)).sum(axis=1)
        saved = np.asarray(row['token_entropies'], dtype=float)
        if saved.shape != h.shape:
            raise ValueError('entropy alignment mismatch')
        delta.extend(np.abs(saved - h).tolist())
        tokens += len(h)
    audit['native_entropy_cache_check'] = {
        'source': str(cache), 'bytes': cache.stat().st_size,
        'sha256': actual_hash, 'matches_remote_manifest': True,
        'answers': len(data), 'tokens': tokens,
        'max_abs_vs_top15_conditional_entropy': float(max(delta)),
        'mean_abs_vs_top15_conditional_entropy': float(np.mean(delta)),
        'interpretation': 'Native H1 is cached top15 conditional entropy, not full-vocabulary entropy.'}

    requirements = [
        'results/renyi_position_temporal_fusion_v1/CHECKPOINT.sqlite',
        'results/renyi_locator_feature_bank_v1/SCORES_FROZEN.npz',
        'results/renyi_locator_integrated_replay_v1/SCORES_FROZEN.npz',
        'results/selected_q15_finalist_replay_v1/SCORES_FROZEN.npz',
        'results/gate_feature_readout_selection_v1/DETECTORS_FROZEN.npz']
    availability = {}
    for rel in requirements:
        path = ROOT / rel
        state = 'missing'
        if path.is_file():
            with path.open('rb') as handle:
                state = 'lfs_pointer' if handle.read(80).startswith(b'version https://git-lfs') else 'present'
        availability[rel] = state
    audit['raw_replay_asset_availability'] = availability
    audit['manifest_paths'] = {key: manifest[key] for key in ('source_root', 'contract_root')}
    out = ROOT / 'results/remote_temporal_review_20260915/AUDIT.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + '\n', encoding='utf8', newline='\n')
    print(json.dumps({
        'audit': str(out), 'inconsistent_bundles': sum(len(v['inconsistent_pb_bundles']) for v in audit['suites'].values()),
        'native_entropy': audit['native_entropy_cache_check'],
        'units': audit['existing_unit_checks'], 'assets': availability,
        'permutation': audit['within_step_independent_view_permutation']}, indent=2))


if __name__ == '__main__':
    main()
