"""Export completed temporal metrics with readable English method names."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/direct_probability_temporal_v3'
REPRESENTATIONS = {
    'current': 'Current token (17 inputs)',
    'lag8': 'Eight ordered tokens (136 inputs)',
    'delta': 'Current token + change (34 inputs)',
    'shuffled_lag8': 'Current token + shuffled history (control)',
}
SOLVERS = {
    'equal': 'Equal weights', 'iu': 'IU-PCR',
    'diag_lw': 'Diagonal shrinkage',
    'joint_lw': 'Joint-inspired shrinkage (singleton groups)',
    'time_then_rank_iu': 'IU-PCR: time then probability rank',
    'rank_then_time_iu': 'IU-PCR: probability rank then time',
    'chain_liu': 'LIU-PCR with chronological chain',
    'permuted_chain_liu': 'LIU-PCR with permuted chain (control)',
}


def main():
    result = json.loads((OUT / 'METRICS.json').read_text(encoding='utf8'))
    rows = []
    for key, metrics in result['metrics'].items():
        if key.startswith('saved_v2__'):
            continue  # Numerical replay duplicates, not extra candidates.
        if key == 'entropy':
            name = 'Token entropy (reference)'
        else:
            representation, solver = key.split('__')
            name = REPRESENTATIONS[representation] + ' / ' + SOLVERS[solver]
        row = dict(method=name, method_id=key, scope='Full cached localization development',
                   processbench_macro_f1_percent=100 * metrics['pb_all8'],
                   processbench_qwen4b_f1_percent=100 * metrics['pb_q4'],
                   processbench_qwen8b_f1_percent=100 * metrics['pb_q8'],
                   prmbench_within_answer_auc=metrics['prm_within'],
                   prmbench_pooled_auc=metrics['prm_pooled'],
                   prmscore=metrics['prmscore_q08'],
                   valid_answers=metrics['valid_answers'],
                   prmbench_within_answer_denominator=metrics['prm_within_n'])
        for cell, cell_metrics in metrics['pb_cells'].items():
            row[cell + '_f1_percent'] = 100 * cell_metrics['f1']
        rows.append(row)
    with (OUT / 'SUMMARY_READABLE.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Exported {len(rows)} candidates/references, including all eight ProcessBench cells.')


if __name__ == '__main__':
    main()
