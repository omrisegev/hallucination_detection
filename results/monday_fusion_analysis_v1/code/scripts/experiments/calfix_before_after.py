"""Before/after bridge for family_tail_calfix_v1: every arm of the three superseded step runs,
old metrics (overwritten calibration array, PRMBench-only q80) beside the corrected ones.
P1b uses the same calibration pool as the old runs, so old -> P1b isolates the evaluation fix
(same-model calibration, no overwrite); P1b -> P1 is the change to the pooled external contract.

    python -B scripts/experiments/calfix_before_after.py <run_dir>
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MAP = {
    'named_group_fusion_v1': {'B11_equal': 'B11_equal', 'B11_lsml': 'B11_lsml', 'S3_M15_mean_equal': 'F15_equal', 'S3_M15_mean_lsml': 'F15_cov_lsml',
                              'S3_M15_mean_sml': 'F15_cov_sml', 'S3_M15_sml_equal': 'F15w_sml_equal', 'S3_M15_sml_lsml': 'F15w_sml_lsml', 'S3_M15_sml_sml': 'F15w_sml_sml',
                              'S4_M16_mean_equal': 'F16_equal', 'S4_M16_mean_lsml': 'F16_cov_lsml', 'S4_M16_mean_sml': 'F16_cov_sml', 'S4_M16_sml_equal': 'F16w_sml_equal',
                              'S4_M16_sml_lsml': 'F16w_sml_lsml', 'S4_M16_sml_sml': 'F16w_sml_sml', 'all48_equal': 'A48n_equal', 'all48_lsml': 'A48n_cov_lsml',
                              'all48_oriented_equal': 'A48o_equal', 'all48_oriented_lsml': 'A48o_cov_lsml', 'kept28_oriented_equal': 'K28_equal',
                              'kept28_oriented_lsml': 'K28_cov_lsml', 'ct7': 'ct7'},
    'tail_weighted_fusion_v1': {'B11_lsml': 'B11_lsml', 'S3_M15_cov_sml': 'F15_cov_sml', 'S3_M15_equal': 'F15_equal', 'S3_M15_tail1_lsml': 'F15_tail1_lsml',
                                'S3_M15_tail1_sml': 'F15_tail1_sml', 'S3_M15_tail20_lsml': 'F15_tail20_lsml', 'S3_M15_tail20_sml': 'F15_tail20_sml',
                                'S4_M16_cov_sml': 'F16_cov_sml', 'S4_M16_equal': 'F16_equal', 'S4_M16_tail1_lsml': 'F16_tail1_lsml', 'S4_M16_tail1_sml': 'F16_tail1_sml',
                                'S4_M16_tail20_lsml': 'F16_tail20_lsml', 'S4_M16_tail20_sml': 'F16_tail20_sml', 'all48_equal': 'A48n_equal',
                                'kept28_oriented_equal': 'K28_equal', 'ct7': 'ct7'},
    'tail_lsml_banks_v1': {'B11_cov_lsml': 'B11_lsml', 'B11_equal': 'B11_equal', 'B11_oriented_cov_lsml': 'B11o_cov_lsml', 'B11_oriented_equal': 'B11o_equal',
                           'B11_oriented_tail1_lsml': 'B11o_tail1_lsml', 'B11_oriented_tail20_lsml': 'B11o_tail20_lsml', 'B11_tail1_lsml': 'B11_tail1_lsml',
                           'B11_tail20_lsml': 'B11_tail20_lsml', 'CT7s_block421': 'CT7s_block421', 'CT7s_cov_lsml': 'CT7s_cov_lsml', 'CT7s_equal': 'CT7s_equal',
                           'CT7s_tail1_lsml': 'CT7s_tail1_lsml', 'CT7s_tail20_lsml': 'CT7s_tail20_lsml', 'ct7': 'ct7'}}


def main(run: Path):
    new = pd.read_csv(run / 'METRICS.csv').set_index('method'); rows = []
    for stage, mp in MAP.items():
        old = pd.read_csv(ROOT / 'results' / stage / 'run_20260924' / 'METRICS.csv').pivot(index='method', columns='metric', values='estimate')
        for o, m in mp.items():
            r = {'old_stage': stage, 'old_name': o, 'new_method': m, 'old_prmscore_prmbcal': old.loc[o, 'prmscore'], 'old_within_auc': old.loc[o, 'prm_within_auc'], 'old_pb_sla': old.loc[o, 'pb_sla_macro8']}
            if m in new.index:
                n = new.loc[m]
                r.update({'new_prmscore_P1b_prmbcal': n.prmscore_P1b, 'new_prmscore_P1_pooled': n.prmscore_P1, 'new_prmscore_P2_labelsel': n.prmscore_P2,
                          'new_within_auc': n.within_auc, 'new_pb_sla': n.pb_sla_macro8})
                r['delta_prmscore_evalfix'] = r['new_prmscore_P1b_prmbcal'] - r['old_prmscore_prmbcal']
                r['delta_within_auc'] = r['new_within_auc'] - r['old_within_auc']
            rows.append(r)
    T = pd.DataFrame(rows); T.to_csv(run / 'BEFORE_AFTER.csv', index=False)
    pd.set_option('display.width', 250); print(T.drop(columns=['old_stage']).round(4).to_string(index=False))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
