"""Before/after bridge for ct7_token_tail_lsml_calfix_v1 against the frozen ct7_token_tail_lsml_v1.
Old: four fit folds, q80 from other models' out-of-fold scores (quantile_0.8), inner label-selected
quantile (inner_selected), 'ct7' on its historical scale.  New: three fit folds, same-model calibration.

    python -B scripts/experiments/ct7_token_tail_calfix_before_after.py <run_dir>
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]; OLD = ROOT / 'results/ct7_token_tail_lsml_v1'
RENAME = {'ct7': 'ct7_raw', 'ct7_top10_equal7': 'readout_then_equal_ct7views'}


def main(run: Path):
    old = pd.read_csv(OLD / 'SUMMARY.csv').set_index('method'); ops = json.loads((OLD / 'PRMSCORE.json').read_text()); new = pd.read_csv(run / 'METRICS.csv').set_index('method')
    rows = []
    for m in old.index:
        n = RENAME.get(m, m); r = {'old_name': m, 'new_method': n, 'old_prmscore_q80_oof': ops[m]['quantile_0.8']['prmscore'], 'old_prmscore_inner_labelsel': ops[m]['inner_selected']['prmscore'],
                                   'old_within_auc': old.loc[m, 'within_auc'], 'old_pb_sla': old.loc[m, 'sla_macro8']}
        if n in new.index:
            x = new.loc[n]; r.update({'new_prmscore_P1_pooled': x.prmscore_P1, 'new_prmscore_P1b_prmbcal': x.prmscore_P1b, 'new_prmscore_P2_labelsel': x.prmscore_P2,
                                      'new_within_auc': x.within_auc, 'new_pb_sla': x.pb_sla_macro8})
        rows.append(r)
    if 'ct7' in new.index:
        x = new.loc['ct7']; rows.append({'old_name': '(none)', 'new_method': 'ct7', 'new_prmscore_P1_pooled': x.prmscore_P1, 'new_prmscore_P1b_prmbcal': x.prmscore_P1b,
                                         'new_prmscore_P2_labelsel': x.prmscore_P2, 'new_within_auc': x.within_auc, 'new_pb_sla': x.pb_sla_macro8})
    T = pd.DataFrame(rows); T.to_csv(run / 'BEFORE_AFTER.csv', index=False)
    pd.set_option('display.width', 250); print(T.round(4).to_string(index=False))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
