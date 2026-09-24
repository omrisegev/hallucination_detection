"""Evaluation step of family_tail_calfix_v1 (labels enter here only). Called by
family_tail_calfix_run.py after the bundle is sealed, or rerun on an existing run directory:

    python -B scripts/experiments/family_tail_calfix_eval.py <run_dir>

Declared alias: the bundle stores the frozen bank11 L-SML under its rule name 'B11_cov_lsml';
the protocol calls it 'B11_lsml'. The alias renames it at evaluation; no score changes.
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from calfix_common import MAIN, Population, dump  # noqa: E402
import calfix_before_after as BA  # noqa: E402
import calfix_evaluate as EV  # noqa: E402

RENAME = {'B11_cov_lsml': 'B11_lsml'}
TL = ['B11_lsml', 'F15_tailtie_lsml', 'F15_cov_lsml', 'K28_cov_lsml', 'K28_equal', 'F15_equal', 'B11_equal', 'B11_partition_equal']
DESC = [('A48o_equal', 'A48n_equal', 'orientation under equal'), ('A48o_cov_lsml', 'A48n_cov_lsml', 'orientation under L-SML'),
        ('F15_tailtie_lsml', 'F15_tail20_lsml', 'tie-aware vs historical marks'), ('F15_tail20_lsml', 'F15_cov_lsml', 'historical tail vs covariance'),
        ('F15_tail20_lsml', 'F15_equal', 'historical tail vs equal'), ('K28_tailtie_lsml', 'K28_cov_lsml', 'tail vs covariance on 28'),
        ('K28_tailtie_lsml', 'K28_equal', 'tail vs equal on 28'), ('A48o_tailtie_lsml', 'A48o_cov_lsml', 'tail vs covariance on 48'),
        ('A48o_tailtie_lsml', 'A48o_equal', 'tail vs equal on 48'), ('A48o_cov_lsml', 'A48o_equal', 'learned vs equal on 48'),
        ('B11_tailtie_lsml', 'B11_lsml', 'tail vs covariance on bank11'), ('F15_tailtie_lsml', 'K28_tailtie_lsml', 'family layer under tail L-SML'),
        ('K28_tailtie_lsml', 'A48o_tailtie_lsml', 'filtering under tail L-SML'),
        ('K28if_equal', 'K28_equal', 'in-fold vs retrospective selection (equal 28)'), ('K28if_cov_lsml', 'K28_cov_lsml', 'in-fold vs retrospective (L-SML 28)'),
        ('F15if_equal', 'F15_equal', 'in-fold vs retrospective (families equal)'), ('F15if_tailtie_lsml', 'F15_tailtie_lsml', 'in-fold vs retrospective (candidate)'),
        ('F15if_tailtie_lsml', 'B11_lsml', 'in-fold candidate vs anchor'), ('A48if_equal', 'A48o_equal', 'in-fold vs retrospective orientation'),
        ('F15if_cov_lsml', 'F15if_equal', 'in-fold: L-SML vs equal on families'), ('F15if_tailtie_lsml', 'F15if_equal', 'in-fold: tail vs equal on families'),
        ('K28if_equal', 'A48if_equal', 'in-fold filtering under equal')] + [(a, 'ct7', 'vs CT7 reference') for a in TL] + [('ct7', 'ct7_raw', 'CT7 normalization')]


def evaluate_run(OUT: Path, pop, P: dict, status: dict, T0: float, n_failures: int = 0) -> dict:
    PRIMARY = [tuple(x) for x in P['primary_contrasts_P1_prmscore_holm13']]
    res = EV.evaluate(OUT, OUT, PRIMARY, DESC, pop=pop, rename=RENAME)
    # frozen external source validation bridge: same fits, thresholds and PRMScore for the bank11 trio and CT7
    src = MAIN / 'results/lsml_external_generalization_v1/evaluation/source'
    cm = json.loads((src / 'VALIDATION_METRICS.json').read_text())['arms']; codex = json.loads((src / 'VALIDATION_FITS.json').read_text())
    thr = json.loads((OUT / 'THRESHOLDS.json').read_text())
    bridge2 = {a: {'codex_prmscore': cm[c]['prmscore'], 'ours_P1': res['point'][a]['prmscore_P1'], 'codex_within_auc': cm[c]['within_auc'], 'ours_within_auc': res['point'][a]['within_auc'],
                   'max_threshold_diff': max(abs(r['thresholds'][c] - thr[a]['P1'][r['test']]['tau']) for r in codex)}
               for a, c in (('B11_lsml', 'frozen_lsml'), ('B11_equal', 'frozen_equal'), ('B11_partition_equal', 'frozen_partition_equal'), ('ct7', 'ct7'))}
    bridge2['status'] = 'PASS' if all(abs(v['codex_prmscore'] - v['ours_P1']) < 1e-9 and abs(v['codex_within_auc'] - v['ours_within_auc']) < 1e-9 and v['max_threshold_diff'] < 1e-9 for v in bridge2.values()) else 'FAIL'
    dump(OUT / 'BRIDGE_EXTERNAL_SOURCE.json', bridge2)
    status.update({'status': 'COMPLETE' if bridge2['status'] == 'PASS' else 'BRIDGE_FAIL', 'finished': datetime.now().isoformat(timespec='seconds'), 'failures': n_failures,
                   'aliases': RENAME, 'seconds': time.perf_counter() - T0})
    dump(OUT / 'RUN_STATUS.json', status)
    BA.main(OUT)
    pd.set_option('display.width', 250)
    MT = pd.read_csv(OUT / 'METRICS.csv'); print(MT[['method', 'prmscore_P1', 'prmscore_P1b', 'prmscore_P2', 'within_auc', 'pb_sla_macro8']].round(4).to_string(index=False))
    CT = res['contrasts']; print(CT[(CT.family == 'primary') & (CT.endpoint == 'prmscore_P1')].round(4).to_string(index=False))
    print(json.dumps(bridge2, indent=1)); print(json.dumps(status))
    return res


if __name__ == '__main__':
    out = Path(sys.argv[1]); st = json.loads((out / 'RUN_STATUS.json').read_text(encoding='utf8'))
    st['evaluation_rerun'] = datetime.now().isoformat(timespec='seconds')
    evaluate_run(out, Population(), json.loads((ROOT / 'results/family_tail_calfix_v1/PROTOCOL.json').read_text(encoding='utf8')), st, time.perf_counter(), st.get('failures', 0))
