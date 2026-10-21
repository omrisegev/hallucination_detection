"""Section 3 report: old (stale) vs new numbers side by side, with parity anchors.

    python section3_report.py <local dir with bank_sweep_digitfix_v1/, controls_digitfix_v1/, score_banks_thinkfix_mask1/>

  tables/SECTION3_3B_SWEEP.csv      Math-Shepherd bank sweep, every arm x rule: old (zero-digit bank) vs new (digit-fixed)
  tables/SECTION3_3B_CONTROLS.csv   Math-Shepherd controls, every arm: old vs new
  tables/SECTION3_3A_BANKS.csv      PRMBench within-AUC per bank x arm: old template (removed sections 3/5) vs thinkfix_mask1
  SECTION3_PARITY.csv               3b: arms without digit channels (N11/N13, PRM_native, LSML_CONT) must not change;
                                    3a: LSML_CONT and ALL_EQ_ORIENTED vs thinkfix_mask1/cont_baselines/ABSOLUTE.csv
"""
import sys
from pathlib import Path
import pandas as pd

H = Path(__file__).resolve().parents[1]; R = H.parent; T = H / 'tables'


def side(old, new, keys, vals):
    m = old.merge(new, on=keys, how='outer', suffixes=('_old', '_new'))
    for v in vals: m[f'{v}_delta'] = m[f'{v}_new'] - m[f'{v}_old']
    return m


def main(d):
    d = Path(d); par = []
    o = pd.read_csv(R / 'prm_boosting_bank_sweep_v1/SWEEP_RESULTS_math_shepherd.csv')
    n = pd.read_csv(d / 'bank_sweep_digitfix_v1/SWEEP_RESULTS_math_shepherd.csv')
    S = side(o[['arm', 'rule', 'within_auc', 'prmscore']], n[['arm', 'rule', 'within_auc', 'prmscore']], ['arm', 'rule'], ['within_auc', 'prmscore'])
    S['has_digit_channels'] = S.arm.str.contains('N14|N16|S5|S6')
    S.to_csv(T / 'SECTION3_3B_SWEEP.csv', index=False)
    for _, r in S[~S.has_digit_channels].iterrows():
        for v in ('within_auc', 'prmscore'):
            par.append(dict(item='3b sweep (no digit channels: must not change)', key=f"{r.arm}[{r.rule}] {v}", old=r[f'{v}_old'], new=r[f'{v}_new'],
                            diff=r[f'{v}_delta'], status='PASS' if abs(r[f'{v}_delta']) < 1e-9 else 'FAIL'))
    o = pd.read_csv(R / 'prm_boosting_controls_v1/CONTROLS_RESULTS_math_shepherd.csv')
    n = pd.read_csv(d / 'controls_digitfix_v1/CONTROLS_RESULTS_math_shepherd.csv')
    C = side(o[['arm', 'within_auc', 'prmscore']], n[['arm', 'within_auc', 'prmscore']], ['arm'], ['within_auc', 'prmscore'])
    C['has_digit_channels'] = C.arm.str.contains('N16')
    C.to_csv(T / 'SECTION3_3B_CONTROLS.csv', index=False)
    for _, r in C[~C.has_digit_channels].iterrows():
        for v in ('within_auc', 'prmscore'):
            par.append(dict(item='3b controls (no digit channels: must not change)', key=f"{r.arm} {v}", old=r[f'{v}_old'], new=r[f'{v}_new'],
                            diff=r[f'{v}_delta'], status='PASS' if abs(r[f'{v}_delta']) < 1e-9 else 'FAIL'))
    o = pd.read_csv(R / 'sml_weight_estimator_v1/processbench_banks/RESULTS.csv')
    n = pd.read_csv(d / 'score_banks_thinkfix_mask1/RESULTS.csv')
    B = side(o[['bank', 'arm', 'prmb_within_auc']], n[['bank', 'arm', 'prmb_within_auc']], ['bank', 'arm'], ['prmb_within_auc'])
    B.to_csv(T / 'SECTION3_3A_BANKS.csv', index=False)
    A = pd.read_csv(R / 'thinkfix_rescore_v1/thinkfix_mask1/cont_baselines/ABSOLUTE.csv')
    for _, r in n[n.arm.isin(['LSML_CONT', 'ALL_EQ_ORIENTED'])].iterrows():
        ref = A[(A.bank == r.bank) & (A.scorer == r.arm)]
        if len(ref):
            dlt = r.prmb_within_auc - float(ref.prmb_within_auc.iloc[0])
            par.append(dict(item='3a vs thinkfix_mask1 ABSOLUTE.csv', key=f'{r.bank} {r.arm} prmb_within_auc', old=float(ref.prmb_within_auc.iloc[0]),
                            new=r.prmb_within_auc, diff=dlt, status='PASS' if abs(dlt) < 1e-9 else 'FAIL'))
    P = pd.DataFrame(par); P.to_csv(H / 'SECTION3_PARITY.csv', index=False)
    print(P.groupby(['item', 'status']).size().to_string()); print('max |diff| PASS rows', P[P.status == 'PASS']['diff'].abs().max())


if __name__ == '__main__':
    main(sys.argv[1])
