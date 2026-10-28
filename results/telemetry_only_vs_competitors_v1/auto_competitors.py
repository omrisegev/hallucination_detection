"""AUTO (TELE16 + PRM, free L-SML grouping over [z(PRM), the 16 backbone channels]; the PRM is one of the channels)
under the same PRISM, SLIDER and CROP protocols as compute.py, on the PRM-backbone cells.

AUTO is re-fitted with stage8_auto_forced_k.auto unchanged, exactly as results/prm_sub5_all_benchmarks_v1/auto_new.py
does (PRMBench: z of PRM_Q80; other benchmarks: z of the external 'PRM' arm). Every metric comes from compute.py's
functions (PRMBench: prmbench_core; ProcessBench: processbench_core; CROP: crop_table6 / crop_cross), run with
MODE = 'prm' and the arm list [AUTO_N16, TELE_N16, MEAN_N16]; TELE_N16 / MEAN_N16 here are the BACKBONE telemetry arms.

Parity first: AUTO's PRMBench PRMScore and within-AUC vs stage8/AUTO_FORCEDK_RESULTS.json, AUTO's ProcessBench /
CRV / PRM800K main metric vs prm_sub5_all_benchmarks_v1/AUTO_NEW.csv, backbone TELE_N16 vs STATUS_MAIN, CROP kept of
backbone TELE_N16 vs CROP_TABLE6_CAB.csv.

    python auto_competitors.py     # -> RESULTS_BACKBONE.csv (all arms), AUTO rows appended to RESULTS.csv, PARITY_AUTO.json
"""
import csv, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compute as C  # noqa: E402
A, E, CAB = C.A, C.E, C.CAB
sys.path.insert(0, str(A.ROOT / 'results/prm_boost_math_rigor_v1'))
import stage8_auto_forced_k as S8  # noqa: E402

S8_REF = {(r['bench'], r['prm']): r['A']['auto'] for r in
          json.loads((A.ROOT / 'results/prm_boost_math_rigor_v1/stage8/AUTO_FORCEDK_RESULTS.json').read_text())}
AUTO_NEW = {(r['benchmark'], r['prm'], r['arm'], r['metric']): float(r['value']) for r in E.read_csv(A.HERE / 'AUTO_NEW.csv')}
INFO = {}


def add_auto(c, bench, prm):
    zp = E.answer_z(c.arms['PRM_Q80' if bench == 'prmbench' else 'PRM'], c.off)
    s, info = S8.auto(E.channels(c, 'N16'), zp)
    c.arms['AUTO_N16'] = s; INFO[f'{bench}/{prm}'] = {k: info[k] for k in ('K', 'prm_alone')}


def main():
    C.MODE, C.CELL_HOOK, C.TELE = 'prm', add_auto, ['AUTO_N16', 'TELE_N16', 'MEAN_N16']
    crop_ref = {(r['benchmark'], r['prm'], r['arm']): float(r['kept']) for r in E.read_csv(A.HERE / 'vendor/CROP_TABLE6_CAB.csv')
                if r['mode'] == 'prm' and r['grid'] == 'q201'}
    for prm in C.PAIRS:
        # PRMBench (collection cell, PRM backbone)
        c = E.load('prmbench', prm); add_auto(c, 'prmbench', prm)
        n0 = len(C.ROWS); C.prmbench_core(c, prm)
        got = lambda arm, m='PRMScore': next(r['value'] for r in C.ROWS[n0:] if r['protocol'] == 'PRISM T2' and r['arm'] == arm
                                             and r['metric'] == m and r.get('pool') == 'non-control')
        ref = S8_REF[('prmbench', prm)]
        C.gate(f'{prm} PRMBench AUTO PRMScore GAP vs stage8', got('AUTO_N16'), ref['gap_prmscore'])
        C.gate(f'{prm} PRMBench AUTO within-AUC vs stage8', float(np.nanmean(E.within(c, 'AUTO_N16'))), ref['within_auc'], 1e-4)
        C.gate(f'{prm} PRMBench backbone TELE_N16 PRMScore GAP vs STATUS_MAIN', got('TELE_N16'), A.S1.STATUS[('prmbench', prm, 'TELE_N16')]['GAP'])
        del c
        # ProcessBench (backbone cell)
        c = C.cab_cell('processbench', prm); R = C.processbench_core(c, prm)
        C.gate(f'{prm} ProcessBench AUTO avg F1 NC vs AUTO_NEW', R['AUTO_N16_NC']['avg_f1'], AUTO_NEW[('processbench', prm, 'AUTO_N16', 'avg_f1_NC')])
        del c
        # CROP Table 6 on CRV-Arith, CRV-GSM8K, PRM800K
        for bench in ('crv_arith', 'crv_gsm8k', 'prm800k'):
            c = C.cab_cell(bench, prm)
            C.gate(f'{prm} {bench} AUTO macro-F1 GAP vs AUTO_NEW', CAB.row(c, 'prm', bench, 'common', 'AUTO_N16', 'GAP')['macro_f1'],
                   AUTO_NEW[(bench, prm, 'AUTO_N16', 'macro_f1_GAP')])
            kept = C.crop_table6(c, bench, prm)
            C.gate(f'{prm} {bench} backbone TELE_N16 CROP kept vs CROP_TABLE6_CAB', kept[('TELE_N16', 'q201')],
                   crop_ref[(bench, A.PFILE.get(prm, prm), 'TELE_N16')])
            del c
        C.crop_cross('crv_arith', 'crv_gsm8k', prm); C.crop_cross('crv_gsm8k', 'crv_arith', prm)
        print(prm, 'done', flush=True)
    for r in C.ROWS:
        r['standalone_model'] = 'PRM backbone (not standalone)'
    E.write_csv(HERE / 'RESULTS_BACKBONE.csv', C.ROWS)
    fails = [p['check'] for p in C.PAR if not p['ok']]
    json.dump(dict(n_checks=len(C.PAR), failed=fails, max_abs_diff=max(p['abs_diff'] for p in C.PAR), auto_fit=INFO, checks=C.PAR),
              open(HERE / 'PARITY_AUTO.json', 'w'), indent=1, default=float)
    print('rows', len(C.ROWS), 'parity', len(C.PAR), 'failed', fails)
    if fails: raise SystemExit(3)
    # AUTO rows into RESULTS.csv (the backbone TELE/MEAN rows stay in RESULTS_BACKBONE.csv, to keep RESULTS.csv standalone)
    old = [r for r in csv.DictReader(open(HERE / 'RESULTS.csv', encoding='utf-8')) if r['arm'] != 'AUTO_N16']
    new = [r for r in C.ROWS if r['arm'] == 'AUTO_N16']
    cols = list(old[0]); E.write_csv(HERE / 'RESULTS.csv', old + [{k: r.get(k, '') for k in cols} for r in new])
    print('RESULTS.csv: +', len(new), 'AUTO rows')


if __name__ == '__main__':
    main()
