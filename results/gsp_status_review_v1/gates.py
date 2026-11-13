import csv
from pathlib import Path
R = Path('/home/user/hallucination_detection/results')
OUT = Path(__file__).resolve().parent
MET = {'prmbench': 'PRMScore', 'socratic': 'PRMScore', 'hard2verify': 'balanced F1', 'processbench': 'macro SLA'}
mine = {}
for b in MET:
    for r in csv.DictReader(open(OUT / f'RESULTS_{b}.csv')): mine[(b, r['prm'], r['arm'], r['metric'])] = float(r['value'])
n = bad = 0; lines = []
for b, m in MET.items():
    for r in csv.DictReader(open(R / f'sml_readout_calibration_v1/RESULTS_{b}.csv')):
        for ref, me in (('A0 top10', 'F3 GSP-T | R1'), ('A2 min rank-1 misfit', 'F6 A2 -> F3 GSP-T | R1')):
            if r['arm'] == ref and r['metric'] in (m, 'within-AUC', 'FE-AUC position-matched'):
                v = mine[(b, r['prm'], me, r['metric'])]; ok = abs(v - float(r['value'])) < 1e-6; n += 1; bad += not ok
                lines.append(f"{'PASS' if ok else 'FAIL'} {b} {r['prm']} {me} {r['metric']} mine={v:.4f} ref={float(r['value']):.4f}")
ref = list(csv.DictReader(open(R / 'c4u_tele16_all_benchmarks_v1/RESULTS_sa.csv')))
for b, proto, met, kw in (('prmbench', 'PRISM T2', 'PRMScore', dict(pool='non-control')), ('processbench', 'MtG T3 SLA', 'macro SLA', {})):
    for prm in ['qwen_prm_7b', 'qwen_prm_72b', 'pure_prm', 'rlhflow_prm']:
        h = [float(r['value']) for r in ref if r['prm'] == prm and r['benchmark'] == b and r['protocol'] == proto and r['arm'] == 'TELE_N16' and r['metric'] == met and all(r.get(k) == v for k, v in kw.items())]
        v = mine[(b, prm, 'F2 L-SML | R0', met)]; ok = len(h) == 1 and abs(v - h[0]) < 1e-6; n += 1; bad += not ok
        lines.append(f"{'PASS' if ok else 'FAIL'} {b} {prm} F2 L-SML | R0 {met} mine={v:.4f} stored TELE_N16={h}")
lines.append(f'{n} checks, {bad} failed')
open(OUT / 'GATES.txt', 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines))
