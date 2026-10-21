"""Compact console view of the stage-2 CSVs (no new computation)."""
import csv
from pathlib import Path
H = Path(__file__).resolve().parent
rd = lambda f: list(csv.DictReader(open(H / f)))
f = lambda v, d=2: f'{float(v):.{d}f}' if v not in ('', 'nan', None) else 'nan'
print('== policies ==')
for r in rd('RESULTS.csv'):
    print(r['bench'][:3], f"{r['arm']:24s} {r['rule']:7s} PS={f(r['metric'])} MAE={f(r['count_mae'],3)} pass={f(r['pass_share_noncontrol'],3)} "
          f"zacc={f(r['zero_acc'],3)} exact={f(r['exact_count_acc'],3)} {'BEST' if r['best_of_6_label_selected']=='True' else ''}")
print('== diagnostics (SUCCESS_SET) ==')
for r in rd('DIAGNOSTICS.csv'):
    if r['target'] == 'SUCCESS_SET':
        print(r['bench'][:3], f"{r['arm']:24s} {r['population']:14s} {r['quantity']:10s} N={r['n_answers']} succ={f(r['success_rate'],3)} "
              f"med s/f={f(r['median_success'],3)}/{f(r['median_failure'],3)} AUC={f(r['auroc_pooled'],3)} strat={f(r['auroc_exact_n_strat'],3)} "
              f"cov={r['strat_coverage']} pS|n={f(r['partial_spearman_given_n'],3)} rho_n={f(r['spearman_with_n'],2)}")
print('== logit ==')
for r in rd('LOGIT.csv'):
    print(r['bench'][:3], f"{r['arm']:24s} {r['model'][-9:]:9s} z_logn={f(r['z_logn'],1)} z_logG1={f(r['z_logG1'],1)} z_second={f(r['z_second'],1)} b_G1={f(r['b_logG1'],2)} b_2={f(r['b_second'],2)}")
print('== boot diag ==')
for r in rd('BOOT_DIAG.csv'):
    print(r['bench'][:3], f"{r['arm']:24s} {r['contrast']} {f(r['diff'],3)} [{f(r['lo'],3)}, {f(r['hi'],3)}]")
print('== clean ==')
for r in rd('CLEAN.csv'):
    print({k: v for k, v in r.items() if v != ''})
print('== tau best ==')
T = rd('TAU_CURVE.csv'); keys = sorted({(r['bench'], r['arm'], r['fallback']) for r in T})
for k in keys:
    rr = [r for r in T if (r['bench'], r['arm'], r['fallback']) == k]; b = max(rr, key=lambda r: float(r['metric']))
    q0 = [r for r in rr if r['tau_quantile'] == '0.0'][0]
    print(k, 'best q', b['tau_quantile'], 'tau', f(b['tau'], 3), 'PS', f(b['metric']), 'pass', f(b['pass_share_noncontrol'], 3), '| q0 PS', f(q0['metric']))
print('== contrasts ==')
for r in rd('CONTRASTS.csv'):
    print(r['bench'][:3], f"{r['arm']:24s} {r['contrast']:16s} {float(r['diff']):+.2f} [{float(r['lo']):+.2f}, {float(r['hi']):+.2f}]")
