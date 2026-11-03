"""Merge the per-PRM outputs in out_aircc/ into RESULTS.csv, BOOTSTRAP.csv, DIAG.csv and TABLES.md."""
import csv, json
from pathlib import Path

D = Path(__file__).resolve().parent; S = D / 'out_aircc'
PRMS = [('qwen_prm_7b', 'Qwen-7B'), ('qwen_prm_72b', 'Qwen-72B'), ('pure_prm', 'PURE'), ('rlhflow_prm', 'RLHFlow')]


def merge(kind):
    rows = [r for t, _ in PRMS for r in csv.DictReader(open(S / f'{kind}_{t}.csv'))]
    with open(D / f'{kind}.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
    return rows


res, boot, diag = merge('RESULTS'), merge('BOOTSTRAP'), merge('DIAG')
fit = {t: json.loads((S / f'FIT_{t}.json').read_text()) for t, _ in PRMS}
R = {(r['model'], r['arm']): r for r in res}
Bd = {(r['model'], r['unit'], r['contrast'], r['metric']): r for r in boot}
L = []


def ci(t, n, mt, unit='answer'):
    r = Bd[(t, unit, f'{n} - Z17', mt)]; d, lo, hi = float(r['delta']), float(r['ci_low']), float(r['ci_high'])
    k = 2 if mt == 'prmscore' else 4
    return f"{d:+.{k}f} [{lo:+.{k}f}, {hi:+.{k}f}]" + ('*' if lo > 0 or hi < 0 else '')


L += ['### Table 1. Metrics per arm (PRMBench, N = 6,969 answers; within-AUC over 6,030 answers with an error)', '',
      '| PRM | arm | within-AUC | pooled step AUC | answer AUC (max step) | PRMScore |', '|---|---|---|---|---|---|']
for t, nm in PRMS:
    for a in ('PRM_native', 'PRM_Q80_PROB', 'Z17', 'CAT34'):
        r = R[(t, a)]
        L.append(f"| {nm} | {a} | {float(r['within_auc']):.4f} | {float(r['pooled_auc']):.4f} | "
                 f"{float(r['answer_auc_max']):.4f} | {float(r['prmscore']):.2f} |")
L += ['', '### Table 2. Difference vs Z17 (95% CI; answer-level bootstrap, 1,000 draws; * = CI excludes 0)', '',
      '| PRM | contrast | within-AUC diff | PRMScore diff | PRMScore diff, source-group bootstrap |', '|---|---|---|---|---|']
for t, nm in PRMS:
    for a in ('CAT34', 'PRM_Q80_PROB'):
        L.append(f"| {nm} | {a} - Z17 | {ci(t, a, 'within_auc')} | {ci(t, a, 'prmscore')} | {ci(t, a, 'prmscore', 'source_group')} |")
wins = sum(float(Bd[(t, 'answer', 'CAT34 - Z17', 'prmscore')]['ci_low']) > 0 for t, _ in PRMS)
L += ['', f'Pre-declared rule (CAT34 beats Z17 if PRMScore diff > 0 with CI excluding 0 in >= 3/4 PRMs): '
      f'**{wins}/4 -> {"MET" if wins >= 3 else "NOT MET"}**.', '',
      '### Table 3. Grouping and spectrum', '',
      '| PRM | Z17 K | CAT34 K | raw_i in same CAT34 group as z_i | lead eigenvalue share, Z17 inputs | lead eigenvalue share, CAT34 inputs |',
      '|---|---|---|---|---|---|']
for t, nm in PRMS:
    f = fit[t]
    L.append(f"| {nm} | {f['Z17']['K']} | {f['CAT34']['K']} | {f['CAT34']['n_pairs_same_group']}/17 | "
             f"{f['Z17']['lead_eig_share_all_inputs']:.3f} | {f['CAT34']['lead_eig_share_all_inputs']:.3f} |")
L += ['', '### Table 4. |corr(raw_i, z_i)| by step class (raw = global rank-normal; mean over the 4 PRMs)', '',
      '| channel | clean steps | error steps | pairs grouped together (of 4 PRMs) |', '|---|---|---|---|']
for c in [r['channel'] for r in diag if r['model'] == 'qwen_prm_7b']:
    rr = [r for r in diag if r['channel'] == c]
    L.append(f"| {c} | {sum(float(r['abs_corr_raw_z_clean']) for r in rr) / 4:.2f} | "
             f"{sum(float(r['abs_corr_raw_z_error']) for r in rr) / 4:.2f} | {sum(r['paired_in_CAT34'] == 'True' for r in rr)} |")
L += ['', '### Table 5. CAT34 groups per PRM', '']
for t, nm in PRMS:
    L.append(f"- **{nm}**: " + '; '.join(f"G{g} = " + ', '.join(v) for g, v in fit[t]['CAT34']['groups'].items()))
(D / 'TABLES.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
print('\n'.join(L))
