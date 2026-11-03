"""Merge socratic/out/ per-PRM files into RESULTS.csv, BOOTSTRAP.csv, DIAG.csv and TABLES.md."""
import csv, json
from pathlib import Path

D = Path(__file__).resolve().parent; S = D / 'out'
PRMS = [('qwen_prm_7b', 'Qwen-7B'), ('qwen_prm_72b', 'Qwen-72B'), ('pure_prm', 'PURE'), ('rlhflow_prm', 'RLHFlow')]


def merge(kind):
    rows = [r for t, _ in PRMS for r in csv.DictReader(open(S / f'{kind}_{t}.csv'))]
    with open(D / f'{kind}.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
    return rows


res, boot, diag = merge('RESULTS'), merge('BOOTSTRAP'), merge('DIAG')
fit = {t: json.loads((S / f'FIT_{t}.json').read_text()) for t, _ in PRMS}
R = {(r['model'], r['arm']): r for r in res}
Bd = {(r['model'], r['contrast'], r['metric']): r for r in boot}


def ci(t, a, mt):
    r = Bd[(t, f'{a} - Z17_Q80', mt)]; d, lo, hi = (float(r[k]) for k in ('delta', 'ci_low', 'ci_high'))
    f = 100 if mt == 'official_metric' else 1; k = 2 if mt == 'official_metric' else 4
    return f"{d * f:+.{k}f} [{lo * f:+.{k}f}, {hi * f:+.{k}f}]" + ('*' if lo > 0 or hi < 0 else '')


L = ['### Table 1. Socratic-PRMBench, 2,995 answers (PRMScore x100; within-AUC over included non-empty steps)', '',
     '| PRM | arm | within-AUC | pooled step AUC | PRMScore |', '|---|---|---|---|---|']
for t, nm in PRMS:
    for a in ('PRM_native', 'PRMPROB_Q80', 'POS_ONLY_Q80', 'Z17_Q80', 'CAT34_Q80'):
        r = R[(t, a)]
        L.append(f"| {nm} | {a} | {float(r['within_auc']):.4f} | {float(r['pooled_auc']):.4f} | {100 * float(r['official_metric']):.2f} |")
L += ['', '### Table 2. Difference vs Z17 (95% CI, answer-level bootstrap, 1,000 draws; * = CI excludes 0)', '',
      '| PRM | contrast | PRMScore diff | within-AUC diff |', '|---|---|---|---|']
for t, nm in PRMS:
    for a in ('CAT34_Q80', 'PRMPROB_Q80', 'POS_ONLY_Q80'):
        L.append(f"| {nm} | {a} - Z17 | {ci(t, a, 'official_metric')} | {ci(t, a, 'within_auc')} |")
wins = sum(float(Bd[(t, 'CAT34_Q80 - Z17_Q80', 'official_metric')]['ci_low']) > 0 for t, _ in PRMS)
L += ['', f'Pre-declared rule (PRMScore diff > 0 with CI excluding 0 in >= 3/4 PRMs): **{wins}/4 -> '
      f'{"MET" if wins >= 3 else "NOT MET"}**.', '',
      '### Table 3. Grouping and spectrum', '',
      '| PRM | Z17 K | CAT34 K | raw_i in same group as z_i | lead eigenvalue share Z17 / CAT34 | constant views dropped |', '|---|---|---|---|---|---|']
for t, nm in PRMS:
    f = fit[t]
    L.append(f"| {nm} | {f['Z17']['K']} | {f['CAT34']['K']} | {f['CAT34']['n_pairs_same_group']}/17 | "
             f"{f['Z17']['lead_eig_share_all_inputs']:.3f} / {f['CAT34']['lead_eig_share_all_inputs']:.3f} | "
             f"{', '.join(f['CAT34']['dropped_constant']) or 'none'} |")
L += ['', '### Table 4. |corr(raw_i, z_i)| by step class (mean over the 4 PRMs)', '',
      '| channel | clean steps | error steps | pairs grouped together (of 4) |', '|---|---|---|---|']
for c in [r['channel'] for r in diag if r['model'] == 'qwen_prm_7b']:
    rr = [r for r in diag if r['channel'] == c]
    L.append(f"| {c} | {sum(float(r['abs_corr_raw_z_clean']) for r in rr) / 4:.2f} | "
             f"{sum(float(r['abs_corr_raw_z_error']) for r in rr) / 4:.2f} | {sum(r['paired_in_CAT34'] == 'True' for r in rr)} |")
L += ['', '### Table 5. CAT34 groups per PRM', '']
for t, nm in PRMS:
    L.append(f"- **{nm}**: " + '; '.join(f"G{g} = " + ', '.join(v) for g, v in fit[t]['CAT34']['groups'].items()))
(D / 'TABLES.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
print('\n'.join(L))
