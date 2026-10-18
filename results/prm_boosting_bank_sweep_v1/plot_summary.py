"""Summary figure of the PRM bank sweep (job 288319): PRMScore per arm, and CHAIN gain over the PRM per bank."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

D = Path(__file__).resolve().parent
PRMS = ['qwen_prm_7b', 'qwen_prm_72b', 'pure_prm', 'rlhflow_prm', 'math_shepherd']
NAME = ['Qwen-PRM-7B', 'Qwen-PRM-72B', 'PURE-PRM', 'RLHFlow-PRM', 'MathShepherd']
R = {m: {(r['arm'], r['rule']): float(r['prmscore']) for r in csv.DictReader(open(D / f'SWEEP_RESULTS_{m}.csv'))}
     for m in PRMS}
B = {m: {r['contrast']: r for r in csv.DictReader(open(D / f'SWEEP_BOOTSTRAP_{m}.csv')) if r['metric'] == 'prmscore'}
     for m in PRMS}
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK})

fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 9.5), facecolor=SURF)
for ax in (a1, a2):
    ax.set_facecolor(SURF); ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.grid(axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)

# Panel 1: PRMScore per PRM
arms = [('PRM_native', 'PRM alone', '#2a78d6'), ('TELE_N16', 'Our signals alone (N16)', '#eb6834'),
        ('PRMG_N16', 'PRM + signals, L-SML (PRMG_N16)', '#1baf7a'),
        ('CHAIN_N16', 'PRM + signals, L-SML chain (CHAIN_N16)', '#eda100')]
x, w = np.arange(len(PRMS)), 0.2
for i, (arm, lab, c) in enumerate(arms):
    v = [R[m][(arm, 'Q80')] for m in PRMS]
    bars = a1.bar(x + (i - 1.5) * w, [y - 40 for y in v], w * 0.92, bottom=40, color=c, label=lab)
    for b, y in zip(bars, v):
        a1.text(b.get_x() + b.get_width() / 2, y + 0.3, f'{y:.1f}', ha='center', va='bottom', fontsize=7.5, color=INK2, zorder=5,
                bbox=dict(boxstyle='square,pad=0.1', fc=SURF, ec='none'))
for j, m in enumerate(PRMS):
    y = R[m][('LSML_CONT', 'Q80')]
    a1.hlines(y, j - 2 * w, j + 2 * w, colors=INK, linestyles='--', lw=1.2, zorder=4,
              label='Reference: 3 label-chosen signals + PRM (LSML_CONT)' if j == 0 else None)
a1.set_xticks(x, NAME); a1.set_ylim(40, 75); a1.set_ylabel('PRMScore (official, Q80 flags)')
a1.set_title('A. PRMScore on PRMBench (6,969 answers per PRM)', loc='left', fontsize=12, color=INK)
a1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False, fontsize=9)

# Panel 2: CHAIN gain over PRM alone, per bank, with 95% CI
banks = [('N11', 'N11 (no digits)', '#86b6ef'), ('N13', 'N13 (no digits)', '#5598e7'),
         ('N14', 'N14 (with digits)', '#256abf'), ('N16', 'N16 (with digits)', '#104281')]
for i, (b, lab, c) in enumerate(banks):
    rows = [B[m][f'CHAIN_{b}[Q80] - PRM_native[Q80]'] for m in PRMS]
    d = np.array([float(r['delta']) for r in rows])
    lo = d - np.array([float(r['ci_low']) for r in rows]); hi = np.array([float(r['ci_high']) for r in rows]) - d
    a2.bar(x + (i - 1.5) * w, d, w * 0.92, color=c, label=lab, yerr=[lo, hi],
           error_kw=dict(ecolor=INK2, lw=1, capsize=2))
a2.axhline(0, color=INK, lw=1)
a2.set_xticks(x, NAME); a2.set_ylabel('PRMScore gain over PRM alone')
a2.set_title('B. Gain from adding our signals to the PRM (CHAIN arm), by signal bank; bars = 95% CI',
             loc='left', fontsize=12, color=INK)
a2.legend(loc='upper left', ncol=4, frameon=False, fontsize=9)
fig.text(0.01, 0.005, 'PRMBench development data. Source: results/prm_boosting_bank_sweep_v1/SWEEP_RESULTS_*.csv, '
         'SWEEP_BOOTSTRAP_*.csv (AIRCC job 288319).', fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.02, 1, 1))
fig.savefig(D / 'SUMMARY.png', dpi=150, facecolor=SURF)
