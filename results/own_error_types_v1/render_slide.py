"""Slide figure P1: natural first errors on the models' own answers by type, next to planted PRMBench errors (N16)."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
T = {'bg': '#fcfcfb', 'ink': '#1b2430', 'ink2': '#5b6575', 'grid': '#e4e2dc', 'ours': '#2a78d6', 'ref': '#eb6834', 'muted': '#898781'}
D = pd.read_csv(OUT / 'PER_CATEGORY.csv').set_index('category')
ROWS = [('counterfactual', 'counter-\nfactual'), ('step_contradiction', 'step\ncontradiction'), ('domain_inconsistency', 'domain\ninconsistency'),
        ('missing_condition', 'missing\ncondition'), (None, None), ('calculation_error', 'calculation\nerror'), ('misread_problem', 'misread\nproblem'),
        ('wrong_method', 'wrong\nmethod'), (None, None), ('ALL', 'all own\nerrors')]
fig, ax = plt.subplots(figsize=(15, 5.4), facecolor=T['bg']); ax.set_facecolor(T['bg'])
for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
for sp in ('left', 'bottom'): ax.spines[sp].set_color(T['muted'])
ax.grid(axis='y', color=T['grid'], linewidth=.8); ax.set_axisbelow(True); ax.tick_params(colors=T['ink2'])
ticks, labs = [], []
for x, (k, lab) in enumerate(ROWS):
    if k is None: continue
    r = D.loc[k]; n = int(r.n_answers)
    ax.errorbar(x - .15, r['B16__BASE__wall'], yerr=[[r['B16__BASE__wall'] - r['B16__BASE__wall_lo']], [r['B16__BASE__wall_hi'] - r['B16__BASE__wall']]],
                fmt='o', color=T['ours'], markersize=8, capsize=3)
    if pd.notna(r.prmb_N16_within_mean5):
        ax.plot(x + .1, r.prmb_N16_within_mean5, 'o', mfc='none', mec=T['ours'], mew=1.8, markersize=9)
        ax.plot(x + .3, r.prmb_supervised_prm_within, 'o', color=T['ref'], markersize=8)
    ax.plot(x + .45, r['STEP_INDEX__wall'], 'D', mfc='none', mec=T['muted'], mew=1.5, markersize=7)
    ticks.append(x); labs.append(f'{lab}\nn={n} ({n / 417:.0%})' if k != 'ALL' else f'{lab}\nn=417')
ax.axhline(.5, color=T['muted'], linewidth=.8, linestyle=':')
ax.set_xticks(ticks); ax.set_xticklabels(labs, color=T['ink'], fontsize=9.5)
ax.text(1.5, 1.02, 'kinds PRMBench also plants (30%)', transform=ax.get_xaxis_transform(), ha='center', color=T['ink2'], fontsize=10)
ax.text(6, 1.02, 'kinds PRMBench never plants (70%)', transform=ax.get_xaxis_transform(), ha='center', color=T['ink2'], fontsize=10)
ax.set_ylabel('Within-answer AUC', color=T['ink2']); ax.set_ylim(.25, 1.0)
hs = [plt.Line2D([], [], marker='o', ls='', color=T['ours'], markersize=8, label='Own answers, natural errors (label-free, N16 channels, 95% interval)'),
      plt.Line2D([], [], marker='o', ls='', mfc='none', mec=T['ours'], markersize=9, label='PRMBench, planted errors (label-free, N16)'),
      plt.Line2D([], [], marker='o', ls='', color=T['ref'], markersize=8, label='PRMBench, supervised PRM-7B'),
      plt.Line2D([], [], marker='D', ls='', mfc='none', mec=T['muted'], markersize=7, label='Own answers, step index alone')]
ax.legend(handles=hs, loc='lower center', bbox_to_anchor=(.5, -.36), ncol=2, frameon=False, labelcolor=T['ink'], fontsize=9.5)
ax.set_title('Natural first errors in Qwen3 own answers (GSM8K, MATH), by kind', color=T['ink'], fontsize=11.5, loc='left', pad=22)
fig.text(.01, -.13, 'Own answers: first-error step vs all other steps; types by two blind Claude classification passes (kappa 0.86 on 120). PRMBench AUC compares all error steps with all correct steps. '
         'redundancy, circular and deception never occur as natural first errors; kinds with n < 10 not shown.', color=T['ink2'], fontsize=8.3)
fig.tight_layout(); (OUT / 'figures').mkdir(exist_ok=True)
fig.savefig(OUT / 'figures/P1_planted_natural_light.png', dpi=170, bbox_inches='tight', facecolor=T['bg'])
fig.savefig(OUT / 'figures/P1_planted_natural_light.svg', bbox_inches='tight', facecolor=T['bg'])
print('ok')
