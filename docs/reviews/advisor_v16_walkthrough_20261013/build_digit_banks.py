import os
os.environ['MPLCONFIGDIR'] = '/private/tmp/advisor-v16-digit-preview/mpl'
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path('/Users/osegev/Desktop/hallucination_detection/results/advisor_completion_2026-10-11')
out = Path(__file__).resolve().parent
chain = ['N14', 'N16', 'N25', 'N43', 'N56']
banks = json.loads((root/'sources/research/results/monday_nested_banks_v2/BANKS.json').read_text())
rows = list(csv.DictReader((root/'controls/ALL_BANK_TOP1_AUC.csv').open()))
rows = [r for r in rows if r['classification']=='ALL_NONCONTROL' and r['control']=='unrestricted']
values = {(r['bank'], r['method']): float(r['within_auc']) for r in rows}
lr = list(csv.DictReader((root/'report/legacy/PRMBENCH_COMPARISON.csv').open()))
for r in lr:
    if r['table']=='lr_reference' and r['scorer']=='LR' and r['metric']=='prmb_within_auc':
        values[(r['bank'], 'LR')] = float(r['thinkfix_mask1'])

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.1), layout='constrained')
fig.suptitle('Adding features while keeping all digit channels', fontsize=17, fontweight='bold')
methods = [m for m in sorted({r['method'] for r in rows}) if m not in ['PRM_NATIVE','STEP_INDEX','LSML_CONT_LEVELSIGN','IU_PCR_LEVELSIGN']]
selected = {'ALL_EQ': ('Plain average', '#78818a'), 'LSML_CONT': ('L-SML', '#176b9a'), 'IU_PCR': ('IU-PCR', '#d38128'), 'LR': ('LR (supervised)', '#8a4590')}
x = np.arange(len(chain))
baseline = np.array([values[(b,'ALL_EQ')] for b in chain])
for m in methods:
    if m not in selected:
        y = np.array([values[(b,m)] for b in chain])
        axes[0].plot(x,y,color='#c3cbd0',alpha=.6,lw=1,zorder=1)
for m,(label,color) in selected.items():
    y = np.array([values[(b,m)] for b in chain])
    axes[0].plot(x,y,'o-',color=color,label=label,lw=2.5,ms=6,zorder=3)
    axes[0].annotate(f'{y[-1]:.3f}',(x[-1],y[-1]),xytext=(6,0),textcoords='offset points',va='center',color=color,fontsize=10)
    if m!='ALL_EQ':
        axes[1].plot(x,100*(y-baseline),'o-',color=color,label=label,lw=2.5,ms=6)
        axes[1].annotate(f'{100*(y[-1]-baseline[-1]):+.2f}',(x[-1],100*(y[-1]-baseline[-1])),xytext=(6,0),textcoords='offset points',va='center',color=color,fontsize=10)
axes[0].set(title='Within-answer AUC', ylabel='AUC', ylim=(.735,.835))
axes[1].set(title='Gain over the plain average in the same bank', ylabel='AUC percentage points', ylim=(-.4,8.2))
axes[1].axhline(0,color='#78818a',lw=1)
for ax in axes:
    ax.set_xticks(x,chain)
    ax.set_xlim(-.2,4.55)
    ax.set_xlabel('Strictly nested banks: +2, +9, +18, +13 channels')
    ax.grid(axis='y',alpha=.2)
    ax.spines[['top','right']].set_visible(False)
axes[0].legend(loc='upper left',frameon=False,fontsize=9)
axes[1].legend(loc='upper left',frameon=False,fontsize=9)
fig.text(.02,-.04,'Saved v11 scores; corrected prefix + mask1; 6,030 answers. Faint curves: other label-free methods. Point estimates; no CI shown.',fontsize=9,color='#53606a')
fig.savefig(out/'digit_banks_preview.png',dpi=170,bbox_inches='tight')
fig.savefig(out/'digit_banks_preview.svg',bbox_inches='tight')
for b in chain:
    base=values[(b,'ALL_EQ')]
    print(b, ' | '.join(f'{label}: {values[(b,m)]:.4f} ({100*(values[(b,m)]-base):+.2f}pp)' for m,(label,_) in selected.items()))
