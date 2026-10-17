"""Plot the unchanged section-1 comparisons from saved v11 CSV rows."""
import os
os.environ['MPLCONFIGDIR'] = '/private/tmp/advisor-v16-digit-preview/mpl'
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
root = HERE.parents[2] / 'results/advisor_completion_2026-10-11'
rows = list(csv.DictReader((root / 'report/legacy/PRMBENCH_COMPARISON.csv').open()))
methods = [('ALL_EQ','Plain average'),('DS_FILTER_EQ','DS-filter average'),('LSML_CONT','L-SML'),('IU_PCR','IU-PCR'),('LR','LR (supervised)')]
fig,ax = plt.subplots(figsize=(10,4.4),layout='constrained')
for i,(method,label) in enumerate(methods):
    row=next(r for r in rows if r['bank']=='N16' and r['metric']=='prmb_within_auc' and r['scorer']==method)
    old=float(row['old_replay']); current=float(row['thinkfix_mask1'])
    color='#8a4590' if method=='LR' else '#176b9a'
    ax.plot([current,old],[i,i],color='#bdc7ce',lw=4,zorder=1)
    ax.scatter([old],[i],s=70,facecolors='white',edgecolors='#8998a4',linewidth=2,zorder=3,label='Old input' if i==0 else None)
    ax.scatter([current],[i],s=65,color=color,zorder=3,label='Corrected prefix + mask1' if i==0 else None)
    ax.annotate(f'{current:.3f}',(current,i),xytext=(-9,0),textcoords='offset points',ha='right',va='center',fontsize=11,color=color)
    ax.annotate(f'{old:.3f}  (−{100*(old-current):.2f} pp)',(old,i),xytext=(9,0),textcoords='offset points',ha='left',va='center',fontsize=10,color='#596b78')
ax.set_yticks(range(len(methods)),[x[1] for x in methods])
ax.set_xlim(.752,.843)
ax.set_ylim(4.6,-1)
ax.set_xlabel('Within-answer AUC · N16 · 6,030 answers')
ax.set_title('The correction lowers AUC; the main comparison remains',fontsize=15,fontweight='bold',pad=18)
ax.grid(axis='x',alpha=.2)
ax.spines[['top','right','left']].set_visible(False)
ax.legend(loc='upper left',ncol=2,frameon=False,fontsize=9)
fig.savefig(HERE/'slide1_corrections.png',dpi=170,bbox_inches='tight')
fig.savefig(HERE/'slide1_corrections.svg',bbox_inches='tight')
print(HERE/'slide1_corrections.png')
