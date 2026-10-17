"""Display saved first-token diagnostics; no tokenization or inference is run."""
import os
os.environ['MPLCONFIGDIR'] = '/private/tmp/advisor-v16-digit-preview/mpl'
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
source = ROOT / 'results/advisor_completion_2026-10-11/sources/research/dataset_cache/four_localization/prmbench_qwen3_8b_telemetry_thinkfix/THINKFIX_STATS.json'
d = json.loads(source.read_text())
fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained',gridspec_kw={'width_ratios':[1.8,1]})
for key,label,color in [('old','Old prefix','#8998a4'),('new','Corrected prefix','#176b9a')]:
    y=[d[key][f'surprisal_median_pos{i}'] for i in range(5)]
    axes[0].plot(range(5),y,'o-',label=label,color=color,lw=2.3)
    axes[0].annotate(f'{y[0]:.1f}',(0,y[0]),xytext=(9,2),textcoords='offset points',color=color,fontsize=12)
axes[0].set(title='Median surprisal of the first five content tokens',xlabel='Original content-token position (0-based)',ylabel='Surprisal (nats)',xticks=range(5),ylim=(0,39))
axes[0].legend(frameon=False,loc='upper right')
axes[1].bar(['Old prefix','Corrected prefix'],[100*d['old']['frac_first_position_top1_think'],100*d['new']['frac_first_position_top1_think']],color=['#8998a4','#176b9a'],width=.6)
axes[1].set(title='First-position top prediction is <think>',ylabel='Share of answers (%)',ylim=(0,115))
axes[1].text(0,103,'100%',ha='center',fontsize=12)
axes[1].text(1,3,'0%',ha='center',fontsize=12)
for ax in axes:
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.18)
fig.text(.02,-.08,'Qwen3-8B · PRMBench · 6,969 fixed answers including controls · saved THINKFIX_STATS.json.\nToken 0 remains high after the prefix fix; mask1 is a separate removal, not a new model prediction.',fontsize=9,color='#596b78')
fig.savefig(HERE/'prefix_token_diagnostic.png',dpi=170,bbox_inches='tight')
fig.savefig(HERE/'prefix_token_diagnostic.svg',bbox_inches='tight')
print(HERE/'prefix_token_diagnostic.png')
