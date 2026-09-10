"""Small English comparison chart from saved full-development metrics."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'results/surprisal_power_fusion_v1'
    m=json.loads((out/'METRICS.json').read_text(encoding='utf8'))['metrics']
    fig,axes=plt.subplots(1,3,figsize=(12,4.5))
    fig.subplots_adjust(left=.06,right=.99,bottom=.19,top=.73,wspace=.24)
    panels=[('pb_all8',100,'ProcessBench: macro F1 (%)','ref__k15__raw','Varentropy, K=15'),
        ('prm_within',1,'PRMBench: within-answer AUC','ref__k15__equal','Normalized contributions, K=15'),
        ('prmscore_q08',1,'PRMScore','ref__k50__raw','Varentropy, K=50')]
    for ax,(key,scale,title,ref,ref_name) in zip(axes,panels):
        for method,label,color in [('equal','Normalized equal weights','#246c96'),('iu','IU-PCR','#d56a20')]:
            values=[m[f'd{d}__{method}'][key]*scale for d in (1,2,3)]
            ax.plot([1,2,3],values,'o-',color=color,label=label,lw=2)
        baseline=m[ref][key]*scale
        ax.axhline(baseline,color='#555555',ls='--',lw=1.3,label='Prior leader')
        ax.set_title(f'{title}\nPrior: {ref_name} = {baseline:.4f}',fontsize=9);ax.set_xticks([1,2,3],['1','1 + 2','1 + 2 + 3'])
        ax.set_xlabel('Included powers (ranks + chosen token)');ax.grid(alpha=.15)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.92),ncol=3,frameon=False)
    fig.suptitle('Full cached localization benchmark | Same labels, folds, gate and step readout',fontsize=11,y=.98)
    fig.savefig(out/'COMPARISON.png',dpi=160)
    plt.close(fig)
    print(out/'COMPARISON.png')


if __name__=='__main__':main()
