"""Publication-friendly paired contrasts; metrics stay in separate panels."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/lsml_external_generalization_v1/evaluation'

def main():
    assert (OUT/'RED_TEAM.md').exists()
    comparisons=json.loads((OUT/'CONTRASTS.json').read_text(encoding='utf8'))
    cells=['hard2verify_qwen3_8b','socratic_qwen3_8b','socratic_qwq32b']
    titles=['Hard2Verify / Qwen3-8B\nBalanced F1','Socratic / Qwen3-8B\nPRMScore','Socratic / QwQ-32B\nPRMScore']
    fig,axes=plt.subplots(1,3,figsize=(13.5,4.6),sharey=True,constrained_layout=True)
    labels=['Frozen vs equal','Frozen vs partition equal','Local vs equal','Local vs partition equal','Frozen vs CT7','Local vs CT7']
    for ax,cell,title in zip(axes,cells,titles):
        rows=[r for r in comparisons if r['cell']==cell]
        assert len(rows)==6
        ax.axvline(0,color='0.6',linewidth=1)
        for i,r in enumerate(rows):
            lo,hi=[100*v for v in r['ci_bonferroni']];x=100*r['delta']
            color='#17669c' if r['left']=='frozen_lsml' else '#ab5821'
            ax.plot([lo,hi],[i,i],color=color,lw=2)
            ax.plot(x,i,'o',color=color,ms=6)
        ax.set_title(title,fontsize=11);ax.set_xlabel('L-SML difference (percentage points)')
        ax.set_yticks(range(6),labels);ax.grid(axis='x',alpha=.2)
        ax.spines[['top','right']].set_visible(False)
    axes[0].invert_yaxis()
    fig.suptitle('Source-frozen transfer: paired source-question intervals\n100,000 draws; Bonferroni correction across18 primary contrasts',fontsize=12)
    fig.savefig(OUT/'PAIRED_CONTRASTS.png',dpi=180)
    fig.savefig(OUT/'PAIRED_CONTRASTS.pdf')
    plt.close(fig)
    print('saved PAIRED_CONTRASTS.png/pdf')
if __name__=='__main__':main()
