import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, json
from fast import wauc, get
from pm import pm_auc, az
D=get(); B,O='#257de0','#f47230'
N16=['q15_H1','q15_VE1','chosen_surprisal','logprob_margin','true_tail50','energy_level','energy_innovation','top15_turnover','top50_js','dominant_freq16','bocpd_p0','realized_z','realized_drv','digit_alternative','digit_spread','digit_alternative_innovation']
T3={'chosen_surprisal','realized_z','realized_drv'}
res={f:{k:(wauc(d['err'],d['col'](f),d['off'],d['keep']),pm_auc(d,d['col'](f))) for k,d in D.items()} for f in N16}
json.dump(res,open('auc_table.json','w'),indent=1)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,2,figsize=(12,6.5),sharey=True)
y=np.arange(len(N16))[::-1]; h=0.38
for ax,j,t in zip(axs,(0,1),('within-answer AUC (Omri\'s comparison)\nPB: prefix up to first error','position-matched first-error AUC\nerror at step j vs still-clean steps at j')):
    ax.barh(y+h/2,[res[f]['PB'][j] for f in N16],h,color=B,label='ProcessBench',edgecolor='white',linewidth=1)
    ax.barh(y-h/2,[res[f]['PRMB'][j] for f in N16],h,color=O,label='PRMBench',edgecolor='white',linewidth=1)
    ax.axvline(.5,color='#999',lw=1,ls='--'); ax.set_xlim(.3,.9); ax.set_title(t,fontsize=10.5,loc='left'); ax.grid(axis='x',color='#eee'); ax.set_axisbelow(True)
axs[0].set_yticks(y); axs[0].set_yticklabels(N16)
for lab in axs[0].get_yticklabels():
    if lab.get_text() in T3: lab.set_fontweight('bold')
axs[1].legend(loc='upper left',bbox_to_anchor=(0.62,1.13),frameon=False,ncol=1)
fig.suptitle('TELE16 single-feature AUC vs step error, standalone Qwen2.5-Math-7B (bold = the 3 features)',x=.01,ha='left',fontsize=12)
fig.tight_layout(); fig.savefig('/mnt/project-files/realized_z_stability_2026-11-28/fig1_feature_auc.png',dpi=150)
# fig2 position trend
F4=['chosen_surprisal','realized_drv','realized_z','q15_H1']
fig,axs=plt.subplots(1,4,figsize=(13,3.6),sharey=True)
for ax,f in zip(axs,F4):
    for (k,d),c,lab in zip(D.items(),(B,O),('ProcessBench','PRMBench')):
        z=az(d['col'](f),d['off']); m=d['keep']&~d['err']; idx=d['idx']
        xs=np.arange(10); ys=[z[m&(idx==j)].mean() for j in xs]
        ax.plot(xs,ys,color=c,lw=2,marker='o',ms=4,label=lab)
    ax.axhline(0,color='#999',lw=1); ax.set_title(f,fontsize=10.5,fontweight='bold' if f in T3 else 'normal'); ax.set_xlabel('step index'); ax.grid(color='#eee')
axs[0].set_ylabel('mean answer-z value\n(correct steps only)'); axs[0].legend(frameon=False)
fig.suptitle('Correct steps only: chosen_surprisal and realized_drv start much higher at step 0-1 on PB; realized_z has the same shape on both',x=.01,ha='left',fontsize=12)
fig.tight_layout(); fig.savefig('/mnt/project-files/realized_z_stability_2026-11-28/fig2_position_trend.png',dpi=150)
