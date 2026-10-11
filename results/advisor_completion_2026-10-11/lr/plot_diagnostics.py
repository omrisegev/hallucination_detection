"""Additional task-7 figures generated only from archived CSV material."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
FIG=OUT/'figures'
COLORS={'LR':'#805fa6','LSML':'#1968b4','IU':'#168f80'}
BANKS=['N11','N13','N22','N40','N53','N14','N16','N25','N43','N56']
FAMILY_LABELS={'F1':'Uncertainty','F2':'Surprise','F3':'Changes','F4':'Within-step shape',
               'F5':'Rhythm','F6':'Variability / drift','F7':'Digits'}

def read(name):
    with (OUT/name).open() as f:return list(csv.DictReader(f))

def save(fig,name):
    fig.tight_layout()
    for suffix in ('.svg','.png'):
        fig.savefig(FIG/(name+suffix),bbox_inches='tight',dpi=170)
    plt.close(fig)

def main():
    FIG.mkdir(exist_ok=True)
    weights=read('WEIGHT_STABILITY.csv')
    for bank in ('N16','N56'):
        rows=[r for r in weights if r['bank']==bank]
        names=[r['channel'] for r in rows if r['method']=='LR']
        lookup={(r['method'],r['channel']):float(r['mean_weight']) for r in rows}
        fig,axes=plt.subplots(1,2,figsize=(12,5))
        for ax,method in zip(axes,('LSML','IU')):
            color=COLORS[method]
            available=[n for n in names if (method,n) in lookup]
            lr=np.array([lookup['LR',n] for n in available]);lf=np.array([lookup[method,n] for n in available])
            ax.scatter(lr,lf,s=40,color=color)
            lim=max(float(np.max(np.abs(lr))),float(np.max(np.abs(lf))))*1.12
            ax.plot([-lim,lim],[-lim,lim],color='#aaa',ls='--',lw=1)
            ax.axhline(0,color='#ddd');ax.axvline(0,color='#ddd')
            ax.set_xlabel('LR common-scale normalized weight')
            ax.set_ylabel(f'{method} common-scale normalized weight')
            ax.set_title(f'{bank}: signed contributions, fold means')
            # Three callouts remain legible; the full channel labels are in
            # the companion bar chart/CSV rather than overlapping here.
            chosen=np.argsort(np.abs(lr-lf))[-min(3,len(lr)):]
            for rank,j in enumerate(chosen):
                if abs(lr[j])>lim*.5:
                    offset=(-5,10) if lr[j]>0 else (5,10)
                    ha='right' if lr[j]>0 else 'left'
                else:
                    offset=(12,35);ha='left'
                ax.annotate(available[j],(lr[j],lf[j]),fontsize=7,xytext=offset,
                    ha=ha,textcoords='offset points',arrowprops={'arrowstyle':'-','lw':.6,'color':'#888'})
        save(fig,'scatter_'+bank)
    families=read('FAMILY_MASS.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),sharey=True)
    for ax,bank in zip(axes,('N16','N56')):
        rs=[r for r in families if r['bank']==bank];names=sorted(set(r['family'] for r in rs));x=np.arange(len(names))
        for j,method in enumerate(('LR','LSML','IU')):
            mass=[next((float(r['mean_abs_weight_mass']) for r in rs if r['method']==method and r['family']==f),0) for f in names]
            ax.bar(x+(j-1)*.25,mass,.25,label=method,color=COLORS[method])
        ax.set_xticks(x,[f+': '+FAMILY_LABELS.get(f,f) for f in names],rotation=25,ha='right',fontsize=8)
        ax.set_title(bank);ax.set_xlabel('Channel family');ax.legend()
    axes[0].set_ylabel('Mean sum of absolute normalized weights')
    save(fig,'family_weight_mass')
    ablation=read('ABLATIONS.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,5.3),sharex=True)
    for ax,bank in zip(axes,('N16','N56')):
        rs=[r for r in ablation if r['bank']==bank and r['probe'].startswith('drop_family_')]
        vals=np.array([float(r['within_auc_delta']) for r in rs]);low=np.array([float(r['lo95']) for r in rs]);high=np.array([float(r['hi95']) for r in rs])
        ax.errorbar(vals,np.arange(len(rs)),xerr=np.vstack([vals-low,high-vals]),fmt='o',color=COLORS['LR'],capsize=3)
        ax.set_yticks(np.arange(len(rs)),[r['probe'].replace('drop_family_','')+': '+FAMILY_LABELS.get(r['probe'].replace('drop_family_',''),'') for r in rs])
        ax.axvline(0,color='#999',lw=1);ax.set_title(bank);ax.set_xlabel('Within-answer AUC change versus full LR')
        ax.grid(axis='x',alpha=.2)
    fig.suptitle('Family omission: source-group paired descriptive 95% intervals')
    save(fig,'family_ablation')
    four=read('FOURWAY_SUMMARY.csv')
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),sharey=True)
    for ax,method in zip(axes,('LSML','IU')):
        rs=[r for r in four if r['bank']=='N16' and r['method']==method and r['axis']=='classification']
        y=np.arange(len(rs));bottom=np.zeros(len(rs))
        for outcome,color in zip(('ours_only','lr_only','both','neither'),(COLORS[method],COLORS['LR'],'#5e9f80','#c7c7c7')):
            vals=np.array([int(r[outcome])/int(r['n']) for r in rs])
            ax.barh(y,vals,left=bottom,color=color,label=outcome.replace('_',' '));bottom+=vals
        ax.set_yticks(y,[r['value'] for r in rs],fontsize=8)
        ax.set_xlim(0,1);ax.set_title('N16 '+method+' versus new source-group OOF LR');ax.set_xlabel('Fraction of error-bearing answers')
    axes[1].legend(bbox_to_anchor=(1.02,1),loc='upper left')
    save(fig,'fourway_by_type')
    # Rebuild the primary figures from archived tables using the report's
    # shared palette, after the underlying scorer's legacy plotting step.
    metrics=read('METRICS.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),sharey=True)
    for ax,banks in zip(axes,[BANKS[:5],BANKS[5:]]):
        for method,arm,label in [('LR','LR_FULL','LR'),('LSML','LSML_FULL','L-SML'),('IU','IU_FULL','IU-PCR')]:
            values=[next((float(r['within_answer_auc']) for r in metrics if r['bank']==b and r['arm']==arm and r['complete']=='True'),np.nan) for b in banks]
            ax.plot(range(5),values,'o-',label=label,color=COLORS[method])
        ax.set_xticks(range(5),banks);ax.set_xlabel('Feature bank');ax.grid(alpha=.2);ax.legend()
    axes[0].set_ylabel('Mean within-answer AUC')
    fig.suptitle('New source-group OOF reference: more features versus fusion method')
    save(fig,'bank_gap')
    for bank in ('N16','N56'):
        rs=[r for r in weights if r['bank']==bank];names=[r['channel'] for r in rs if r['method']=='LR'];x=np.arange(len(names))
        fig,ax=plt.subplots(figsize=(13,4.5 if bank=='N16' else 7))
        for j,method,label in [(0,'LR','LR'),(1,'LSML','L-SML'),(2,'IU','IU-PCR')]:
            vals=[next(float(r['mean_weight']) for r in rs if r['method']==method and r['channel']==name) for name in names]
            ax.bar(x+(j-1)*.25,vals,.25,label=label,color=COLORS[method])
        ax.set_xticks(x,names,rotation=70 if bank=='N16' else 90,ha='right',fontsize=8)
        ax.set_ylabel('Signed contribution per training SD / sum |weights|')
        ax.set_title(f'{bank}: common-scale weights; means across source-group folds')
        ax.legend();ax.axhline(0,color='#777',lw=.8)
        save(fig,'weights_'+bank)
    fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
    arms=['LR_FULL','LR__LR_HALF','LR__LSML_HALF','LR__IU_HALF','LSML_FULL','LSML__LR_HALF','LSML__LSML_HALF','IU_FULL','IU__LR_HALF','IU__IU_HALF']
    labels=['LR / full','LR / LR half','LR / L-SML half','LR / IU half','L-SML / full','L-SML / LR half','L-SML / own half','IU-PCR / full','IU-PCR / LR half','IU-PCR / own half']
    for ax,bank in zip(axes,('N16','N56')):
        vals=[next((float(r['within_answer_auc']) for r in metrics if r['bank']==bank and r['arm']==arm and r['complete']=='True'),np.nan) for arm in arms]
        ax.barh(np.arange(len(arms)),vals,color=[COLORS['LR']]*4+[COLORS['LSML']]*3+[COLORS['IU']]*3)
        ax.set_yticks(np.arange(len(arms)),labels,fontsize=8);ax.set_xlim(.5,.9);ax.set_title(bank)
        ax.set_xlabel('Within-answer AUC');ax.grid(axis='x',alpha=.2)
    fig.suptitle('Matched half-bank crossover: selection versus weighting')
    save(fig,'crossover')

if __name__=='__main__':main()
