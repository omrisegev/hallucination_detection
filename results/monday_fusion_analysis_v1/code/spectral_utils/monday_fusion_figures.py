"""Standalone PNG/SVG research figures; unavailable panels are explicitly open."""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Rectangle
from spectral_utils.monday_fusion_protocol import ROOT,RUNS,dump,sha

BANKS=['B13','B16','B20','B23','B32','B35','B51','B54']
VARIANTS=['original','remove_2','remove_3','random_2','random_3']

def save(fig,path):
    fig.patch.set_facecolor('white')
    fig.savefig(path.with_suffix('.png'),dpi=160,facecolor='white',bbox_inches='tight')
    fig.savefig(path.with_suffix('.svg'),facecolor='white',bbox_inches='tight')
    plt.close(fig)

def pipeline(directory):
    stages=['Features','Answer\nnormalization','DS filter','Partition\n+ merge','Within\nweights','Between\nweights','Step\nscores','Decision\nrule']
    fig,ax=plt.subplots(figsize=(17,3.4)); ax.set(xlim=(-.2,16),ylim=(-1,2)); ax.axis('off')
    for j,label in enumerate(stages):
        x=2*j; color='#cce8f7' if j==0 else '#ffe4ab' if j in (4,5) else '#e8e8e8'
        ax.add_patch(FancyBboxPatch((x,.2),1.6,.9,boxstyle='round,pad=.06',facecolor=color,edgecolor='#333333'))
        ax.text(x+.8,.65,label,ha='center',va='center',fontsize=11)
        if j<7: ax.annotate('',xy=(x+1.96,.65),xytext=(x+1.68,.65),arrowprops={'arrowstyle':'->','color':'#333333'})
    ax.text(.8,1.6,'Experiment 1:\nreduce LEVEL',ha='center',fontsize=11)
    ax.text(9.9,-.5,'Experiment 2: change estimator votes / quota; matched continuous scores',ha='center',fontsize=11)
    ax.text(12.9,1.6,'Position and decision layers',ha='center',fontsize=11)
    save(fig,directory/'F1_pipeline')

def render(experiment):
    out=RUNS[experiment]; directory=out/'figures'; directory.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'text.color':'#202020','axes.labelcolor':'#202020','xtick.color':'#202020',
                         'ytick.color':'#202020','axes.facecolor':'white','figure.facecolor':'white','svg.fonttype':'none'})
    done=['F1']; opened={}; pipeline(directory)
    if not(out/'METRICS.csv').exists():
        dump(out/'FIGURE_STATUS.json',{'complete':done,'open':{f'F{i}':'No completed metrics' for i in range(2,13)}}); return
    m=pd.read_csv(out/'METRICS.csv')
    main=m[(m.cell=='prmbench_qwen3_8b')&(m.position_variant=='CONTENT')&(m.decision_rule=='ANSWER_Z_Q80')]
    if 'score_scope' in main: main=main[main.score_scope.isna()|(main.score_scope=='OOF')]
    variants=VARIANTS if experiment=='level_reduction_v1' else ['channels_answer','groups_answer','channels_cell','groups_cell']
    fig,axes=plt.subplots(2,4,figsize=(18,9),sharey=True)
    for bk,ax in zip(BANKS,axes.flat):
        for recipe,t in main[main.bank==bk].groupby('recipe'):
            lookup=t.set_index('variant')['within_answer_auc']
            ys=[lookup.get(v,np.nan) for v in variants]
            ax.plot(range(len(variants)),ys,'o',lw=2 if recipe in ('MEAN_EQ','LR_full') else 1,
                    linestyle='--' if recipe.startswith('LR_') else '-',label=recipe)
        ax.set_title(bk); ax.set_xticks(range(len(variants)),variants,rotation=35,ha='right'); ax.grid(alpha=.2)
        ax.set_ylabel('Within-answer AUC')
    handles,labels=axes.flat[0].get_legend_handles_labels(); fig.legend(handles,labels,loc='lower center',ncol=6,bbox_to_anchor=(.5,-.06))
    fig.tight_layout(); save(fig,directory/'F2_bank_performance'); done.append('F2')
    if(out/'CONTRASTS.csv').exists():
        c=pd.read_csv(out/'CONTRASTS.csv'); c=c[(c.status=='OK')&(c.comparison=='matched')&(c.endpoint=='within_answer_auc')]
        fig,axes=plt.subplots(3,3,figsize=(20,19),sharex=True)
        for bk,ax in zip(BANKS+['EQUAL_MEAN_8_BANKS'],axes.flat):
            t=c[c.bank==bk].reset_index(drop=True)
            labels=[str(r.variant)+' / '+str(r.recipe) for r in t.itertuples()]
            ax.hlines(np.arange(len(t)),t.lo95,t.hi95); ax.plot(t.delta,np.arange(len(t)),'o',ms=3)
            ax.axvline(0,color='#555555',lw=1); ax.set_yticks(range(len(t)),labels,fontsize=6); ax.set_title(bk); ax.invert_yaxis()
            ax.set_xlabel('AUC difference vs matched EQ (95% interval)')
        fig.tight_layout(); save(fig,directory/'F3_paired_contrasts'); done.append('F3')
    else: opened['F3']='Paired bootstrap incomplete'
    weight=[]
    for bk in BANKS:
        records=[]
        for var in variants:
            p=out/'chunks'/f'prmbench_qwen3_8b__{bk}__{var}.diagnostics.json'
            if p.exists(): records.append((var,json.loads(p.read_text(encoding='utf8'))))
        if not records: continue
        names=json.loads((out/'INPUT_MANIFEST.json').read_text(encoding='utf8'))['banks'][bk]
        matrix=np.full((len(names),len(variants)),-2.)
        outlines=[]
        for var,record in records:
            d=record['diagnostics']; col=variants.index(var); surviving=d.get('survivors',[]); groups=d.get('groups_after',[])
            for name in d.get('removed',[]): matrix[names.index(name),col]=-1
            for name,g in zip(surviving,groups): matrix[names.index(name),col]=g
            for name in d.get('MEAN_top_DS_group',{}).get('members',[]): outlines.append((names.index(name),col))
            for method,dd in d.items():
                if isinstance(dd,dict) and 'effective_groups' in dd:
                    weight.append({'bank':bk,'variant':var,'method':method,'effective_groups':dd['effective_groups'],'negative_weights':dd['negative_weights']})
        fig,ax=plt.subplots(figsize=(9,max(6,len(names)*.22)))
        ax.imshow(matrix,aspect='auto',cmap='tab20',vmin=-2,vmax=17,interpolation='nearest')
        for row,col in outlines: ax.add_patch(Rectangle((col-.48,row-.48),.96,.96,fill=False,edgecolor='black',lw=1.5))
        ax.set_xticks(range(len(variants)),variants,rotation=25,ha='right'); ax.set_yticks(range(len(names)),names,fontsize=8)
        ax.set_title('PRMBench / '+bk+': groups (outlined: top MEAN-within DS weight)\n-1 removed, -2 filtered; group IDs local to each fit')
        save(fig,directory/('F4_groups_'+bk))
    done.append('F4')
    if weight:
        w=pd.DataFrame(weight); w.to_csv(out/'WEIGHT_CONCENTRATION.csv',index=False)
        fig,axes=plt.subplots(2,4,figsize=(18,9),sharey=True)
        for bk,ax in zip(BANKS,axes.flat):
            for method,t in w[w.bank==bk].groupby('method'):
                lookup=t.set_index('variant'); ax.plot(range(len(variants)),[lookup.loc[v,'effective_groups'] if v in lookup.index else np.nan for v in variants],'o-',label=method)
            ax.set_title(bk); ax.set_xticks(range(len(variants)),variants,rotation=35,ha='right'); ax.set_ylabel('Effective groups: 1 / sum(p²), p = |w| / sum|w|'); ax.grid(alpha=.2)
        handles,labels=axes.flat[0].get_legend_handles_labels(); fig.legend(handles,labels,loc='lower center',ncol=8,bbox_to_anchor=(.5,-.05))
        fig.tight_layout(); save(fig,directory/'F5_weight_concentration'); done.append('F5')
        fig,axes=plt.subplots(2,4,figsize=(18,9),sharey=True)
        for bk,ax in zip(BANKS,axes.flat):
            for method,t in w[w.bank==bk].groupby('method'):
                lookup=t.set_index('variant'); ax.plot(range(len(variants)),[lookup.loc[v,'negative_weights'] if v in lookup.index else np.nan for v in variants],'o-',label=method)
            ax.set_title(bk); ax.set_xticks(range(len(variants)),variants,rotation=35,ha='right'); ax.set_ylabel('Number of negative group weights'); ax.grid(alpha=.2)
        handles,labels=axes.flat[0].get_legend_handles_labels(); fig.legend(handles,labels,loc='lower center',ncol=8,bbox_to_anchor=(.5,-.05))
        fig.tight_layout(); save(fig,directory/'F5_negative_weights')
    if(out/'ESTIMATOR_ACCURACY.csv').exists():
        samples=[]; marks=[]
        for p in (out/'chunks').glob('prmbench*.diagnostics.json'):
            cell,bk,var=p.name.removesuffix('.diagnostics.json').split('__'); d=json.loads(p.read_text(encoding='utf8'))['diagnostics']
            for within in ('MEAN','HEM'):
                for method,v in d.get(within+'_estimator_accuracy',{}).items():
                    if method not in ('DS','MOM'): continue
                    for field in ('psi','eta'):
                        samples.extend({'method':method,'input':var,'parameter':field,'truth':a,'estimated':b} for a,b in zip(v['truth'][field],v['estimated'][field]))
                for label,h in d.get(within+'_marks_distribution',{}).items():
                    marks.extend({'quota':var.split('_')[-1],'kind':label,'marks':int(k),'count':n} for k,n in h['histogram'].items())
        s=pd.DataFrame(samples); fig,axes=plt.subplots(1,3,figsize=(15,4))
        for field,ax in zip(['psi','eta'],axes):
            for (method,kind),t in s[s.parameter==field].groupby(['method','input']): ax.scatter(t.truth,t.estimated,s=8,alpha=.4,label=method+'/'+kind)
            ax.plot([0,1],[0,1],color='#555555'); ax.set(xlabel='True '+field,ylabel='Estimated '+field,xlim=(0,1)); ax.legend(fontsize=6)
        a=pd.read_csv(out/'ESTIMATOR_ACCURACY.csv'); a=a[a.estimator!='SML']; means=a.groupby(['estimator','vote_type','quota'])[['estimated_prevalence','true_prevalence']].mean()
        means.plot.bar(ax=axes[2],fontsize=7); axes[2].set_ylabel('Prevalence'); fig.tight_layout(); save(fig,directory/'F6_estimator_accuracy'); done.append('F6')
        h=pd.DataFrame(marks); fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True)
        for label,ax in zip(['clean','erroneous'],axes):
            for quota,t in h[h.kind==label].groupby('quota'):
                z=t.groupby('marks')['count'].sum(); ax.plot(z.index,z/z.sum(),'o-',label=quota)
            ax.set(title=label,xlabel='Marks per answer and vote',ylabel='Fraction'); ax.legend()
        fig.tight_layout(); save(fig,directory/'F7_marks_per_answer'); done.append('F7')
    else:
        opened['F6']='Estimator-input experiment only'; opened['F7']='Estimator-input experiment only'
    if(out/'POSITION_DECISION_COMPLETE.json').exists():
        fig,axes=plt.subplots(1,2,figsize=(21,7))
        for benchmark,field,ax in zip(['PRMBench','ProcessBench'],['within_answer_auc','pb_first_error_accuracy'],axes):
            mask=m.cell.str.startswith('prm') if benchmark=='PRMBench' else m.cell.str.startswith('pb_')
            focus=m[mask&(m.decision_rule=='ANSWER_Z_Q80')&(~m.recipe.str.startswith('LR_'))]
            summary=focus.groupby(['recipe','position_variant'])[field].agg(['mean','count'])
            summary.loc[summary['count']!=(len(variants)*8*(1 if benchmark=='PRMBench' else 8)),'mean']=np.nan
            summary['mean'].unstack().plot.bar(ax=ax); ax.set_title(benchmark+' (incomplete coverage left blank)'); ax.set_ylabel(field)
        fig.suptitle('Content, position and equal split: diagnostic comparison, not causal percentage shares')
        fig.tight_layout(); save(fig,directory/'F8_position'); done.append('F8')
        null_rows=[]
        for path in (out/'layers').glob('*.diagnostics.json'):
            ce,bk,var=path.name.removesuffix('.diagnostics.json').split('__')
            records=json.loads(path.read_text(encoding='utf8')).get('nulls',{})
            for key,v in records.items():
                recipe,pos=key.split('__'); observed=v.get('observed_within_auc',v.get('observed_first_error_accuracy'))
                null_rows.append({'benchmark':'PRMBench' if ce.startswith('prm') else 'ProcessBench','bank':bk,'cell':ce,'variant':var,
                                 'recipe':recipe,'position_variant':pos,'observed':observed,'same_length_singletons':v['same_length_singletons'],
                                 'shuffle':v['within_answer_shuffle']['mean'],'swap':v['whole_answer_same_length_swap']['mean']})
        if null_rows:
            null=pd.DataFrame(null_rows); null.to_csv(out/'POSITION_NULLS.csv',index=False)
            fig,axes=plt.subplots(1,2,figsize=(20,7))
            for benchmark,ax in zip(['PRMBench','ProcessBench'],axes):
                t=null[(null.benchmark==benchmark)&(null.position_variant=='CONTENT')]
                t.groupby('recipe')[['observed','shuffle','swap']].mean().plot.bar(ax=ax)
                ax.set_title(benchmark+' / content scores / label nulls'); ax.set_ylabel('Within-answer AUC' if benchmark=='PRMBench' else 'First-error accuracy')
            fig.tight_layout(); save(fig,directory/'F8_position_nulls')
        fig,axes=plt.subplots(2,2,figsize=(16,12))
        reference_variant='original' if experiment=='level_reduction_v1' else 'groups_answer'
        decision=m[(~m.recipe.str.startswith('LR_'))&(m.position_variant=='CONTENT')&(m.variant==reference_variant)].copy()
        for field,benchmark,ax in zip(['prmscore','pb_official_f1','clean_flag_rate','clean_flag_rate'],['PRMBench','ProcessBench','PRMBench','ProcessBench'],axes.flat):
            mask=decision.cell.str.startswith('prm') if benchmark=='PRMBench' else decision.cell.str.startswith('pb_')
            t=decision[mask].groupby(['recipe','decision_rule'])[field].agg(['mean','count'])
            t.loc[t['count']!=(8 if benchmark=='PRMBench' else 64),'mean']=np.nan
            table=t['mean'].unstack(); im=ax.imshow(table.to_numpy(),vmin=0,vmax=1,aspect='auto',cmap='viridis')
            ax.set_xticks(range(len(table.columns)),table.columns,rotation=35,ha='right'); ax.set_yticks(range(len(table.index)),table.index)
            ax.set_title(benchmark+' / '+field+' / '+reference_variant); fig.colorbar(im,ax=ax,fraction=.025)
        fig.tight_layout(); save(fig,directory/'F9_decisions'); done.append('F9')
    else: opened.update(F8='Position layers pending',F9='Decision layers pending')
    if(out/'LR_COMPLETE.json').exists():
        selected=json.loads((out/'SELECTION.json').read_text(encoding='utf8'))['winner']
        chosen_variant=selected['variant']
        if experiment=='estimator_input_v1' and chosen_variant=='original': chosen_variant='groups_answer'
        lr=m[(m.cell=='prmbench_qwen3_8b')&(m.decision_rule=='ANSWER_Z_Q80')&(m.variant==chosen_variant)]
        rows=[]
        for bk in BANKS:
            t=lr[lr.bank==bk]; us=t[(t.recipe==selected['recipe'])&(t.position_variant=='CONTENT')].within_answer_auc
            row={'bank':bk,'Selected content recipe (development)':float(us.iloc[0]) if len(us)==1 else np.nan}
            for pos,label in [('CONTENT','without POSZ'),('POS','with POSZ')]:
                for scope,title in [('OOF','LR OOF'),('IN_SAMPLE_DIAGNOSTIC','LR in sample')]:
                    values=t[(t.recipe=='LR_full')&(t.position_variant==pos)&(t.score_scope==scope)].within_answer_auc
                    row[title+' '+label]=float(values.iloc[0]) if len(values)==1 else np.nan
            rows.append(row)
        comparison=pd.DataFrame(rows).set_index('bank'); comparison.to_csv(out/'LR_GAP_REFERENCE.csv')
        fig,ax=plt.subplots(figsize=(14,6)); comparison.plot.bar(ax=ax); ax.set_ylabel('Within-answer AUC'); ax.set_title('Same selected feature variant: '+chosen_variant+'; LR is a supervised reference')
        fig.tight_layout(); save(fig,directory/'F10_LR_reference'); done.append('F10')
    else: opened['F10']='Nested OOF LR incomplete'
    referenced=[]; combined=ROOT/'results/monday_combined_recipe_v1/run_20261003'
    status=combined/'COMPETITOR_STATUS.json'; lock=combined/'LOCKED_RECIPE.json'
    if status.exists() and lock.exists() and json.loads(status.read_text(encoding='utf8')).get('recipe_lock_sha256')==sha(lock):
        done+=['F11','F12']
        referenced=[os.path.relpath(p,out).replace('\\','/') for p in sorted((combined/'figures').glob('F1[12]*.png'))]
    else: opened.update(F11='Final recipe and decision freeze required',F12='Final recipe and decision freeze required')
    dump(out/'FIGURE_STATUS.json',{'complete':done,'open':opened,'referenced_figures':referenced,'formats':['PNG','SVG'],'visual_review':'PENDING; opaque white figure backgrounds for light/dark hosts'})
