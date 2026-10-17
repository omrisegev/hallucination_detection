"""Post-freeze comparator forensics, always using the same final recipe arrays."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from spectral_utils.monday_fusion_protocol import ROOT,WORK,dump,sha
from spectral_utils.monday_fusion_figures import save
from spectral_utils.monday_fusion_metrics import evaluate,top_flags

def run_competitors(data,lock_path):
    lock=json.loads(lock_path.read_text(encoding='utf8'))
    if lock.get('status')!='FROZEN' or lock.get('decision_chosen_by')!='Omri':
        raise RuntimeError('Competitor analysis requires the completed recipe and Omri-selected frozen decision')
    scores_path=ROOT/lock['step_scores']
    if sha(scores_path)!=lock['step_scores_sha256']: raise RuntimeError('Final score seal mismatch')
    z=np.load(scores_path); p=data.pop
    if not np.array_equal(z['offsets'],p.off): raise RuntimeError('Final score population mismatch')
    out=ROOT/'results/monday_combined_recipe_v1/run_20261003'; out.mkdir(parents=True,exist_ok=True)
    source=ROOT/'.worktrees/readout-quickest-detection-v1/results/step_evidence_v1/INPUT_FREEZE.json'
    frozen=json.loads(source.read_text(encoding='utf-8-sig')); mtg_path=frozen['mindgap']['path']
    if sha(mtg_path)!=frozen['mindgap']['sha256']: raise RuntimeError('Mind the Gap source changed')
    with np.load(mtg_path) as mtg: mtg_score=mtg['step_m5'].astype(float)
    if mtg_score.shape!=(p.total,): raise RuntimeError('Mind the Gap step alignment shape mismatch')
    dump(out/'COMPARATOR_MANIFEST.json',{'final_lock':str(lock_path),'final_lock_sha256':sha(lock_path),
         'final_scores_sha256':sha(scores_path),'prm_metadata':data.manifest['inputs']['prm_metadata'],
         'mind_the_gap':frozen['mindgap'],'mind_the_gap_recipe':'Shannon / paper EMA span5 / later-token attribution / top5 drop mean / native argmax',
         'scope':'Cached supervised PRM and our teacher-forced Mind the Gap reproduction; not paper headline numbers.'})
    overlap=[]; pb_rows=[]; prm_metrics=[]
    for bk in lock['banks']:
        score=z[bk+'__score']; flags=z[bk+'__flags'].astype(bool)
        if not np.isfinite(score).all(): raise RuntimeError('Incomplete final recipe scores: '+bk)
        for i in np.flatnonzero(p.prm):
            a,b=p.off[i:i+2]; meta=p.meta[p.ids[i]]; gold=p.labels[a:b]
            critic=np.asarray(meta['labels'])==0; reward=np.asarray(meta['rewards'],float)
            if critic.shape!=gold.shape or reward.shape!=gold.shape or not np.isfinite(reward).all():
                raise RuntimeError('Incomplete PRM comparison: '+p.ids[i])
            ours=flags[a:b]
            overlap.append({'bank':bk,'answer':p.ids[i],'source_group':p.groups[i],'classification':meta['classification'],
                'category':meta['category'],'n_errors':int(gold.sum()),'n_steps':b-a,
                'both':int((gold&ours&critic).sum()),'only_ours':int((gold&ours&~critic).sum()),
                'only_PRM':int((gold&~ours&critic).sum()),'neither':int((gold&~ours&~critic).sum()),
                'ours_clean_flag':bool(ours.any()) if not gold.any() else None,
                'PRM_clean_flag':bool(critic.any()) if not gold.any() else None})
        # Each category carries native and all three top-k rows, with error-count distribution.
        for category in sorted(set(p.meta[p.ids[i]]['classification'] for i in np.flatnonzero(p.prm))):
            ans=np.array([i for i in np.flatnonzero(p.prm) if p.meta[p.ids[i]]['classification']==category])
            rows=p.rows(np.isin(np.arange(p.n),ans)); off=np.r_[0,np.cumsum(p.ns[ans])]
            bank={'answers':ans,'rows':rows,'off':off,'V':np.empty((len(rows),0))}
            critic_score=np.concatenate([-np.asarray(p.meta[p.ids[i]]['rewards'],float) for i in ans])
            critic_flags=np.concatenate([np.asarray(p.meta[p.ids[i]]['labels'])==0 for i in ans])
            for name,s,f in [('ours',score[rows],flags[rows]),('supervised_PRM',critic_score,critic_flags)]:
                for rule in ['NATIVE','TOP_1','TOP_2','TOP_3']:
                    ff=f if rule=='NATIVE' else top_flags(s,off,np.full(len(ans),int(rule[-1])))
                    metrics,_=evaluate(p,bank,s,ff)
                    prm_metrics.append({'bank':bk,'classification':category,'method':name,'rule':rule,'top2_post_hoc':rule=='TOP_2',**metrics})
        for i in np.flatnonzero(p.pb):
            a,b=p.off[i:i+2]; target=int(p.target[i]); ns=b-a
            own_peak=int(np.argmax(score[a:b])); mtg_peak=int(np.argmax(mtg_score[a:b]))
            own_first=int(np.flatnonzero(flags[a:b])[0]) if flags[a:b].any() else -1
            length='1-5' if ns<=5 else '6-10' if ns<=10 else '11-20' if ns<=20 else '21+'
            pos='clean' if target<0 else str(min(3,int(4*target/max(ns-1,1))))
            pb_rows.append({'bank':bk,'cell':p.cells[i],'answer':p.ids[i],'source_group':p.groups[i],
                'target':target,'n_steps':ns,'length_bin':length,'error_position_quartile':pos,
                'ours_argmax_hit':own_peak==target if target>=0 else None,'ours_decision_hit':own_first==target,
                'MTG_native_argmax_hit':mtg_peak==target if target>=0 else None,
                'ours_miss':('no_flag' if own_first<0 else 'early' if own_first<target else 'late' if own_first>target else 'hit') if target>=0 else None,
                'MTG_miss':('early' if mtg_peak<target else 'late' if mtg_peak>target else 'hit') if target>=0 else None,
                'ours_clean_flag':own_first>=0 if target<0 else None,'MTG_native_clean_flag':True if target<0 else None})
    ov=pd.DataFrame(overlap); pb=pd.DataFrame(pb_rows)
    ov.to_csv(out/'PRM_OVERLAP.csv',index=False); pd.DataFrame(prm_metrics).to_csv(out/'PRM_CATEGORY_METRICS.csv',index=False)
    pb.to_csv(out/'MTG_FIRST_ERROR_FORENSICS.csv',index=False)
    figures=out/'figures'; figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,4,figsize=(20,10))
    for bk,ax in zip(lock['banks'],axes.flat):
        ov[ov.bank==bk].groupby('classification')[['both','only_ours','only_PRM','neither']].sum().plot.bar(stacked=True,ax=ax)
        ax.set_title(bk); ax.set_ylabel('Annotated error steps'); ax.tick_params(axis='x',rotation=35)
    fig.tight_layout(); save(fig,figures/'F11_PRM_error_overlap')
    fig,axes=plt.subplots(2,2,figsize=(15,11))
    for key,ax in zip(['error_position_quartile','length_bin'],axes[0]):
        pb[pb.target>=0].groupby(key)[['ours_argmax_hit','ours_decision_hit','MTG_native_argmax_hit']].mean().plot.bar(ax=ax)
        ax.set_ylabel('First-error hit rate (descriptive)')
    miss=pd.concat([pb[pb.target>=0].ours_miss.value_counts().rename('ours'),pb[pb.target>=0].MTG_miss.value_counts().rename('MTG')],axis=1).fillna(0)
    miss.reindex(['early','late','no_flag','hit']).fillna(0).plot.bar(ax=axes[1,0]); axes[1,0].set_ylabel('Erroneous model-answer records across banks')
    pb[pb.target<0].groupby('cell')[['ours_clean_flag','MTG_native_clean_flag']].mean().plot.bar(ax=axes[1,1]); axes[1,1].set_ylabel('Clean-answer flag rate')
    fig.suptitle('Same final recipe; MTG native argmax has no abstention rule')
    fig.tight_layout(); save(fig,figures/'F12_MTG_first_error')
    dump(out/'COMPETITOR_STATUS.json',{'status':'COMPUTED_PENDING_CLAUDE_REVIEW','prm_answer_bank_rows':len(ov),
         'pb_answer_bank_rows':len(pb),'recipe_lock_sha256':sha(lock_path),'deck_eligible':False})
    from spectral_utils.monday_fusion_figures import render
    from spectral_utils.monday_fusion_report import report
    from spectral_utils.monday_fusion_protocol import RUNS
    for experiment in RUNS: render(experiment); report(experiment)
