"""Corrected feature anatomy, conditional dependence and MoM quota diagnostics."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import csv,importlib.util,json,sys,time
import numpy as np
from scipy.stats import spearmanr
OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parents[1];sys.path.insert(0,str(ROOT))
from spectral_utils.advisor_statistics import answer_auc,source_bootstrap,posterior_from_mom
from spectral_utils.advisor_lr import fit_labelfree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
CODE=OUT/'sources/research/results/monday_fusion_analysis_v1/code/scripts/experiments';sys.path.insert(0,str(CODE))
import er_stage_a as SA,er_stage_b as SB
SA.CVF=CODE
from prepare_population import answer_std

def clean(x):
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v) for v in x]
    return x
def dump(p,x):p.write_text(json.dumps(clean(x),indent=2)+'\n')
def csvwrite(p,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',newline='') as f:w=csv.DictWriter(f,keys);w.writeheader();w.writerows(clean(rows))
def main():
    t=time.time();dest=OUT/'statistics';dest.mkdir(exist_ok=True);I=OUT/'inputs';p=np.load(I/'population.npz');off=p['offsets'];y=p['y'];groups=p['group'];cls=p['classification'];n=np.diff(off);ai=np.repeat(np.arange(len(n)),n)
    b=np.load(I/'bank_V_N56.npz');V=b['V'];names=b['columns'].astype(str);n16=list(np.load(I/'bank_V_N16.npz')['columns'].astype(str));level=V[:,[list(names).index(c) for c in ['q15_H1','q15_VE1','logprob_margin','true_tail50','energy_level']]].mean(1)
    corr=np.array([np.corrcoef(V[:,j],level)[0,1] if V[:,j].std()>0 else 0 for j in range(56)]);sg=np.where(corr<0,-1,1);score=V*sg
    pos=np.concatenate([(np.arange(k)+.5)/k for k in n]);bins=np.minimum((pos*5).astype(int),4)
    auc,pairs=answer_auc(y,np.column_stack([score,pos]),off);controlled,cpairs=answer_auc(y,np.column_stack([score,pos]),off,bins)
    ng=len(set(groups));weights=np.random.default_rng(20261011).multinomial(ng,np.full(ng,1/ng),size=5000).astype(float)
    rows=[];types=sorted(set(cls)-{'correct','multi_solutions'})
    for typ in ['ALL']+types:
        mask=np.ones(len(n),bool) if typ=='ALL' else cls==typ
        boot=source_bootstrap(auc,groups,mask,weights);cboot=source_bootstrap(controlled,groups,mask,weights)
        for j,name in enumerate(list(names)+['STEP_INDEX']):
            sel=mask&np.isfinite(auc[:,j]);cs=mask&np.isfinite(controlled[:,j]);lo,hi=np.nanquantile(boot[:,j],[.025,.975]);cl,ch=np.nanquantile(cboot[:,j],[.025,.975])
            rows.append({'type':typ,'channel':name,'in_N16':name in n16,'lf_sign':int(sg[j]) if j<56 else 1,'corr_with_level':float(corr[j]) if j<56 else None,'auc':float(np.nanmean(auc[sel,j])),'lo95':lo,'hi95':hi,'n_answers':int(sel.sum()),'position_controlled_auc':float(np.nanmean(controlled[cs,j])),'position_lo95':cl,'position_hi95':ch,'n_position_answers':int(cs.sum()),'n_position_pairs':int(cpairs[cs].sum())})
        print('channelstats',typ,flush=True)
    csvwrite(dest/'CHANNEL_BY_TYPE.csv',rows);np.savez_compressed(dest/'CHANNEL_PER_ANSWER.npz',idx=p['idx'],columns=np.concatenate([names,np.array(['STEP_INDEX'])]),auc=auc,position_auc=controlled,position_pairs=cpairs,lf_sign=sg)
    # All channels and both signs are retained; orientation never uses labels.
    X=np.load(I/'bank_V_N16.npz')['V'];center=SB.answer_center(X,off)
    ce=np.corrcoef(center[y],rowvar=False);cv=np.corrcoef(center[~y],rowvar=False)
    np.savez_compressed(dest/'CONDITIONAL_CORRELATIONS.npz',names=np.array(n16),error=ce,valid=cv,difference=ce-cv)
    model=fit_labelfree(X,'LSML_CONT');raw=model.predict(X);frozen=np.load(I/'bank_N16.npz')['S__LSML_CONT'];deviation=float(np.max(np.abs(answer_std(raw[:,None],off).ravel()-frozen)));assert deviation<1e-9
    group_scores=np.column_stack([X[:,q['columns']]@np.array(q['weights']) for q in model.diagnostic['group_weights']]);group_scores=answer_std(group_scores,off)
    marks={'channels':SB.random_tie_marks(X,off,.2,np.random.default_rng(20260928).random(X.shape)),'groups':SB.random_tie_marks(group_scores,off,.2,np.random.default_rng(20260929).random(group_scores.shape))}
    estimates=[];estimator_status={};posterior=None
    for scope,M in marks.items():
        truth=SA.truth(M,y)
        centered=SB.answer_center(M,off);ec=np.corrcoef(centered[y],rowvar=False);vc=np.corrcoef(centered[~y],rowvar=False)
        np.savez_compressed(dest/f'{scope}_VOTE_CONDITIONAL.npz',error=ec,valid=vc,difference=ec-vc,votes=M)
        for method in ['MOM','DS']:
            try:
                est=SA.tensor_mom_estimate(M,0 if scope=='channels' else None) if method=='MOM' else SA.em_estimate(M,'ds')
                rho=float(spearmanr(est['pi'],truth['pi']).statistic)
                summary={'status':'PASS','scope':scope,'estimator':method,'prevalence_hat':est['prevalence'],'prevalence_true':float(y.mean()),'prevalence_error':est['prevalence']-float(y.mean()),'reliability_spearman':rho,'psi_MAE':float(np.mean(np.abs(est['psi']-truth['psi']))),'eta_MAE':float(np.mean(np.abs(est['eta']-truth['eta']))),'out_of_range':est.get('out_of_range',0),'n_votes':M.shape[1]}
                estimates.append(summary);estimator_status[scope+'/'+method]={'summary':summary,'estimate':est,'truth':truth}
                if scope=='groups' and method=='MOM':posterior=posterior_from_mom(M,est)
            except Exception as exc:estimator_status[scope+'/'+method]={'status':'FAILED','reason':repr(exc)}
    csvwrite(dest/'ESTIMATOR_ACCURACY.csv',estimates);dump(dest/'ESTIMATORS.json',estimator_status)
    quota=[];qflags={};quota_counts=None
    if posterior is not None:
        quota_counts=np.floor(np.add.reduceat(posterior,off[:-1])+.5).astype(int)
        z=np.load(I/'bank_N16.npz')
        for method in ['LSML_CONT','IU_PCR']:
            s=z['S__'+method];flags=np.zeros(len(y),bool)
            for a,b,k in zip(off[:-1],off[1:],quota_counts):flags[a+np.argsort(-s[a:b],kind='stable')[:k]]=True
            qflags[method]=flags
            for typ in ['ALL_NONCONTROL']+sorted(set(cls)):
                am=cls!='correct' if typ=='ALL_NONCONTROL' else cls==typ;sm=am[ai];yy=y[sm];ff=flags[sm]
                tp=int((yy&ff).sum());fp=int((~yy&ff).sum());fn=int((yy&~ff).sum());tn=int((~yy&~ff).sum());fe=2*tp/max(2*tp+fp+fn,1);fv=2*tn/max(2*tn+fp+fn,1)
                quota.append({'method':method,'rule':'GROUP_MOM_POSTERIOR_COUNT_DIAGNOSTIC','type':typ,'n_answers':int(am.sum()),'n_steps':int(sm.sum()),'prm_score':fv if typ=='multi_solutions' else (fe+fv)/2 if typ!='correct' else None,'f1_valid':fv,'f1_error':fe,'flag_rate':float(ff.mean()),'answer_flag_rate':float((quota_counts[am]>0).mean()),'controls_falseflag_rate':float((quota_counts[am]>0).mean()) if typ=='correct' else None,'count_MAE':float(np.abs(quota_counts[am]-np.add.reduceat(y.astype(int),off[:-1])[am]).mean())})
        np.savez_compressed(dest/'MOM_POSTERIOR_QUOTA.npz',idx=p['idx'],posterior=posterior,count=quota_counts,**qflags)
    csvwrite(dest/'MOM_QUOTA.csv',quota)
    # Scientific figures are saved together with the plotted values.
    fig,axs=plt.subplots(1,3,figsize=(16,5))
    for ax,m,title in zip(axs,[ce,cv,ce-cv],['Error steps','Valid steps','Error − valid']):
        im=ax.imshow(m,cmap='RdBu_r',vmin=-1,vmax=1);ax.set_title(title);ax.set_xticks(range(16),n16,rotation=90,fontsize=6);ax.set_yticks(range(16),n16,fontsize=6)
    fig.colorbar(im,ax=axs,shrink=.5);fig.subplots_adjust(bottom=.35,left=.16,wspace=.5)
    for ext in ['svg','png']:fig.savefig(dest/f'conditional_correlations.{ext}',dpi=170)
    plt.close(fig)
    mat=np.array([[next(r['auc'] for r in rows if r['channel']==c and r['type']==t) for t in types] for c in list(names)+['STEP_INDEX']])
    fig,ax=plt.subplots(figsize=(11,15));im=ax.imshow(mat,cmap='viridis',vmin=.3,vmax=.9,aspect='auto');ax.set_xticks(range(len(types)),types,rotation=35,ha='right');ax.set_yticks(range(57),[('* ' if c in n16 else '')+c for c in list(names)+['STEP_INDEX']],fontsize=7);fig.colorbar(im,ax=ax,label='Within-answer AUC');ax.set_title('Corrected 56 channels × error type · * belongs to N16');fig.tight_layout()
    for ext in ['svg','png']:fig.savefig(dest/f'channel_by_type.{ext}',dpi=170)
    plt.close(fig)
    dump(dest/'STATUS.json',{'status':'COMPLETE','n_answers':len(n),'n_steps':len(y),'n_groups':ng,'canonical_LSML_replay_max_abs':deviation,'bootstrap_draws':5000,'seed':20261011,'orientation_rule':'PartB Pearsonwith5LEVEL answer-zmean; signfixedwithoutlabels','conditional_correlation':'afteranswercentering; classconditioningdiagnosticonly, notindependenceproof','quota_status':'COMPLETE' if posterior is not None else 'FAILED','quota_model':'N16continuousLSMLvirtualclassifiers, answer-z, topceil(.2*n)votes randomties20260929; MoM; roundposteriorerrorsum; canonicalclippingreported','runtime_s':time.time()-t})
    print('STATISTICS_COMPLETE',time.time()-t,flush=True)
if __name__=='__main__':main()
