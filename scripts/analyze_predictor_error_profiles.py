"""Full-population descriptive profiles, without changing models or scores."""
from pathlib import Path
import sys,json,csv,time,html,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from scripts import run_temporal_research_baseline as base
from scripts.run_predictor_subset_study import sha,write,DATA
from spectral_utils.context_training import FeatureBundle
from spectral_utils.predictor_subset_fusion import METHODS,PREDICTORS

OUT=ROOT/'results/predictor_error_profiles_v1';SOURCE=ROOT/'results/predictor_subset_iu_v1'
FEATURES=['H0lim','VE0','VE0.75','VE1','H0lim innovation']
GROUPS=['Found by >=1 combination','Missed by all32; gate open','Gate closed']
COLORS=['#237c59','#c4403e','#67788a']
KEYS=['log2_tokens','error_start_fraction','error_step_fraction','feature_change_l2','predictor_disagreement','mean_signed_residual']
TITLES=['Answer length (log2 tokens)','First-error start / answer tokens','First-error step / answer tokens',
        'Feature change vs preceding16 tokens (L2)','Predictor disagreement at error step','Mean signed residual at error step']

def figure_save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=150);fig.savefig(OUT/(name+'.svg'))
    p=OUT/(name+'.svg');p.write_text('\n'.join(x.rstrip() for x in p.read_text(encoding='utf8').splitlines())+'\n',encoding='utf8')

def bootstrap(X,category,strata,groups):
    """Direct paired source-group bootstrap, including both model views."""
    levels,gi=np.unique(groups,return_inverse=True);ng=len(levels);k=X.shape[1]
    ns=int(strata.max())+1;finite=np.isfinite(X)
    counts=np.zeros((ng,2,ns,k));sums=np.zeros_like(counts)
    for c in range(2):
        for st in range(ns):
            for j in range(k):
                take=(category==c)&(strata==st)&finite[:,j]
                counts[:,c,st,j]=np.bincount(gi[take],minlength=ng)
                sums[:,c,st,j]=np.bincount(gi[take],weights=X[take,j],minlength=ng)
    C=counts.sum(0);S=sums.sum(0);support=(C[0]>=5)&(C[1]>=5)
    weights=np.where(support,2*C[0]*C[1]/np.maximum(C[0]+C[1],1),0)
    weights/=weights.sum(0)
    means=np.divide(S,C,out=np.zeros_like(S),where=C>0)
    raw=S.sum(1)/C.sum(1);matched=(weights[None,:,:]*means).sum(1)
    draws_raw=[];draws_matched=[];missing=0;rng=np.random.default_rng(39020260915)
    cm=counts.reshape(ng,-1);sm=sums.reshape(ng,-1)
    for start in range(0,10000,128):
        b=min(128,10000-start);W=rng.multinomial(ng,np.full(ng,1/ng),size=b).astype(float)
        cd=(W@cm).reshape(b,2,ns,k);sd=(W@sm).reshape(b,2,ns,k)
        rd=sd.sum(2)/cd.sum(2);draws_raw.append(rd[:,1]-rd[:,0])
        md=np.divide(sd,cd,out=np.zeros_like(sd),where=cd>0)
        available=(cd[:,0]>0)&(cd[:,1]>0);ww=weights[None,:,:]*available
        missing+=int(np.any((~available)&(weights[None,:,:]>0),axis=(1,2)).sum())
        adjusted=(ww[:,None,:,:]*md).sum(2)/ww.sum(1)[:,None,:]
        draws_matched.append(adjusted[:,1]-adjusted[:,0])
    rawdraw=np.concatenate(draws_raw);matchdraw=np.concatenate(draws_matched);q=[.05/12/2,1-.05/12/2]
    result={}
    for j,key in enumerate(KEYS):
        result[key]=dict(found_n=int(C[0,:,j].sum()),missed_n=int(C[1,:,j].sum()),
            found_mean=float(raw[0,j]),missed_mean=float(raw[1,j]),delta_missed_minus_found=float(raw[1,j]-raw[0,j]),
            ci=np.quantile(rawdraw[:,j],q).tolist(),matched_found_mean=float(matched[0,j]),matched_missed_mean=float(matched[1,j]),
            matched_delta=float(matched[1,j]-matched[0,j]),matched_ci=np.quantile(matchdraw[:,j],q).tolist(),
            retained_strata=int(support[:,j].sum()),matched_found_n=int(C[0,support[:,j],j].sum()),
            matched_missed_n=int(C[1,support[:,j],j].sum()))
    return dict(metrics=result,draws=10000,source_groups=ng,ci_level=1-.05/12,
        draws_with_any_empty_retained_stratum=missing,
        empty_stratum_rule='Fixed harmonic weights renormalized over populated retained strata in that draw; no fabricated means.')

def run():
    began=time.perf_counter();OUT.mkdir(exist_ok=True)
    write(OUT/'RUN_STATE.json',dict(status='ANALYZING_FROZEN_SCORES'))
    artifact=json.loads((SOURCE/'ARTIFACTS.json').read_text());sourceaudit=json.loads((SOURCE/'AUDIT.json').read_text())
    for name in ['RESIDUALS.npy','SCORES_FROZEN.npz']:
        if sha(SOURCE/name)!=artifact['files'][name]['sha256']:raise ValueError('Source hash changed '+name)
    bundle=FeatureBundle(DATA,'innovation5');records,j=base.load_contract(ROOT.parents[1])
    assert [r['uid'] for r in records]==[m['uid'] for m in bundle.metadata]
    assert sha(DATA/'features.npy')==bundle.manifest['files']['features.npy']
    with np.load(SOURCE/'SCORES_FROZEN.npz') as f:scores={n:f[n] for n in f.files}
    with np.load(ROOT/'results/temporal_research_baseline_v1/SCORES_FROZEN.npz') as f:gatepercent=f['gate_percentile']
    gate=gatepercent>=.33;offset=j['offsets'];target=j['target']
    pb=np.array([r['cell'].startswith('pb_') for r in records]);error=pb&(target>=0);ids=np.flatnonzero(error)
    peak={n:np.array([np.argmax(s[offset[i]:offset[i+1]]) for i in range(len(records))]) for n,s in scores.items()}
    hits={n:error&gate&(p==target) for n,p in peak.items()};found=np.logical_or.reduce([hits[n] for n in METHODS])
    cat=np.where(~gate,2,np.where(found,0,1));residual=np.load(SOURCE/'RESIDUALS.npy',mmap_mode='r')
    previous=json.loads((SOURCE/'ERROR_ANALYSIS.json').read_text())['overlap']
    counts=np.bincount(cat[ids],minlength=3);assert counts.tolist()==[1589,2032,821]
    assert counts[1]+counts[2]==previous['missed_by_all32']
    strict=error&~np.logical_or.reduce([hits[n] for n in list(METHODS)+['ridge','tcn__real','bocpd','noreset','mean16']])
    X=np.full((len(ids),6),np.nan);steps=np.empty(len(ids),int);rawrows=[]
    bins=np.arange(-64,129,4);profile=np.full((len(ids),len(bins)-1,10),np.nan)
    for row,i in enumerate(ids):
        m=bundle.metadata[i];a=m['offset'];n=m['tokens'];st=m['step_start']+int(target[i]);u,v=np.asarray(bundle.spans[st])-a
        raw=np.asarray(bundle.features[a:a+n],float)[:,bundle.columns];z=(raw-bundle.mean[i])/bundle.scale[i]
        r=np.asarray(residual[a:a+n]);assert len(z)==n and 0<=u<v<=n
        change=float(np.linalg.norm(z[u:min(v,u+16)].mean(0)-z[max(0,u-16):u].mean(0))) if u else np.nan
        # Std across predictions == std across residuals, common observed target.
        X[row]=[np.log2(n),u/n,(v-u)/n,change,r[u:v].std(1).mean(),r[u:v].mean()]
        steps[row]=m['step_stop']-m['step_start']
        for b,(lo,hi) in enumerate(zip(bins[:-1],bins[1:])):
            aa=max(0,int(u+lo));bb=min(n,int(u+hi))
            if bb>aa:profile[row,b]=np.r_[z[aa:bb].mean(0),r[aa:bb].mean(0)]
        rawrows.append(dict(uid=m['uid'],cell=m['cell'],group_id=m['group_id'],category=int(cat[i]),tokens=n,steps=int(steps[row]),
            first_error_step=int(target[i])+1,error_start=int(u),error_stop=int(v),gate_percentile=float(gatepercent[i]),
            **{k:None if not np.isfinite(x) else float(x) for k,x in zip(KEYS,X[row])}))
    cells=np.array([records[i]['cell'] for i in ids]);groups=np.array([records[i]['group_id'] for i in ids])
    quartiles=np.quantile(X[:,0],[.25,.5,.75]);lengthbin=np.searchsorted(quartiles,X[:,0],side='right')
    posbin=np.minimum((X[:,1]*3).astype(int),2);_,strata=np.unique(np.array([f'{c}/{l}/{p}' for c,l,p in zip(cells,lengthbin,posbin)]),return_inverse=True)
    print('[profiles] all4442 rows and event profiles; grouped bootstrap',flush=True)
    uncertainty=bootstrap(X,cat[ids],strata,groups)
    summaries={}
    for c in range(3):
        mask=cat[ids]==c;summaries[GROUPS[c]]=dict(n=int(mask.sum()),unique_sources=len(set(groups[mask])),
            median_tokens=float(np.median(2**X[mask,0])),median_steps=float(np.median(steps[mask])),
            diagnostics={k:dict(n=int(np.isfinite(X[mask,q]).sum()),median=float(np.nanmedian(X[mask,q])),
                q25=float(np.nanquantile(X[mask,q],.25)),q75=float(np.nanquantile(X[mask,q],.75))) for q,k in enumerate(KEYS)})
    with (OUT/'ANSWER_DIAGNOSTICS.csv').open('w',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rawrows[0]));writer.writeheader();writer.writerows(rawrows)
    # Matched illustrative pair: same highest-overlap cell, nearby covariates.
    cell=sorted(set(cells),key=lambda c:(-min(np.sum((cells==c)&(cat[ids]==0)),np.sum((cells==c)&(cat[ids]==1))),c))[0]
    pool=np.flatnonzero((cells==cell)&(cat[ids]==1));coord=X[:,[0,1,2]];scale=np.std(coord,0)+1e-12
    center=np.median(coord[pool],0);mi=min(pool,key=lambda k:(float(np.sum(((coord[k]-center)/scale)**2)),records[ids[k]]['uid']))
    pool=np.flatnonzero((cells==cell)&(cat[ids]==0)&(groups!=groups[mi]));ci=min(pool,key=lambda k:(float(np.sum(((coord[k]-coord[mi])/scale)**2)),records[ids[k]]['uid']))
    selected=[('matched_missed',int(ids[mi])),('matched_found',int(ids[ci]))]
    for label,rid in [('watermelon','gsm8k::gsm8k-142'),('gate_closed_locker','gsm8k::gsm8k-290')]:
        selected.append((label,next(i for i,r in enumerate(records) if r['cell']=='pb_gsm8k_q4' and r['row_id']==rid)))
    # Plotting: all samples retained in histograms and every token in traces.
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for q,ax in enumerate(axes.flat):
        data=X[:,q];finite=data[np.isfinite(data)];edges=np.linspace(finite.min(),finite.max(),35)
        for c in range(3):
            v=data[(cat[ids]==c)&np.isfinite(data)];ax.hist(v,bins=edges,weights=np.ones(len(v))/len(v),histtype='step',lw=1.8,color=COLORS[c],label=GROUPS[c])
        ax.set(title=TITLES[q],ylabel='Fraction of group / bin');ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=8);fig.suptitle('All PB error answers: descriptive distributions (each answer weighted equally)');fig.tight_layout();figure_save(fig,'HISTOGRAMS');plt.close(fig)
    fig,axes=plt.subplots(2,5,figsize=(18,8),sharex=True);coverage={};profilemeans={}
    for c in range(3):
        pp=profile[cat[ids]==c];num=np.isfinite(pp).sum(0);total=np.nansum(pp,axis=0)
        avg=np.divide(total,num,out=np.full_like(total,np.nan),where=num>0);coverage[GROUPS[c]]=num[:,0].tolist();profilemeans[GROUPS[c]]=avg.tolist()
        for q,ax in enumerate(axes.flat):
            y=avg[:,q].copy();y[num[:,q]<30]=np.nan;ax.plot((bins[:-1]+bins[1:])/2,y,color=COLORS[c],label=GROUPS[c])
    for q,ax in enumerate(axes.flat):
        ax.axvline(0,color='black',ls='--',lw=1);ax.axhline(0,color='gray',lw=.6);ax.grid(alpha=.15)
        ax.set_title(FEATURES[q] if q<5 else PREDICTORS[q-5]+' signed residual')
        ax.set_ylabel('Answer-standardized feature' if q<5 else 'Mean standardized feature error')
        if q>=5:ax.set_xlabel('Tokens from annotated step start')
    axes[0,0].legend(fontsize=7);fig.suptitle('Mean profiles around the FIRST annotated error step; token error location is unknown\n4-token bins; available answers only; no zero padding or causal interpretation');fig.tight_layout();figure_save(fig,'ALIGNED_PROFILES');plt.close(fig)
    specs={c:(p,k,ds) for c,p,k,ds in base.evaluator.source_specs()};textcache={};cases=[];maxdelta=0.
    for name,i in selected:
        m=bundle.metadata[i];a=m['offset'];n=m['tokens'];sl=slice(m['step_start'],m['step_stop']);sp=np.asarray(bundle.spans[sl])-a
        raw=np.asarray(bundle.features[a:a+n],float)[:,bundle.columns];z=(raw-bundle.mean[i])/bundle.scale[i];r=np.asarray(residual[a:a+n]);obs=z.mean(1);pred=obs[:,None]-r
        delta=float(np.max(np.abs((obs[:,None]-pred)-r)));maxdelta=max(maxdelta,delta);assert delta<1e-12
        fig,ax=plt.subplots(4,1,figsize=(15,12),sharex=True);t=np.arange(n)
        for f in range(5):ax[0].plot(t,z[:,f],lw=.75,alpha=.8,label=FEATURES[f])
        ax[1].plot(t,obs,color='black',lw=1,alpha=.65,label='Observed mean of5 z-features')
        for q,p in enumerate(PREDICTORS):
            ax[1].plot(t,pred[:,q],lw=1,label=p);ax[2].plot(t,r[:,q],lw=.8,alpha=.8,label=p)
        chosen=['innovation5','tcn__real','iu__ridge+tcn+noreset','equal__ridge+bocpd+noreset']
        for method in chosen:
            s=scores[method][sl];s=(s-s.mean())/(s.std()+1e-12);ax[3].stairs(s,np.r_[sp[:,0],sp[-1,1]],label=method,lw=1.5)
            best=int(np.argmax(s));ax[3].scatter([sp[best].mean()],[s[best]],s=35)
        u,v=sp[int(target[i])]
        for pane in ax:
            pane.axvspan(u,v,color='#e44848',alpha=.14,label='_first_error_step')
            for st in sp[:,0]:pane.axvline(st,color='gray',alpha=.25,lw=.7)
            pane.grid(alpha=.1);pane.legend(fontsize=7,loc='upper right',ncol=3)
        for num,(st,en) in enumerate(sp):ax[3].text((st+en)/2,ax[3].get_ylim()[0],str(num+1),fontsize=8,ha='center',va='bottom')
        ax[0].set_ylabel('Five observed z-features');ax[1].set_ylabel('Scalar prediction / observation');ax[2].set_ylabel('Signed residual');ax[3].set_ylabel('Step score (z for display)');ax[3].set_xlabel('Answer token index (0-based); numbers denote steps (1-based)')
        fig.suptitle(f"{name}: {records[i]['row_id']} / {m['cell']} | {GROUPS[cat[i]]}\nFirst annotated error: step {target[i]+1}, red span [{u},{v}); gate {'OPEN' if gate[i] else 'CLOSED'} | all tokens shown")
        fig.tight_layout();figure_save(fig,'CASE_'+name);plt.close(fig)
        cell=m['cell']
        if cell not in textcache:
            path,kind,ds=specs[cell];textcache[cell]=base.evaluator.old._source_row_map(base.evaluator.old.load_pickle(path),kind=kind,dataset=ds)
        row=textcache[cell][records[i]['row_id']];assert len(row['steps'])==len(sp) and int(row['label'])==target[i]
        cases.append(dict(name=name,uid=m['uid'],row_id=records[i]['row_id'],cell=cell,category=GROUPS[cat[i]],tokens=n,
            first_error_step=int(target[i])+1,error_span=[int(u),int(v)],gate_percentile=float(gatepercent[i]),question=row['problem'],steps=row['steps'],
            selected_steps={p:None if not gate[i] else int(peak[p][i])+1 for p in chosen},figure='CASE_'+name+'.png'))
        np.savez_compressed(OUT/('CASE_'+name+'.npz'),features=z,predicted_scalar=pred,observed_scalar=obs,residuals=r,spans=sp)
    payload=dict(scope='Post-selection descriptive PB error analysis; no models or scores changed.',counts=dict(zip(GROUPS,map(int,counts))),
        strict_missed_including_five_singletons=int(strict.sum()),strict_gate_open_missed=int((strict&gate).sum()),
        summaries=summaries,uncertainty=uncertainty,case_selection='Highest-overlap cell; missed closest to cohort covariate median, found closest in same cell excluding same source; plus two previous named examples.',
        cases=cases,event_bin_edges=bins.tolist(),event_coverage=coverage,event_means=profilemeans,length_quartiles_log2=quartiles.tolist())
    write(OUT/'ANALYSIS.json',payload)
    write(OUT/'AUDIT.json',dict(status='PASS',all_benchmark_answers=len(records),PB_error_answers=len(ids),source_groups=len(set(groups)),
        outcome_counts_reproduce_step389=True,source_residual_sha256=sha(SOURCE/'RESIDUALS.npy'),source_scores_sha256=sha(SOURCE/'SCORES_FROZEN.npz'),
        features_sha256=sha(DATA/'features.npy'),max_scalar_reconstruction_delta=maxdelta,code_sha256=sha(Path(__file__)),
        protocol_sha256=sha(ROOT/'docs/experiments/PREDICTOR_ERROR_PROFILES_20260915.md')))
    write(OUT/'RUN_STATE.json',dict(status='ANALYZED_PENDING_REPORT',seconds=time.perf_counter()-began,answers=len(ids),new_training=False))
    print(json.dumps(dict(counts=payload['counts'],strict=payload['strict_missed_including_five_singletons'],summaries=summaries,uncertainty=uncertainty,cases=[{k:v for k,v in c.items() if k not in ('question','steps')} for c in cases]),indent=2),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):run()
