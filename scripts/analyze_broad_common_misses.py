"""Broaden the method inventory and test ordinary step-boundary confounding."""
from pathlib import Path
import sys,json,csv,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from scripts.analyze_predictor_error_profiles import OUT,SOURCE,DATA,KEYS,TITLES,FEATURES,bootstrap,figure_save
from scripts.run_predictor_subset_study import sha,write
from scripts import run_temporal_research_baseline as base
from spectral_utils.context_training import FeatureBundle
from spectral_utils.predictor_subset_fusion import METHODS,PREDICTORS

ARCHIVES=['predictor_subset_iu_v1','tcn_aligned_predictor_seed0_v1','aligned_context_predictors_v1',
 'context_weighted_levels_v1','residual_moment_fusion_v1','temporal_linear_context_v1','temporal_position_control_v1',
 'temporal_dufs31_v1','temporal_research_baseline_v1','temporal_research_mechanism_v1','direct_probability_temporal_v3']
CONTROL_TAGS=['shuff','permut','random','zero','profile_only','constant_profile','amplitude','direction']

def run():
    records,j=base.load_contract(ROOT.parents[1]);bundle=FeatureBundle(DATA,'innovation5');offs=j['offsets'];target=j['target']
    error=np.array([r['cell'].startswith('pb_') for r in records])&(target>=0);ids=np.flatnonzero(error)
    with np.load(SOURCE/'SCORES_FROZEN.npz') as f:refs={n:f[n] for n in f.files}
    with np.load(ROOT/'results/temporal_research_baseline_v1/SCORES_FROZEN.npz') as f:gate=f['gate_percentile']>=.33
    anchor_names=['innovation5','original4','ridge','entropy15','RBM12','bocpd']
    inventory=[];allpeaks=[];realpeaks=[];aliases={};excluded=[];family_hits={};curve_hashes=set();current_found=np.zeros(len(records),bool)
    for n in METHODS:
        p=np.array([np.argmax(refs[n][offs[i]:offs[i+1]]) for i in range(len(records))]);current_found|=error&gate&(p==target)
    for folder in ARCHIVES:
        p=ROOT/'results'/folder/('SCORES.npz' if folder=='direct_probability_temporal_v3' else 'SCORES_FROZEN.npz')
        with np.load(p) as f:arrays={n:f[n] for n in f.files if f[n].shape==(145597,)}
        anchors=[(n,a) for n,s in arrays.items() for a in anchor_names if np.allclose(s,refs[a],rtol=0,atol=1e-6,equal_nan=False)]
        if not anchors:excluded.append(dict(archive=folder,reason='No full-array shared reference anchor matched; not admitted on shape alone.'));continue
        local=np.zeros(len(records),bool);rows=[]
        for n,s in arrays.items():
            if not np.isfinite(s).all():rows.append(dict(name=n,included=False,reason='Incomplete score coverage'));continue
            curve_hashes.add(hashlib.sha256(np.asarray(s,dtype='<f8').tobytes()).hexdigest())
            peaks=np.array([np.argmax(s[offs[i]:offs[i+1]]) for i in range(len(records))]);h=hashlib.sha256(peaks.astype('<i4').tobytes()).hexdigest()
            aliases.setdefault(h,[]).append(folder+'/'+n);allpeaks.append(peaks);local|=error&gate&(peaks==target)
            control=any(tag in n.lower() for tag in CONTROL_TAGS)
            if not control:realpeaks.append(peaks)
            rows.append(dict(name=n,included=True,control_tagged=control,peak_sha256=h,PB_final_hits=int((error&gate&(peaks==target)).sum())))
        family_hits[folder]=local
        inventory.append(dict(archive=folder,path=str(p.relative_to(ROOT)),sha256=sha(p),shared_anchors=anchors,rows=rows))
    union=np.logical_or.reduce([p==target for p in allpeaks]);realunion=np.logical_or.reduce([p==target for p in realpeaks]);found=error&gate&union
    cat=np.where(~gate,2,np.where(found,0,1));hard=error&gate&~union
    broad=dict(included_archives=len(inventory),scored_columns=len(allpeaks),unique_score_arrays=len(curve_hashes),unique_peak_vectors=len(aliases),
        found=int(found.sum()),missed_gate_open=int(hard.sum()),gate_closed=int((error&~gate).sum()),
        missed_final=int((error&~found).sum()),missed_raw_regardless_gate=int((error&~union).sum()),
        gate_closed_with_any_correct_raw_peak=int((error&~gate&union).sum()),
        missed_gate_open_without_control_tagged_methods=int((error&gate&~realunion).sum()),
        found_beyond_current32=int((found&~current_found).sum()),
        family_found_beyond_current32={n:int((hit&~current_found).sum()) for n,hit in family_hits.items()},
        scope='Available aligned full-population score archives in this worktree; not every historical experiment. Current gate held fixed. Broad union includes weak methods and controls and is an oracle, not deployable fusion.')
    hard_ids=np.flatnonzero(hard);P=np.stack(allpeaks)[:,hard_ids];diff=P-target[hard_ids][None,:]
    broad['geometry']=dict(any_archived_peak_one_step_away=int(np.any(np.abs(diff)==1,axis=0).sum()),
        every_archived_peak_earlier=int(np.all(diff<0,axis=0).sum()),every_archived_peak_later=int(np.all(diff>0,axis=0).sum()),
        mixed_earlier_and_later=int((np.any(diff<0,axis=0)&np.any(diff>0,axis=0)).sum()),current_methods={})
    for name in ['innovation5','tcn__real','iu__ridge+tcn+noreset','equal__ridge+bocpd+noreset']:
        rank=[];distance=[]
        for i in hard_ids:
            s=refs[name][offs[i]:offs[i+1]];rank.append(1+np.sum(s>s[int(target[i])]))
            distance.append(int(np.argmax(s))-int(target[i]))
        broad['geometry']['current_methods'][name]=dict(first_error_top2=int(np.sum(np.array(rank)<=2)),
            first_error_top3=int(np.sum(np.array(rank)<=3)),median_first_error_rank=float(np.median(rank)),
            selected_one_step_away=int(np.sum(np.abs(distance)==1)),selected_earlier=int(np.sum(np.array(distance)<0)))
    with (OUT/'ANSWER_DIAGNOSTICS.csv').open(encoding='utf8') as f:rows=list(csv.DictReader(f))
    assert [r['uid'] for r in rows]==[records[i]['uid'] for i in ids]
    X=np.array([[np.nan if row[k] in ('','None') else float(row[k]) for k in KEYS] for row in rows])
    cells=np.array([r['cell'] for r in rows]);groups=np.array([r['group_id'] for r in rows]);quart=np.quantile(X[:,0],[.25,.5,.75])
    lb=np.searchsorted(quart,X[:,0],side='right');pb=np.minimum((X[:,1]*3).astype(int),2)
    _,strata=np.unique([f'{c}/{l}/{p}' for c,l,p in zip(cells,lb,pb)],return_inverse=True)
    uncertainty=bootstrap(X,cat[ids],strata,groups)
    summaries={c:dict(n=int(np.sum(cat[ids]==k)),medians={name:float(np.nanmedian(X[cat[ids]==k,q])) for q,name in enumerate(KEYS)}) for k,c in enumerate(['found','missed_open','gate_closed'])}
    # A labelled correct control: immediately preceding step, before first error.
    bins=np.arange(-64,129,4);profiles=np.full((len(ids),2,len(bins)-1,10),np.nan);res=np.load(SOURCE/'RESIDUALS.npy',mmap_mode='r')
    for row,i in enumerate(ids):
        m=bundle.metadata[i];a=m['offset'];n=m['tokens'];z=(np.asarray(bundle.features[a:a+n],float)[:,bundle.columns]-bundle.mean[i])/bundle.scale[i];r=res[a:a+n]
        for kind,step in enumerate([int(target[i]),int(target[i])-1]):
            if step<0:continue
            u=int(bundle.spans[m['step_start']+step,0]-a)
            for b,(lo,hi) in enumerate(zip(bins[:-1],bins[1:])):
                aa=max(0,u+lo);bb=min(n,u+hi)
                if bb>aa:profiles[row,kind,b]=np.r_[z[aa:bb].mean(0),r[aa:bb].mean(0)]
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(15,8));colors=['#237c59','#c4403e','#67788a']
    for q,ax in enumerate(axes.flat):
        finite=X[np.isfinite(X[:,q]),q];edges=np.linspace(finite.min(),finite.max(),35)
        for c,label in enumerate(['Found by archive union','Missed by entire archive; gate open','Gate closed']):
            v=X[(cat[ids]==c)&np.isfinite(X[:,q]),q];ax.hist(v,bins=edges,weights=np.ones(len(v))/len(v),histtype='step',lw=1.7,label=label,color=colors[c])
        ax.set(title=TITLES[q],ylabel='Fraction of group / bin');ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=8);fig.suptitle(f"Broad inventory: {len(allpeaks)} score columns / {len(aliases)} distinct peak vectors; full PB error population");fig.tight_layout();figure_save(fig,'BROAD_HISTOGRAMS');plt.close(fig)
    fig,axes=plt.subplots(2,5,figsize=(18,8),sharex=True);paired_means={};coverage={}
    for c in range(2):
        mask=(cat[ids]==c)&(target[ids]>0);p=profiles[mask];common=np.isfinite(p[:,0])&np.isfinite(p[:,1]);p=np.where(common[:,None],p,np.nan)
        num=np.isfinite(p).sum(0);av=np.divide(np.nansum(p,0),num,out=np.full_like(num,np.nan,dtype=float),where=num>0)
        paired_means[str(c)]=av.tolist();coverage[str(c)]=num[:, :,0].tolist()
        for kind in range(2):
            label=('Found' if c==0 else 'Missed')+(' : error step' if kind==0 else ' : preceding correct step')
            for q,ax in enumerate(axes.flat):ax.plot((bins[:-1]+bins[1:])/2,av[kind,:,q],color=colors[c],ls='-' if kind==0 else '--',alpha=1 if kind==0 else .65,label=label)
    for q,ax in enumerate(axes.flat):
        ax.axvline(0,color='black',lw=.8);ax.axhline(0,color='gray',lw=.5);ax.grid(alpha=.15)
        ax.set_title(FEATURES[q] if q<5 else PREDICTORS[q-5]+' residual')
        if q>=5:ax.set_xlabel('Tokens from step start')
    axes[0,0].legend(fontsize=7);fig.suptitle('Boundary control: first-error step vs immediately preceding CORRECT step\nSame answer pairs per bin; dashed traces can enter the subsequent error step after their start');fig.tight_layout();figure_save(fig,'BOUNDARY_CONTROL');plt.close(fig)
    # Relative-to-start bins cannot guarantee subsequent tokens remain in the
    # control step. Quantify first4 tokens restricted to each actual step.
    boundary={}
    for c in range(2):
        vals=[]
        for i in ids[(cat[ids]==c)&(target[ids]>0)]:
            m=bundle.metadata[i];a=m['offset'];n=m['tokens'];z=(np.asarray(bundle.features[a:a+n],float)[:,bundle.columns]-bundle.mean[i])/bundle.scale[i]
            pair=[]
            for step in [int(target[i]),int(target[i])-1]:
                u,v=np.asarray(bundle.spans[m['step_start']+step])-a;pair.append(float(z[u:min(v,u+4)].mean()))
            vals.append(pair)
        v=np.array(vals);boundary[str(c)]=dict(n=len(v),error_first4_mean=float(v[:,0].mean()),correct_first4_mean=float(v[:,1].mean()))
    # One representative remaining common miss, chosen before viewing curves.
    pool=np.flatnonzero(cat[ids]==1);coords=X[:,[0,1,2]];center=np.median(coords[pool],0);scale=np.std(coords,0)+1e-12
    ix=min(pool,key=lambda k:(float(np.sum(((coords[k]-center)/scale)**2)),records[ids[k]]['uid']));i=int(ids[ix]);m=bundle.metadata[i];a=m['offset'];n=m['tokens'];sl=slice(m['step_start'],m['step_stop']);sp=np.asarray(bundle.spans[sl])-a
    z=(np.asarray(bundle.features[a:a+n],float)[:,bundle.columns]-bundle.mean[i])/bundle.scale[i];r=np.asarray(res[a:a+n]);obs=z.mean(1);mu=obs[:,None]-r
    fig,axes=plt.subplots(4,1,figsize=(15,12),sharex=True)
    for q in range(5):axes[0].plot(z[:,q],lw=.7,alpha=.8,label=FEATURES[q]);axes[1].plot(mu[:,q],lw=.9,label=PREDICTORS[q]);axes[2].plot(r[:,q],lw=.7,alpha=.8,label=PREDICTORS[q])
    axes[1].plot(obs,color='black',alpha=.65,lw=.8,label='Observed mean of5 z-features')
    methods=['innovation5','tcn__real','iu__ridge+tcn+noreset','equal__ridge+bocpd+noreset']
    for method in methods:
        s=refs[method][sl];s=(s-s.mean())/(s.std()+1e-12);axes[3].stairs(s,np.r_[sp[:,0],sp[-1,1]],label=method)
    u,v=sp[int(target[i])]
    for ax in axes:
        ax.axvspan(u,v,color='red',alpha=.12)
        for st in sp[:,0]:ax.axvline(st,color='gray',lw=.7,alpha=.3)
        ax.grid(alpha=.15);ax.legend(fontsize=7,ncol=3)
    for ax,label in zip(axes,['Observed z-features','Scalar predictions / observation','Signed residuals','Step scores (display z)']):ax.set_ylabel(label)
    axes[3].set_xlabel('Answer token index');fig.suptitle(f"Remaining common miss: {records[i]['row_id']} / {m['cell']} | first error step{target[i]+1}\nAll {len(allpeaks)} archived score columns miss with gate OPEN; red area = annotated step");fig.tight_layout();figure_save(fig,'CASE_COMMON_MISS');plt.close(fig)
    cell,path,kind,ds=next(x for x in base.evaluator.source_specs() if x[0]==m['cell']);row=base.evaluator.old._source_row_map(base.evaluator.old.load_pickle(path),kind=kind,dataset=ds)[records[i]['row_id']]
    case=dict(uid=m['uid'],row_id=records[i]['row_id'],cell=m['cell'],tokens=n,first_error_step=int(target[i])+1,error_span=[int(u),int(v)],question=row['problem'],steps=row['steps'])
    with (OUT/'COMMON_MISSES.csv').open('w',encoding='utf8',newline='') as f:
        writer=csv.writer(f);writer.writerow(['uid','row_id','cell','source_group','first_error_step_1based','gate_open','no_archive_peak_correct_even_without_gate'])
        for i in np.flatnonzero(error&~found):writer.writerow([records[i]['uid'],records[i]['row_id'],records[i]['cell'],records[i]['group_id'],int(target[i])+1,bool(gate[i]),bool(not union[i])])
    write(OUT/'BROAD_ANALYSIS.json',dict(inventory=inventory,excluded_archives=excluded,summary=broad,cohorts=summaries,uncertainty=uncertainty,
        first4_boundary_control=boundary,paired_boundary_means=paired_means,paired_boundary_coverage=coverage,case=case,
        original_scope_preserved=True,code_sha256=sha(Path(__file__))))
    print(json.dumps(dict(summary=broad,excluded=excluded,cohorts=summaries,uncertainty=uncertainty,boundary=boundary,case=case),indent=2),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):run()
