"""Direct observation-covariance/vector-IMM, raw-target and benchmark review."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
from collections import Counter
import hashlib
import importlib.util
from pathlib import Path
import time
import warnings
import numpy as np
from sklearn.mixture import GaussianMixture

ROOT=Path(__file__).resolve().parents[1]
def module(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
d=module(ROOT/'scripts/run_fusion_trajectory_imm_v1.py','trajectory_review_driver');OUT=d.OUT
load,save,sha=d.load,d.save,d.sha
metrics=module(ROOT/'scripts/review_fusion_explicit_fallback_v1.py','trajectory_independent_metrics')
vector=module(ROOT/'tests/test_fusion_trajectory_imm.py','trajectory_independent_vector')


def normalize(x,fi):return (x-x[fi].mean())/x[fi].std()


def independent_noise(y):
    diagonal=np.clip((np.median(np.abs(np.diff(y,axis=0)),axis=0)/(.67448975*np.sqrt(2)))**2,.05,1)
    R=np.diag(diagonal);constant=False
    if y.shape[1]==2:
        delta=np.diff(y,axis=0);centered=delta-delta.mean(0);sd=np.sqrt((centered**2).mean(0))
        constant=bool(np.any(sd<=1e-12))
        corr=0. if constant else float(np.clip((centered[:,0]*centered[:,1]).mean()/(sd[0]*sd[1]),-1,1))
        R[0,1]=R[1,0]=corr*np.sqrt(diagonal.prod())
    eig=np.linalg.eigvalsh(R);ridge=max(0.,float((eig[-1]-100*eig[0])/99),float(eig[-1]*1e-10))
    S=R+ridge*np.eye(len(R));inverse=np.linalg.inv(S);ones=np.ones(len(R));den=ones@inverse@ones
    return R,S,ridge,(inverse@ones)/den,1/den,constant


def main():
    started=time.monotonic();m=d.verify();f,e,c=[load(OUT/n) for n in ('SCORES_FROZEN.json','EVALUATION.json','CONTRASTS.json')]
    assert f['status']==c['status']=='COMPLETE' and not f['labels_decoded']
    assert f['manifest_sha256']==sha(OUT/'MANIFEST.json') and e['scores_sha256']==sha(OUT/'SCORES_FROZEN.json') and c['evaluation_sha256']==sha(OUT/'EVALUATION.json')
    for p,h in f['files'].items():assert sha(p)==h,p
    rows={r['uid']:r for r in e['rows']};prior={r['uid']:r for r in load(d.EVALUATION)['rows']};assert len(rows)==len(e['rows'])==110 and set(rows)==set(prior)
    counts=Counter();failures=Counter();coverage=Counter();duplicates=Counter();inherited=Counter();largest=0.;model_records=[]
    official=module(ROOT/'spectral_utils/prmbench.py','trajectory_official_labels')
    for cell,path in d.io_module.RAW_FILES.items():
        with path.open('rb') as handle:raw=d.io_module.metadata.MetadataUnpickler(handle).load()
        index={(r['idx'] if cell.startswith('prm') else cell.split('_')[1]+'::'+str(r['id'])):r for r in raw.values()};assert len(index)==len(raw)
        for row in (r for r in e['rows'] if r['cell']==cell):
            source=index[row['row_id']]
            target=1-np.asarray(official.eval_on_hallucination_step(source['error_steps'],[1]*len(source['steps']))['total_step_acc_list']) if cell.startswith('prm') else source['label']
            np.testing.assert_array_equal(target,row['target'])
            with np.load(d.ORIGINAL/'inputs'/(row['uid']+'.npz'),allow_pickle=False) as a:
                np.testing.assert_array_equal(source['step_token_spans'],np.column_stack((a['step_starts'],a['step_ends'])))
            counts['raw_label_span_joins']+=1
        del raw,index
    families={**d.PAIRS,**{name:(name,) for name in d.SINGLES}}
    for rec in m['selected']:
        uid=rec['uid'];row=rows[uid];old=prior[uid];meta=load(OUT/'scores'/(uid+'.json'));original=load(d.ORIGINAL/'scores'/(uid+'.json'));anchor=load(d.ANCHOR/'scores'/(uid+'.json'))
        for k in ('cell','target','group_id','routing','row_id'):assert row[k]==old[k]
        assert row['group_id']==rec['group_id'] and row['routing']==meta['routing']==original['routing']
        assert not meta['labels_used'] and meta['manifest_sha256']==f['manifest_sha256'] and meta['array_sha256']==sha(OUT/'scores'/(uid+'.npz'))
        for name in m['external_arms']:
            for k in ('valid','decision_valid','fixed_iu_valid','predictions','fixed_iu_predictions','peaks','sources'):assert row[k][name]==old[k][name]
            if old['valid'][name]:np.testing.assert_array_equal(row['scores'][name],old['scores'][name])
            counts['exact_external_row_records']+=1
        with np.load(OUT/'scores'/(uid+'.npz'),allow_pickle=False) as a,np.load(d.ANCHOR/'scores'/(uid+'.npz'),allow_pickle=False) as aa,np.load(d.ORIGINAL/'inputs'/(uid+'.npz'),allow_pickle=False) as inp:
            starts=a['window_starts'];ends=a['window_ends'];fi=a['fit_indices'];ss=a['step_starts'];ee=a['step_ends'];tokens=rec['tokens']
            full=np.arange(0,tokens-7,8);expected_starts=np.unique(np.r_[full,tokens-8])
            np.testing.assert_array_equal(starts,expected_starts);np.testing.assert_array_equal(ends,starts+8);np.testing.assert_array_equal(fi,np.searchsorted(starts,full))
            for k in ('step_starts','step_ends'):np.testing.assert_array_equal(a[k],inp[k])
            native_parent=original['routing']['routes']['dual']!='moment_iu';assert native_parent==row['trajectory_imm']['original_joint_valid']==meta['original_joint_valid']
            identity=m['scoring_namespace']+'/'+rec['cell']+'/'+rec['row_id']+'/moments27_local8'
            for family,components in families.items():
                gd=meta['diagnostics']['families'][family];source_names=[d.SOURCES[k] for k in components];assert gd['sources']==source_names
                expected_fallback=not native_parent and any(k.startswith('joint') for k in components);assert gd['inherited_parent_fallback']==expected_fallback;inherited[family]+=expected_fallback
                if not gd['valid']:
                    assert gd.get('reason');failures[family+'/'+gd['reason']]+=1
                    for name,md in meta['methods'].items():
                        if md['family']==family:assert not md['valid']
                    continue
                y=np.column_stack([aa[name+'__window'] for name in source_names]);np.testing.assert_array_equal(y,a[family+'__components'])
                for name in source_names:assert anchor['methods'][name]['valid']
                np.testing.assert_allclose(y[fi].mean(0),0,atol=1e-8,rtol=0);np.testing.assert_allclose(y[fi].std(0),1,atol=1e-8,rtol=0)
                keep=[0]
                if len(components)==2:
                    rho=np.corrcoef(y[fi],rowvar=False)[0,1];assert rho>-1+1e-10
                    np.testing.assert_allclose(gd['source_correlation'],rho,atol=1e-12,rtol=1e-12)
                    if rho<1-1e-10:keep.append(1)
                assert keep==gd['retained'];duplicates[family]+=len(keep)<len(components);selected=y[:,keep]
                R,S,ridge,w,variance,constant=independent_noise(selected[fi])
                for k,v in [('noise_covariance',R),('regularized_noise',S),('ridge',ridge),('weights',w)]:np.testing.assert_allclose(gd[k],v,atol=1e-11,rtol=1e-11)
                assert gd['constant_difference']==constant and gd['condition']<=100+1e-8
                raw_gls=selected@w;mu=raw_gls[fi].mean();sd=raw_gls[fi].std();gls=(raw_gls-mu)/sd;r=variance/sd**2
                np.testing.assert_allclose(gd['gls_scale']['mean'],mu,atol=1e-12,rtol=1e-12);np.testing.assert_allclose(gd['gls_scale']['sd'],sd,atol=1e-12,rtol=1e-12)
                np.testing.assert_allclose(r,gd['effective_variance'],atol=1e-10,rtol=1e-10)
                np.testing.assert_allclose(gls,a[family+'__gls'],atol=1e-10,rtol=1e-10);counts['independent_noise_gls_replays']+=1
                mean=normalize(selected.mean(1),fi);hold=np.r_[gls[fi],np.repeat(gls[fi][-1],len(starts)-len(fi))]
                if family in d.SINGLES:hold=np.r_[y[fi,0],np.repeat(y[fi,0][-1],len(starts)-len(fi))]
                expected=dict(mean=mean,gls=gls,hold=hold)
                for mode in ('imm','imm_permuted'):
                    if mode not in gd:
                        if mode=='imm':assert gd.get('imm_failure')
                        continue
                    order=np.arange(len(fi))
                    if mode=='imm_permuted':
                        seed=int(hashlib.sha256((identity+'/imm-time-permutation').encode()).hexdigest()[:8],16);order=np.random.default_rng(seed).permutation(len(fi))
                    np.testing.assert_array_equal(gd[mode]['permutation'],order)
                    observed=(selected[fi]-mu)/sd
                    state=vector.vector_imm(observed[order],S/sd**2,[.01*r,r])
                    for name,values in state.items():
                        ordered=np.empty_like(values);ordered[order]=values;state[name]=ordered
                        np.testing.assert_allclose(a[family+'__'+mode+'_'+name],ordered,atol=1e-9,rtol=1e-9)
                    level=state['level'];curve=normalize(level,np.arange(len(fi)))
                    np.testing.assert_allclose(gd[mode]['level_scale']['mean'],level.mean(),atol=1e-10,rtol=1e-10)
                    np.testing.assert_allclose(gd[mode]['level_scale']['sd'],level.std(),atol=1e-10,rtol=1e-10)
                    expected[mode]=np.r_[curve,np.repeat(curve[-1],len(starts)-len(fi))]
                    assert np.all(state['variance']>=0);np.testing.assert_allclose(state['mode_probability'].sum(1),1,atol=1e-12)
                    counts['direct_vector_imm_replays']+=1
                model_records.append(dict(uid=uid,cell=rec['cell'],family=family,retained=len(keep),source_correlation=gd['source_correlation'],
                    weights=gd['weights'],condition=gd['condition'],effective_variance=r,inherited_parent_fallback=expected_fallback,
                    median_fast_mode=float(np.median(a[family+'__imm_mode_probability'][:,1])) if 'imm' in gd else None))
                for name,md in meta['methods'].items():
                    if md['family']!=family:continue
                    for k in ('valid','decision_valid','fixed_iu_valid'):assert md[k]==row[k][name]
                    for dst,src in [('predictions','prediction'),('fixed_iu_predictions','fixed_iu_prediction'),('peaks','peak'),('sources','source_arm')]:assert row[dst][name]==md.get(src)
                    if not md['valid']:
                        assert md['readout'] not in expected;failures[name+'/'+md.get('reason','unknown')]+=1;continue
                    stored=a[name+'__window'];np.testing.assert_allclose(stored,expected[md['readout']],atol=1e-9,rtol=1e-9)
                    largest=max(largest,float(np.max(np.abs(stored-expected[md['readout']]))));coverage[name]+=1
                    accum=np.zeros(tokens);support=np.zeros(tokens)
                    for lo,hi,value in zip(starts,ends,stored):accum[lo:hi]+=value;support[lo:hi]+=1
                    assert (support>0).all();token=accum/support;step=np.array([token[lo:hi].max() for lo,hi in zip(ss,ee)])
                    np.testing.assert_allclose(step,a[name+'__risk'],atol=1e-12,rtol=1e-12);np.testing.assert_array_equal(row['scores'][name],a[name+'__risk'])
                    peak=int(np.argmax(a[name+'__risk']));assert md['peak']==peak
                    samples=stored[fi,None]
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore');gms=[GaussianMixture(n_components=k,n_init=3,max_iter=300,reg_covar=1e-4,random_state=2026090705).fit(samples) for k in (1,2)]
                    if all(g.converged_ for g in gms):
                        bic=[g.bic(samples) for g in gms];opened=bic[1]<bic[0] and np.any(a[name+'__risk']>gms[1].means_.mean())
                        assert md['decision_valid'] and md['prediction']==(peak if opened else -1)
                        np.testing.assert_allclose(md['gate']['bic'],bic,atol=1e-8,rtol=1e-10)
                    else:assert not md['decision_valid']
                    ref=original['methods']['moment__iu'];valid=bool(ref['valid'] and ref['decision_valid']);assert md['fixed_iu_valid']==valid
                    if valid:assert md['fixed_iu_prediction']==(peak if ref['prediction']!=-1 else -1)
                    counts['trajectory_step_gmm_replays']+=1
        counts['answers']+=1
        if counts['answers']%20==0:print('Reviewed',counts['answers'],'/110',flush=True)
    for name,bundle in e['metrics'].items():
        metrics.check_equal(bundle,dict(prm=metrics.prm(e['rows'],name),pb=metrics.pb(e['rows'],name),pb_common_iu_gate=metrics.pb(e['rows'],name,True)));counts['independent_metric_bundles']+=1
    assert len(c['pairs'])==50 and len(e['metrics'])==176
    for p in c['pairs'].values():
        selected=e['rows'] if p['scope']=='all' else [r for r in e['rows'] if r['routing']['routes']['dual']!='moment_iu']
        assert p['selected_ids']==[r['uid'] for r in selected];common=[r for r in selected if r['valid'][p['left']] and r['valid'][p['right']]]
        for side in ('left','right'):
            metrics.check_equal(p[side+'_prm'],metrics.prm(common,p[side]));metrics.check_equal(p[side+'_pb'],metrics.pb(selected,p[side]))
        counts['paired_point_scope_replays']+=1
    checks=[]
    for left,right,scope in [('traj_iu_joint_graph__imm','dual__iu','all'),('traj_iu_joint_graph__imm','traj_iu_joint_graph__hold','all'),
        ('traj_iu_joint_graph__imm','traj_iu_joint_graph__imm_permuted','all'),('traj_iu_joint_graph__imm','traj_equal_graph__imm','all'),
        ('traj_iu_joint_graph__mean','dual__cond100_graph010','native_parent_joint')]:
        p=dict(left=left,right=right,scope=scope);selected=d.select(e['rows'],scope);actual=c['pairs'][d.key(p)]['uncertainty']
        for k,v in metrics.explicit_bootstrap(selected,left,right).items():metrics.check_equal(actual[k],v)
        checks.append({**p,'draws':1000,'status':'MATCH'})
    pb=[r for r in e['rows'] if r['cell'].startswith('pb')];outcomes={};transitions={};short=[]
    for row in pb:
        if row['target']==-1:continue
        with np.load(OUT/'scores'/(row['uid']+'.npz'),allow_pickle=False) as a:
            length=int(a['step_ends'][row['target']]-a['step_starts'][row['target']])
        if length<=32:short.append(dict(uid=row['uid'],tokens=length))
    for name in m['new_arms']+['dual__iu','dual__cond100_graph010','sample_risk_top__equal_graph_perm']:
        clean=[r for r in pb if r['target']==-1];error=[r for r in pb if r['target']!=-1]
        outcomes[name]=dict(clean_correct=sum(r['decision_valid'][name] and r['predictions'][name]==-1 for r in clean),
            error_exact=sum(r['decision_valid'][name] and r['predictions'][name]==r['target'] for r in error),
            raw_peak_exact=sum(r['valid'][name] and r['peaks'][name]==r['target'] for r in error),
            clean_false_alarm=sum(r['decision_valid'][name] and r['predictions'][name]!=-1 for r in clean),
            error_gate_closed=sum(r['decision_valid'][name] and r['predictions'][name]==-1 for r in error))
    for p in m['contrasts']:
        if p['scope']!='all':continue
        left,right=p['left'],p['right'];records=[]
        for r in pb:
            old=bool(r['decision_valid'][right] and r['predictions'][right]==r['target']);new=bool(r['decision_valid'][left] and r['predictions'][left]==r['target'])
            records.append(dict(uid=r['uid'],cell=r['cell'],clean=r['target']==-1,gained=new and not old,lost=old and not new,
                peak_changed=r['peaks'][left]!=r['peaks'][right],gate_changed=(r['predictions'][left]==-1)!=(r['predictions'][right]==-1)))
        transitions[d.key(p)]=dict(rows=records,**{k:sum(r[k] for r in records) for k in ('gained','lost','peak_changed','gate_changed')})
    save(OUT/'DIAGNOSTICS.json',dict(status='POST_SCORE_DESCRIPTIVE',models=model_records,pb_outcomes=outcomes,pb_transitions=transitions,
        pb_first_errors_le32=short,evaluation_sha256=sha(OUT/'EVALUATION.json')))
    report=dict(status='PASS',counts=dict(counts),coverage=dict(coverage),failures=dict(failures),duplicate_collapses=dict(duplicates),
        inherited_fallbacks=dict(inherited),maximum_window_difference=largest,bootstrap_checks=checks,seconds=time.monotonic()-started,
        hashes={str(OUT/n):sha(OUT/n) for n in ('MANIFEST.json','SCORES_FROZEN.json','EVALUATION.json','CONTRASTS.json','DIAGNOSTICS.json')},
        dependencies={str(p):sha(p) for p in [Path(__file__),Path(metrics.__file__),Path(vector.__file__),ROOT/'spectral_utils/prmbench.py']},
        scope='Same-session review, not external: original raw labels/spans, unchanged source curves, independent noise covariance/GLS and direct vector IMM, output normalization/projection, direct AUC/PB and five explicit-row bootstraps. GMM, source metadata reader and official label port shared. Vector reference is the frozen test implementation; underlying feature-fusion fits reused, not refitted. No new model forward or causal-online claim.')
    save(OUT/'REVIEW.json',report);print('REVIEW PASS',dict(counts),flush=True)


if __name__=='__main__':main()
