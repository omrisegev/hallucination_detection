"""Check raw confidence, matrices, native fusion, decisions and v3 evidence."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from collections import Counter
import ctypes
import gc
import hashlib
import importlib.util
from pathlib import Path
import pickle
import time
import warnings
import numpy as np
from scipy.signal import lfilter
from sklearn.mixture import GaussianMixture

ROOT=Path(__file__).resolve().parents[1]
def module(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
d=module(ROOT/'scripts/run_fusion_token_gap_v1.py','gap_review_driver')
OUT=d.OUT
load,save,sha=d.load,d.save,d.sha
metrics=module(ROOT/'scripts/review_fusion_explicit_fallback_v1.py','gap_independent_metrics')
algebra=module(ROOT/'scripts/review_fusion_prediction_quality_v1.py','gap_independent_algebra')
orientation=module(ROOT/'scripts/review_fusion_context_bank_v1.py','gap_independent_orientation')
from spectral_utils.answer_localization_v2 import prepare_local,JOINT_SEED
from spectral_utils.joint_lsml import covariance_matrix,fit_joint_lsml,discover_loao_consensus_groups,_profiled_jacobian_audit
from spectral_utils.adapted_dufs import adapted_dufs_soft_gates
from spectral_utils.laplacian_upcr import build_graph_from_features,IU_FIT_DEFAULTS
from spectral_utils.upcr import upcr_fit


def available_memory():
    class Memory(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in
            ('total','available','page_total','page_available','virtual_total','virtual_available','extended')]
    m=Memory();m.length=ctypes.sizeof(m)
    assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.available


def raw_audit(manifest,rows):
    """Read trusted original arrays only after workers exit; one cache at a time."""
    path=OUT/'RAW_CONFIDENCE_AUDIT.json'
    digest=sha(OUT/'MANIFEST.json');source=sha(Path(__file__))
    if path.exists():
        previous=load(path)
        if previous['manifest_sha256']==digest and previous['reviewer_sha256']==source:return previous
    official=module(ROOT/'spectral_utils/prmbench.py','gap_official_labels')
    counts=Counter();sizes={};maxdiff=0.;maxent=0.;per_answer=[]
    for cell,p in d.io_module.RAW_FILES.items():
        # The pickle is project-owned and frozen by the experiment manifest.
        assert sha(p)==manifest['hashes'][str(p)]
        free=available_memory();sizes[cell]={'bytes':p.stat().st_size,'available_ram_before':free}
        assert free>p.stat().st_size*2+300_000_000,('INSUFFICIENT_RAM_FOR_RAW_REVIEW',cell,free)
        with p.open('rb') as f:container=pickle.load(f)
        raw_index={(r['idx'] if cell.startswith('prm') else cell.split('_')[1]+'::'+str(r['id'])):r for r in container.values()}
        assert len(raw_index)==len(container)
        for rec in (r for r in manifest['selected'] if r['cell']==cell):
            uid=rec['uid'];saved=raw_index[rec['row_id']];row=rows[uid]
            with np.load(d.ORIGINAL/'inputs'/(uid+'.npz'),allow_pickle=False) as z:
                raw=z['raw'];ss=z['step_starts'];ee=z['step_ends']
            ids=np.asarray(saved['gen_token_ids']);top=saved['top_k_logprobs']
            lp=np.asarray(top['logprobs'],float);tid=np.asarray(top['ids'])
            assert lp.shape==tid.shape==(len(ids),50) and len(ids)==len(raw)==rec['tokens']
            assert np.isfinite(lp).all() and np.all(lp<=0) and np.all(np.diff(lp,axis=1)<=0)
            assert all(len(set(v))==50 for v in tid.tolist())
            np.testing.assert_array_equal(np.column_stack((ss,ee)),saved['step_token_spans'])
            for col,key in [(1,'token_entropies'),(15,'token_spilled_energies'),(19,'token_logsumexp')]:
                np.testing.assert_array_equal(raw[:,col],saved[key]);counts['raw_scalar_streams']+=1
            np.testing.assert_array_equal(raw[:,23],lp[:,0]);np.testing.assert_array_equal(raw[:,24],lp[:,0]-lp[:,1])
            pr=np.exp(lp);pr/=pr.sum(axis=1,keepdims=True)+1e-12;sl=-np.log(pr+1e-12)
            reconstructed=np.column_stack((lp[:,0],lp[:,0]-lp[:,1],(pr*sl).sum(1),
                (pr*(sl-(pr*sl).sum(1,keepdims=True))**2).sum(1),
                -np.log((pr**2).sum(1)+1e-12),np.clip(1-pr[:,:5].sum(1),0,1)))
            np.testing.assert_allclose(raw[:,23:29],reconstructed,atol=1e-12,rtol=1e-12)
            counts['raw_topk_derived_streams']+=6
            # Original entropy is a separate float32 top15 softmax calculation.
            p15=np.exp(lp[:,:15]-lp[:,:15].max(1,keepdims=True));p15/=p15.sum(1,keepdims=True)
            h15=-(p15*np.log(p15)).sum(1)
            maxent=max(maxent,float(np.max(np.abs(h15-raw[:,1]))))
            np.testing.assert_allclose(h15,raw[:,1],atol=2e-6,rtol=2e-6)
            matches=tid==ids[:,None];present=matches.any(1);assert (matches.sum(1)<=1).all()
            chosen=(lp*matches).sum(1)[present]
            diff=float(np.max(np.abs(chosen+raw[present,15]))) if present.any() else 0.;maxdiff=max(maxdiff,diff)
            np.testing.assert_allclose(chosen,-raw[present,15],atol=2e-6,rtol=2e-6)
            gap=raw[:,15]+lp[:,0];assert (gap>=0).all()
            np.testing.assert_allclose(gap[tid[:,0]==ids],0,atol=2e-6,rtol=0)
            if cell.startswith('prm'):
                port=official.eval_on_hallucination_step(saved['error_steps'],[1]*len(ss))
                target=1-np.asarray(port['total_step_acc_list'])
            else:target=int(saved['label'])
            np.testing.assert_array_equal(target,row['target'])
            stats={'uid':uid,'tokens':len(ids),'provided_in_top50':int(present.sum()),
                'provided_outside_top50':int((~present).sum()),'provided_top1_id':int((tid[:,0]==ids).sum()),
                'gap_near_zero':int((gap<=1e-5).sum()),'maximum_gap':float(gap.max())}
            per_answer.append(stats)
            for k in ('tokens','provided_in_top50','provided_outside_top50','provided_top1_id','gap_near_zero'):counts[k]+=stats[k]
            counts['raw_answer_and_label_joins']+=1
        del container,raw_index,saved;gc.collect()
        print('Original raw confidence and labels reviewed:',cell,flush=True)
    report={'status':'PASS','counts':dict(counts),'files':sizes,'answers':per_answer,
        'maximum_provided_logprob_difference':maxdiff,'maximum_entropy15_difference':maxent,
        'scope':'Original trusted arrays and writer-source contract; top50 retained probabilities checked directly. Outside-top50 provided probabilities are separately saved but not independently recoverable from top50. No new model forward pass or numeric logit-position verification.',
        'manifest_sha256':digest,'reviewer_sha256':source}
    save(path,report);return report


def matrix_from_raw(raw,bank,starts,ends):
    streams=raw[:,[1,15,19,23,24,25,26,27,28]].copy();streams[:,1]+=streams[:,3]
    if bank=='context':
        histories=[]
        for span in (8,32):
            alpha=2/(span+1)
            histories.append(lfilter([alpha],[1,-(1-alpha)],streams,axis=0,zi=((1-alpha)*streams[0])[None,:])[0])
        return np.array([np.column_stack((streams[a:b].mean(0),histories[0][a:b].mean(0),histories[1][a:b].mean(0))).ravel() for a,b in zip(starts,ends)])
    coordinate=np.linspace(-.5,.5,8);den=float(coordinate@coordinate)
    values=[]
    for a,b in zip(starts,ends):
        w=streams[a:b];mean=w.mean(0)
        values.append(np.column_stack((mean,w.std(0),coordinate@(w-mean)/den)).ravel())
    return np.asarray(values)


def main():
    start=time.monotonic();m=d.verify()
    f,e,c=[load(OUT/n) for n in ('SCORES_FROZEN.json','EVALUATION.json','CONTRASTS.json')]
    assert f['status']==c['status']=='COMPLETE' and not f['labels_decoded'] and not m['labels_decoded_for_scoring']
    assert f['manifest_sha256']==sha(OUT/'MANIFEST.json') and e['scores_sha256']==sha(OUT/'SCORES_FROZEN.json')
    assert c['evaluation_sha256']==sha(OUT/'EVALUATION.json') and len(c['pairs'])==25 and len(m['arms'])==107
    for p,h in f['files'].items():assert sha(p)==h,p
    rows={r['uid']:r for r in e['rows']};parent=load(d.EVALUATION);old={r['uid']:r for r in parent['rows']}
    assert len(rows)==len(e['rows'])==110
    raw_report=raw_audit(m,rows)
    counts=Counter();coverage=Counter();banks=Counter();native=0;fallback=Counter();ks=Counter();failures=Counter()
    group_checks=set();fit_checks=set();graph_checks=set();largest=0.
    for rec in m['selected']:
        uid=rec['uid'];row=rows[uid];prior=old[uid];cell=rec['cell']
        for k in ('group_id','target','row_id','cell','routing'):assert row[k]==prior[k]
        assert row['group_id']==rec['group_id']
        for arm in m['external_arms']:
            for k in ('valid','decision_valid','fixed_iu_valid','predictions','fixed_iu_predictions','peaks','sources'):assert row[k][arm]==prior[k][arm]
            if prior['valid'][arm]:np.testing.assert_array_equal(row['scores'][arm],prior['scores'][arm])
            counts['exact_parent_method_rows']+=1
        meta=load(OUT/'scores'/(uid+'.json'));om=load(d.ORIGINAL/'scores'/(uid+'.json'))
        assert meta['manifest_sha256']==f['manifest_sha256'] and not meta['labels_used']
        assert meta['array_sha256']==sha(OUT/'scores'/(uid+'.npz')) and meta['routing']==om['routing']==row['routing']
        bank='context' if om['routing']['routes']['dual']=='context_joint' else 'moment';banks[bank]+=1
        assert bank==meta['diagnostics']['bank']==row['token_gap']['bank']
        assert row['token_gap']['original_joint_valid']==om['methods'][bank+'__joint0']['valid']
        sh=meta['diagnostics']['shared'];names=meta['diagnostics']['names'];reference=om['methods']['moment__iu']
        with np.load(OUT/'scores'/(uid+'.npz'),allow_pickle=False) as a,np.load(d.ORIGINAL/'inputs'/(uid+'.npz'),allow_pickle=False) as inp,np.load(d.ORIGINAL/'scores'/(uid+'.npz'),allow_pickle=False) as original:
            raw=inp['raw'];fi=a['fit_indices'];ss=a['step_starts'];ee=a['step_ends'];starts=a['window_starts'];ends=a['window_ends']
            full=np.arange(0,len(raw)-7,8);expected_starts=np.unique(np.r_[full,len(raw)-8])
            np.testing.assert_array_equal(starts,expected_starts);np.testing.assert_array_equal(ends,starts+8)
            np.testing.assert_array_equal(fi,np.searchsorted(starts,full))
            for k in ('step_starts','step_ends'):np.testing.assert_array_equal(a[k],inp[k])
            gap=raw[:,15]+raw[:,23];np.testing.assert_array_equal(a['provided_token_gap'],gap)
            x=a['gap__features'];independent=matrix_from_raw(raw,bank,starts,ends)
            np.testing.assert_allclose(independent,x,atol=1e-12,rtol=1e-12)
            keep=[i for i in range(27) if i not in (3,4,5)]
            np.testing.assert_array_equal(x[:,keep],original[bank+'__features'][:,keep]);counts['independent_matrices']+=1
            z,anchor,normal=prepare_local(x,names,fi);np.testing.assert_array_equal(z,a['gap__z'])
            for k in ('active_features','active_p','rank','anchor_feature'):assert normal[k]==sh[k]
            cols=[names.index(n) for n in sh['active_features']];selected=x[:,cols];fitx=selected[fi]
            direct=(selected-fitx.mean(0))/fitx.std(0);direct-=direct[fi].mean(0);direct*=sh['feature_signs']
            np.testing.assert_allclose(direct,z,atol=1e-12,rtol=1e-12);counts['independent_normalizations']+=1
            fit=z[fi];p=fit.shape[1];rep=cell;group=sh.get('grouping');joint_valid=False
            if group and rep not in group_checks:
                g=discover_loao_consensus_groups(fit,np.minimum(3,np.arange(len(fit))*4//len(fit)),k_range=(3,4,6,8),
                    seed=JOINT_SEED,minimum_group_size=3,minimum_held_admissible_fraction=.95,use_minimum_ari_tiebreak=True)
                for k in ('status','K','group_sizes','median_ari','candidates'):algebra.nested_equal(d.io_module.safe(g.get(k)),group.get(k))
                group_checks.add(rep)
            if 'joint' in sh:
                jd=sh['joint'];groups=np.asarray(jd['groups']);sizes=Counter(groups.tolist());assert min(sizes.values())>=3
                assert len(groups)==p and group['K']==len(sizes);ks[str(len(sizes))]+=1
                v,u=a['gap__v'],a['gap__u'];mask=groups[:,None]==groups[None,:]
                common=np.outer(v,v)+mask*np.outer(u,u);observed=np.cov(fit,rowvar=False,ddof=1)
                covariance=common+np.diag(np.maximum(np.diag(observed)-np.diag(common),0))
                np.testing.assert_allclose(covariance,a['gap__covariance'],atol=1e-10,rtol=1e-10)
                jac=_profiled_jacobian_audit(v,u,mask);metrics.check_equal(d.io_module.safe(jac),jd['jacobian'])
                joint_valid=bool(jd['converged'] and jd['multistart']['status']=='PASS' and jac['full_global_rank'] and np.isfinite(jac['condition_number']) and jac['condition_number']<=1e8)
                counts['covariance_and_jacobian_replays']+=1
                if joint_valid and rep not in fit_checks:
                    j=fit_joint_lsml(covariance_matrix(fit),groups,anchor_index=anchor,seed=JOINT_SEED,starts=5,max_sweeps=5000)
                    for v0,v1 in [(v,j.global_loading),(u,j.group_loading),(covariance,j.model_covariance)]:np.testing.assert_allclose(v0,v1,atol=1e-10,rtol=1e-10)
                    assert j.converged and j.multistart_audit['status']=='PASS';fit_checks.add(rep)
            assert joint_valid==sh['joint_valid']==row['token_gap']['joint_valid'];native+=joint_valid
            if not joint_valid:failures[sh.get('joint_failure','unknown')]+=1
            jobs={'equal':(np.ones(p)/p,None)}
            iu=upcr_fit(fit.T,**dict(IU_FIT_DEFAULTS));assert not iu.abstained;jobs['iu']=(iu.w,None);counts['iu_refits']+=1
            gd=sh['gates'];gates=np.asarray(gd['values']);np.testing.assert_array_equal(gates,a['gap__gates'])
            if rep not in graph_checks:
                gg,_=adapted_dufs_soft_gates(fit.T,seeds=(0,1,2),epochs=120);np.testing.assert_allclose(gg,gates,atol=1e-12,rtol=1e-12);graph_checks.add(rep)
            graph=build_graph_from_features(fit.T,gates=gates,k=7).toarray()
            identity=m['scoring_namespace']+'/'+cell+'/'+rec['row_id']+'/moments27_local8'
            seed=int(hashlib.sha256(identity.encode()).hexdigest()[:8],16);permutation=np.random.default_rng(seed).permutation(len(fit))
            assert gd['seed']==seed;np.testing.assert_array_equal(gd['permutation'],permutation);counts['graph_seed_replays']+=1
            if joint_valid:jobs['joint0']=algebra.inverse(a['gap__covariance'],a['gap__v'])
            for kind,gr in [('graph010',graph),('graph_perm',graph[permutation][:,permutation])]:
                degree=np.maximum(gr.sum(1),1e-12);lap=np.eye(len(fit))-gr/np.sqrt(np.outer(degree,degree))
                rough=fit.T@lap@fit/len(fit);rough=(rough+rough.T)/2;tr=np.trace(rough)
                assert np.linalg.eigvalsh(rough).min()>-1e-10
                jobs['equal_'+kind]=algebra.inverse(np.eye(p)+(.1*p/tr*rough if tr>1e-12 else 0),np.ones(p)/p)
                if joint_valid:
                    cv=a['gap__covariance'];jobs[kind]=algebra.inverse(cv+(.1*np.trace(cv)/tr*rough if tr>1e-12 else 0),a['gap__v'])
                counts['independent_laplacians']+=1
            for core,(w,ridge) in jobs.items():
                detail=meta['methods']['gap__'+core];assert detail['valid'] and not detail['fallback_to_gap_iu']
                weight,rule,flipped=orientation.orient(w,fit,anchor)
                np.testing.assert_allclose(weight,detail['standardized_weights'],atol=1e-10,rtol=1e-10)
                assert detail['orientation_rule']==rule and detail['flipped']==flipped
                if ridge is not None:np.testing.assert_allclose(ridge,detail['inverse']['ridge'],atol=1e-10,rtol=1e-10)
                risk=-z@weight;np.testing.assert_allclose(risk,a['native__'+core+'__window'],atol=1e-10,rtol=1e-10)
                largest=max(largest,float(np.max(np.abs(risk-a['native__'+core+'__window']))));counts['independent_fusion_weights']+=1
            for arm in m['new_arms']:
                detail=meta['methods'][arm]
                if arm.startswith('gap__'):
                    core=arm.removeprefix('gap__');fb=core in ('joint0','graph010','graph_perm') and not joint_valid
                    assert detail['fallback_to_gap_iu']==fb;fallback[arm]+=fb
                    expected='native__'+('iu' if fb else core);assert detail['source_arm']==expected
                    if detail['valid']:
                        for suffix in ('window','risk'):np.testing.assert_array_equal(a[arm+'__'+suffix],a[expected+'__'+suffix])
                    if fb:
                        for k in ('valid','decision_valid','prediction','fixed_iu_valid','fixed_iu_prediction','peak'):assert detail.get(k)==meta['methods']['gap__iu'].get(k)
                else:
                    stream=gap if arm=='gap_scalar' else raw[:,15]
                    means=np.array([stream[lo:hi].mean() for lo,hi in zip(starts,ends)]);mf=means[fi]
                    assert detail['valid']==(mf.std()>1e-10)
                    if detail['valid']:np.testing.assert_allclose(a[arm+'__window'],(means-mf.mean())/mf.std(),atol=1e-12,rtol=1e-12)
                for k in ('valid','decision_valid','fixed_iu_valid'):assert row[k][arm]==detail[k]
                for k,j in [('predictions','prediction'),('fixed_iu_predictions','fixed_iu_prediction'),('peaks','peak'),('sources','source_arm')]:assert row[k][arm]==detail.get(j)
                if not detail['valid']:counts['invalid_final_outputs']+=1;continue
                coverage[arm]+=1;stored=a[arm+'__window']
                accum=np.zeros(len(raw));support=np.zeros(len(raw))
                for lo,hi,val in zip(starts,ends,stored):accum[lo:hi]+=val;support[lo:hi]+=1
                assert (support>0).all();token=accum/support;step=np.array([token[lo:hi].max() for lo,hi in zip(ss,ee)])
                np.testing.assert_allclose(step,a[arm+'__risk'],atol=1e-12,rtol=1e-12)
                np.testing.assert_array_equal(row['scores'][arm],a[arm+'__risk'])
                # Preserve exact stored tie-breaking while checking independent
                # overlap arithmetic to the established floating-point tolerance.
                peak=int(np.argmax(a[arm+'__risk']));assert detail['peak']==peak
                samples=stored[fi,None]
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    models=[GaussianMixture(n_components=k,n_init=3,max_iter=300,reg_covar=1e-4,random_state=2026090705).fit(samples) for k in (1,2)]
                if all(g.converged_ for g in models):
                    bic=[g.bic(samples) for g in models];opened=bic[1]<bic[0] and np.any(a[arm+'__risk']>models[1].means_.mean())
                    assert detail['decision_valid'] and detail['prediction']==(peak if opened else -1)
                    np.testing.assert_allclose(detail['gate']['bic'],bic,atol=1e-8,rtol=1e-10)
                else:assert not detail['decision_valid']
                valid=bool(reference['valid'] and reference['decision_valid']);assert valid==detail['fixed_iu_valid']
                if valid:assert detail['fixed_iu_prediction']==(peak if reference['prediction']!=-1 else -1)
                counts['step_gate_fallback_and_row_replays']+=1
        if counts['independent_matrices']%20==0:print('Reviewed fusion outputs:',counts['independent_matrices'],'/110',flush=True)
    for arm,bundle in e['metrics'].items():
        metrics.check_equal(bundle,{'prm':metrics.prm(e['rows'],arm),'pb':metrics.pb(e['rows'],arm),'pb_common_iu_gate':metrics.pb(e['rows'],arm,True)})
        if arm in m['external_arms']:assert bundle==parent['metrics'][arm]
        counts['independent_metric_bundles']+=1
    for pair in c['pairs'].values():
        selected=[r for r in e['rows'] if pair['scope']=='all' or (r['token_gap']['joint_valid'] and (pair['scope']!='native_gap_and_original' or r['token_gap']['original_joint_valid']))]
        assert pair['selected_ids']==[r['uid'] for r in selected]
        common=[r for r in selected if r['valid'][pair['left']] and r['valid'][pair['right']]]
        for side in ('left','right'):
            metrics.check_equal(pair[side+'_prm'],metrics.prm(common,pair[side]));metrics.check_equal(pair[side+'_pb'],metrics.pb(selected,pair[side]))
        counts['paired_point_scope_replays']+=1
    checks=[]
    for left,right,scope in [('gap__iu','dual__iu','all'),('gap__graph010','dual__cond100_graph010','all'),
        ('gap_scalar','surprisal_scalar','all'),('gap__graph010','gap__equal_graph010','native_gap'),
        ('gap__graph010','dual__cond100_graph010','native_gap_and_original')]:
        selected=d.select(e['rows'],scope);actual=c['pairs'][d.key(dict(left=left,right=right,scope=scope))]['uncertainty']
        for k,v in metrics.explicit_bootstrap(selected,left,right).items():metrics.check_equal(actual[k],v)
        checks.append(dict(left=left,right=right,scope=scope,status='MATCH',draws=1000))
    dependencies=[Path(__file__),Path(metrics.__file__),Path(algebra.__file__),Path(orientation.__file__),ROOT/'spectral_utils/prmbench.py']
    report={'status':'PASS','counts':dict(counts),'coverage':dict(coverage),'fixed_banks':dict(banks),'native_joint_valid':native,
        'joint_failures':dict(failures),'fallback_counts':dict(fallback),'K_counts':dict(ks),
        'representative_group_refits':sorted(group_checks),'representative_joint_refits':sorted(fit_checks),'representative_dufs_refits':sorted(graph_checks),
        'maximum_risk_difference':largest,'bootstrap_checks':checks,'raw_confidence_sha256':sha(OUT/'RAW_CONFIDENCE_AUDIT.json'),
        'hashes':{str(OUT/n):sha(OUT/n) for n in ('MANIFEST.json','SCORES_FROZEN.json','EVALUATION.json','CONTRASTS.json')},
        'dependencies':{str(p):sha(p) for p in dependencies},'seconds':time.monotonic()-start,
        'review_notes':['Initial review stopped on an incorrect invocation of the official label port (a report dictionary was treated as a target array). Corrected the review to use raw one-based error steps, an all-valid prediction vector, and total_step_acc_list. Frozen scoring code, labels and predictions were unchanged.'],
        'scope':'Same-session review; raw arrays/official label port, independent feature algebra, normalization, covariance/inverse/Laplacian, overlap projection, pairwise AUC/PB counts and five explicit-row bootstraps. Shared orientation and IU/Joint/grouping/DUFS/GMM kernels disclosed. Not an external review or new model forward pass.'}
    save(OUT/'REVIEW.json',report);print('REVIEW PASS',dict(counts),flush=True)


if __name__=='__main__':main()
