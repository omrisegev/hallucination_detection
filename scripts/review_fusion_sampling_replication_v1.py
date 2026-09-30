"""Review selection, single-answer fitting, dense decisions and raw targets."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[key]='1'
from collections import Counter, defaultdict
import hashlib
import importlib.util
from pathlib import Path
import time
import warnings
import numpy as np
from scipy.signal import lfilter
from sklearn.mixture import GaussianMixture

ROOT=Path(__file__).resolve().parents[1]
def module(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
d=module(ROOT/'scripts/run_fusion_sampling_replication_v1.py','sampling_review_driver')
OUT=d.OUT; load,save,sha=d.load,d.save,d.sha
metrics=module(ROOT/'scripts/review_fusion_explicit_fallback_v1.py','sampling_independent_metrics')
algebra=module(ROOT/'scripts/review_fusion_prediction_quality_v1.py','sampling_independent_algebra')
orientation=module(ROOT/'scripts/review_fusion_context_bank_v1.py','sampling_independent_orientation')
from spectral_utils.answer_localization_v2 import prepare_local,JOINT_SEED
from spectral_utils.joint_lsml import covariance_matrix,fit_joint_lsml,discover_loao_consensus_groups,_profiled_jacobian_audit
from spectral_utils.adapted_dufs import adapted_dufs_soft_gates
from spectral_utils.laplacian_upcr import build_graph_from_features,IU_FIT_DEFAULTS
from spectral_utils.upcr import upcr_fit
from spectral_utils.fusion_window_sampling import choose_all,perturb_windows,diffusion_indices,selection_geometry
from spectral_utils.fusion_context_bank import context_matrix
from spectral_utils.answer_localization_v2 import moment_plan,moment_matrix


def independent_matrix(raw,bank,starts,ends):
    x=raw[:,[1,15,19,23,24,25,26,27,28]]
    if bank=='context':
        history=[]
        for span in (8,32):
            alpha=2/(span+1)
            history.append(lfilter([alpha],[1,-(1-alpha)],x,axis=0,zi=((1-alpha)*x[0])[None,:])[0])
        return np.array([np.column_stack((x[a:b].mean(0),history[0][a:b].mean(0),history[1][a:b].mean(0))).ravel() for a,b in zip(starts,ends)])
    t=np.linspace(-.5,.5,8)
    return np.array([np.column_stack((x[a:b].mean(0),x[a:b].std(0),(t@x[a:b])/(t@t))).ravel() for a,b in zip(starts,ends)])


def selected_scope(rows,p):
    if p['scope']=='all':return rows
    subset=[r for r in rows if r['sampling']['eligible']]
    if p['scope']=='eligible':return subset
    assert p['scope']=='eligible_both_native'
    s=p['left'].split('__')[0][len('sample_'):]
    return [r for r in subset if r['sampling']['original_joint_valid'] and r['sampling']['joint_valid'][s]]


def main():
    start=time.monotonic();manifest=d.verify()
    frozen,e,contrasts=[load(OUT/n) for n in ('SCORES_FROZEN.json','EVALUATION.json','CONTRASTS.json')]
    assert frozen['status']==contrasts['status']=='COMPLETE' and not frozen['labels_decoded']
    assert frozen['manifest_sha256']==sha(OUT/'MANIFEST.json') and e['scores_sha256']==sha(OUT/'SCORES_FROZEN.json')
    assert contrasts['evaluation_sha256']==sha(OUT/'EVALUATION.json')
    for p,h in frozen['files'].items():assert sha(p)==h,p
    rows={r['uid']:r for r in e['rows']}; prior={r['uid']:r for r in load(d.EVALUATION)['rows']}
    assert len(rows)==len(e['rows'])==110 and set(rows)==set(prior)
    official=module(ROOT/'spectral_utils/prmbench.py','sampling_official_labels')
    counts=Counter();banks=Counter();native=Counter();failures=Counter();coverage=Counter();fallback=Counter();sizes=Counter()
    group_checks=set();fit_checks=set();gate_checks=set();selector_checks=set();largest=0.
    for cell,path in d.io_module.RAW_FILES.items():
        with path.open('rb') as f: container=d.io_module.metadata.MetadataUnpickler(f).load()
        index={(v['idx'] if cell.startswith('prm') else cell.split('_')[1]+'::'+str(v['id'])):v for v in container.values()}
        assert len(index)==len(container)
        for row in (r for r in e['rows'] if r['cell']==cell):
            source=index[row['row_id']]
            target=1-np.asarray(official.eval_on_hallucination_step(source['error_steps'],[1]*len(source['steps']))['total_step_acc_list']) if cell.startswith('prm') else source['label']
            np.testing.assert_array_equal(target,row['target'])
            with np.load(d.ORIGINAL/'inputs'/(row['uid']+'.npz'),allow_pickle=False) as a:
                np.testing.assert_array_equal(source['step_token_spans'],np.column_stack((a['step_starts'],a['step_ends'])))
                for field,col in [('token_entropies',1),('token_spilled_energies',15),('token_logsumexp',19)]:
                    np.testing.assert_array_equal(source[field],a['raw'][:,col]);counts['raw_scalar_streams']+=1
            counts['raw_label_span_joins']+=1
        del container,index
    summaries=[]
    # POST-EVALUATION diagnostic motivated by the large pooled-risk AUC change.
    # Positive affine changes preserve each answer's ranking mathematically.
    # They are not registered candidates, and their PB decisions are not scored.
    affine_cores=('iu','graph010','equal','equal_graph_perm')
    affine_rows={direction:d.deepcopy(e['rows']) for direction in ('sample_to_full_scale','full_to_sample_scale')}
    affine_index={direction:{r['uid']:r for r in rr} for direction,rr in affine_rows.items()}
    for rec in manifest['selected']:
        uid=rec['uid'];cell=rec['cell'];row=rows[uid];old=prior[uid]
        for k in ('target','group_id','routing','cell','row_id'):assert row[k]==old[k]
        assert row['group_id']==rec['group_id']
        for arm in manifest['external_arms']:
            for k in ('valid','decision_valid','fixed_iu_valid','predictions','fixed_iu_predictions','peaks','sources'):assert row[k][arm]==old[k][arm]
            if old['valid'][arm]:np.testing.assert_array_equal(row['scores'][arm],old['scores'][arm])
            counts['unchanged_external_row_records']+=1
        meta=load(OUT/'scores'/(uid+'.json'));om=load(d.ORIGINAL/'scores'/(uid+'.json'));am=load(d.ANCHOR/'scores'/(uid+'.json'))
        assert not meta['labels_used'] and meta['manifest_sha256']==frozen['manifest_sha256']
        assert meta['array_sha256']==sha(OUT/'scores'/(uid+'.npz')) and meta['routing']==om['routing']==row['routing']
        dg=meta['diagnostics'];bank='context' if om['routing']['routes']['dual']=='context_joint' else 'moment';banks[bank]+=1
        assert dg['bank']==row['sampling']['bank']==bank
        with np.load(OUT/'scores'/(uid+'.npz'),allow_pickle=False) as a,np.load(d.ORIGINAL/'inputs'/(uid+'.npz'),allow_pickle=False) as inp,np.load(d.ANCHOR/'scores'/(uid+'.npz'),allow_pickle=False) as aa:
            raw=inp['raw'];starts=a['window_starts'];ends=a['window_ends'];fi=a['fit_indices'];ss=a['step_starts'];ee=a['step_ends']
            full=np.arange(0,len(raw)-7,8);expected=np.unique(np.r_[full,len(raw)-8])
            np.testing.assert_array_equal(starts,expected);np.testing.assert_array_equal(ends,starts+8);np.testing.assert_array_equal(fi,np.searchsorted(starts,full))
            for k in ('step_starts','step_ends'):np.testing.assert_array_equal(a[k],inp[k])
            x=a['features'];np.testing.assert_allclose(x,independent_matrix(raw,bank,starts,ends),atol=1e-12,rtol=1e-12);counts['independent_matrices']+=1
            n=len(fi);m=min(n,max(32,(n+1)//2));eligible=m<n;assert eligible==dg['eligible']==row['sampling']['eligible']
            identity=manifest['scoring_namespace']+'/'+cell+'/'+rec['row_id']+'/moments27_local8'
            expected_choices=None
            if eligible and cell not in selector_checks:
                expected_choices,_=choose_all(x[fi],identity)
                for rep in (0,1):
                    changed=perturb_windows(raw,identity,rep);builder=moment_matrix if bank=='moment' else context_matrix
                    perturbed,_=builder(changed,moment_plan(len(raw),8));pc,pd=choose_all(perturbed[fi],identity)
                    for selector,positions in expected_choices.items():
                        recorded=dg['selectors'][selector]['block_perturbation'][rep]
                        assert recorded['status']==pd[selector]['status']
                        if selector in pc:
                            left,right=set(map(int,positions)),set(map(int,pc[selector]));assert recorded['jaccard']==len(left&right)/len(left|right)
                        else:assert recorded['jaccard'] is None
                    counts['representative_perturbation_replays']+=1
                selector_checks.add(cell)
            for selector in d.SELECTORS:
                sd=dg['selectors'][selector];fd=dg['fits'][selector]
                if sd['status']=='FAILED':
                    assert fd['selector_failed'] and selector+'__selected' not in a
                    for core in d.CORES:assert not row['valid'][d.arm_name(selector,core)]
                    failures[selector+'/selector_failed']+=1;continue
                selected=a[selector+'__selected'];positions=np.searchsorted(fi,selected)
                assert len(selected)==(n if selector=='full' else m) and np.all(np.diff(selected)>0)
                np.testing.assert_array_equal(fi[positions],selected)
                if selector=='full' or not eligible:np.testing.assert_array_equal(selected,fi)
                elif selector=='uniform':np.testing.assert_array_equal(positions,np.rint(np.linspace(0,n-1,m)).astype(int))
                elif selector=='risk_top':np.testing.assert_array_equal(positions,sorted(sorted(range(n),key=lambda j:(-x[fi[j],0],j))[:m]))
                elif selector.startswith('dufs_'):
                    p=np.asarray(sd['probabilities']);assert p.shape==(n,)
                    np.testing.assert_array_equal(positions,sorted(sorted(range(n),key=lambda j:(-p[j],j))[:m]))
                    if selector=='dufs_permuted':
                        seed=int(hashlib.sha256((identity+'/sampling-permutation').encode()).hexdigest()[:8],16)
                        perm=np.random.default_rng(seed).permutation(n);np.testing.assert_array_equal(perm,sd['permutation'])
                        np.testing.assert_array_equal(p,np.asarray(dg['selectors']['dufs_transposed']['probabilities'])[perm])
                else:
                    assert selector=='window_diffusion';pos,_=diffusion_indices(selection_geometry(x[fi]),m);np.testing.assert_array_equal(pos,positions)
                if expected_choices is not None:np.testing.assert_array_equal(positions,expected_choices[selector])
                covered=np.zeros(len(raw),bool)
                for lo,hi in zip(starts[selected],ends[selected]):covered[lo:hi]=True
                fractions=np.array([covered[lo:hi].mean() for lo,hi in zip(ss,ee)])
                np.testing.assert_array_equal(fractions,a[selector+'__step_support_fraction'])
                np.testing.assert_array_equal(sd['quartile_counts'],np.bincount(np.minimum(3,starts[selected]*4//len(raw)),minlength=4))
                assert sd['largest_start_gap_tokens']==int(np.diff(np.r_[0,starts[selected],len(raw)]).max())
                counts['selection_and_support_checks']+=1
                item=dict(uid=uid,cell=cell,selector=selector,eligible=eligible,fit_rows=len(selected),original_rows=n,
                    joint_valid=fd['joint_valid'],token_fraction=float(covered.mean()),largest_gap=sd['largest_start_gap_tokens'],
                    perturbation_jaccard=[v['jaccard'] for v in sd.get('block_perturbation',[])])
                if cell.startswith('pb') and row['target']!=-1:
                    gold=row['target'];item.update(first_error_tokens=int(ee[gold]-ss[gold]),first_error_fit_support=float(fractions[gold]))
                summaries.append(item)
                replay=np.array_equal(selected,fi);assert replay==fd['replay']
                if replay:
                    joint_valid=om['methods'][bank+'__joint0']['valid']
                    for core,source in d.ANCHORS.items():
                        arm=d.arm_name(selector,core);md=meta['methods'][arm]
                        for k in ('valid','decision_valid','fixed_iu_valid','prediction','fixed_iu_prediction','peak'):assert md.get(k)==am['methods'][source].get(k)
                        if md['valid']:
                            for suffix in ('window','risk'):np.testing.assert_array_equal(a[arm+'__'+suffix],aa[source+'__'+suffix])
                        counts['exact_anchor_replays']+=1
                else:
                    sh=fd['shared'];names=dg['names'];rep=(cell,selector)
                    if sh.get('preparation_failure'):
                        assert not any(row['valid'][d.arm_name(selector,c)] for c in d.CORES);failures[selector+'/preparation']+=1;continue
                    z,anchor,normal=prepare_local(x,names,selected);np.testing.assert_array_equal(z,a[selector+'__z'])
                    for k in ('active_features','active_p','rank','anchor_feature'):assert normal[k]==sh[k]
                    for k in ('mean','sd','feature_signs'):np.testing.assert_allclose(normal[k],sh[k],atol=1e-12,rtol=1e-12)
                    cols=[names.index(name) for name in sh['active_features']];xr=x[:,cols];fitx=xr[selected]
                    independent=(xr-fitx.mean(0))/fitx.std(0);independent-=independent[selected].mean(0);independent*=sh['feature_signs']
                    np.testing.assert_allclose(independent,z,atol=1e-12,rtol=1e-12);counts['selected_normalization_replays']+=1
                    fit=z[selected];p=fit.shape[1];joint_valid=False;group=sh.get('grouping')
                    if group and rep not in group_checks:
                        g=discover_loao_consensus_groups(fit,np.minimum(3,np.arange(len(fit))*4//len(fit)),k_range=(3,4,6,8),seed=JOINT_SEED,
                            minimum_group_size=3,minimum_held_admissible_fraction=.95,use_minimum_ari_tiebreak=True)
                        for k in ('status','K','group_sizes','median_ari','candidates'):algebra.nested_equal(d.io_module.safe(g.get(k)),group.get(k))
                        group_checks.add(rep)
                    if 'joint' in sh:
                        jd=sh['joint'];groups=np.asarray(jd['groups']);assert min(Counter(groups).values())>=3 and len(groups)==p
                        v,u=a[selector+'__v'],a[selector+'__u'];mask=groups[:,None]==groups[None,:]
                        component=np.outer(v,v)+mask*np.outer(u,u);observed=np.cov(fit,rowvar=False,ddof=1)
                        cv=component+np.diag(np.maximum(np.diag(observed)-np.diag(component),0.))
                        np.testing.assert_allclose(cv,a[selector+'__covariance'],atol=1e-10,rtol=1e-10)
                        jac=_profiled_jacobian_audit(v,u,mask);metrics.check_equal(d.io_module.safe(jac),jd['jacobian'])
                        joint_valid=bool(jd['converged'] and jd['multistart']['status']=='PASS' and jac['full_global_rank'] and np.isfinite(jac['condition_number']) and jac['condition_number']<=1e8)
                        counts['covariance_jacobian_replays']+=1;sizes[selector+'/'+str(len(set(groups)))]+=1
                        if joint_valid and rep not in fit_checks:
                            j=fit_joint_lsml(covariance_matrix(fit),groups,anchor_index=anchor,seed=JOINT_SEED,starts=5,max_sweeps=5000)
                            for left,right in [(v,j.global_loading),(u,j.group_loading),(cv,j.model_covariance)]:np.testing.assert_allclose(left,right,atol=1e-10,rtol=1e-10)
                            assert j.converged and j.multistart_audit['status']=='PASS';fit_checks.add(rep)
                    assert joint_valid==sh['joint_valid'];jobs={'equal':(np.ones(p)/p,None)}
                    if not joint_valid:failures[selector+'/'+sh.get('joint_failure','unknown')]+=1
                    iu=upcr_fit(fit.T,**dict(IU_FIT_DEFAULTS));counts['iu_refits']+=1
                    if not iu.abstained:jobs['iu']=(iu.w,None)
                    else:assert not fd['native']['iu']['valid']
                    if joint_valid:jobs['joint0']=algebra.inverse(a[selector+'__covariance'],a[selector+'__v'])
                    if 'gates' in sh:
                        gd=sh['gates'];gates=np.asarray(gd['values']);np.testing.assert_array_equal(gates,a[selector+'__gates'])
                        if rep not in gate_checks:
                            gates2,_=adapted_dufs_soft_gates(fit.T,seeds=(0,1,2),epochs=120);np.testing.assert_allclose(gates,gates2,atol=1e-12,rtol=1e-12);gate_checks.add(rep)
                        gr=build_graph_from_features(fit.T,gates=gates,k=7).toarray();seed=int(hashlib.sha256(identity.encode()).hexdigest()[:8],16)
                        perm=np.random.default_rng(seed).permutation(len(fit));np.testing.assert_array_equal(perm,gd['permutation']);assert seed==gd['seed']
                        for kind,graph in [('graph010',gr),('graph_perm',gr[perm][:,perm])]:
                            degree=np.maximum(graph.sum(1),1e-12);lap=np.eye(len(fit))-graph/np.sqrt(np.outer(degree,degree))
                            rough=fit.T@lap@fit/len(fit);rough=(rough+rough.T)/2;tr=np.trace(rough);assert np.linalg.eigvalsh(rough).min()>-1e-10
                            jobs['equal_'+kind]=algebra.inverse(np.eye(p)+(.1*p/tr*rough if tr>1e-12 else 0),np.ones(p)/p)
                            if joint_valid:
                                cv=a[selector+'__covariance'];jobs[kind]=algebra.inverse(cv+(.1*np.trace(cv)/tr*rough if tr>1e-12 else 0),a[selector+'__v'])
                            counts['independent_laplacians']+=1
                    for core,(weight,ridge) in jobs.items():
                        detail=fd['native'][core];assert detail['valid'],(uid,selector,core,detail)
                        w,rule,flipped=orientation.orient(weight,fit,anchor)
                        np.testing.assert_allclose(w,detail['standardized_weights'],atol=1e-10,rtol=1e-10)
                        assert detail['orientation_rule']==rule and detail['flipped']==flipped
                        if ridge is not None:np.testing.assert_allclose(ridge,detail['inverse']['ridge'],atol=1e-10,rtol=1e-10)
                        risk=-z@w;stored=a[selector+'__native_'+core+'__window'];largest=max(largest,float(np.max(np.abs(risk-stored))))
                        np.testing.assert_allclose(risk,stored,atol=1e-10,rtol=1e-10);counts['independent_native_weights']+=1
                assert joint_valid==fd['joint_valid']==row['sampling']['joint_valid'][selector];native[selector]+=joint_valid
                # Check every exposed recipe, including the dense replay aliases.
                for core in d.CORES:
                    arm=d.arm_name(selector,core);detail=meta['methods'][arm]
                    fb=core in ('joint0','graph010','graph_perm') and not joint_valid
                    assert detail['fallback_to_sample_iu']==fb;fallback[arm]+=fb
                    if not replay:
                        source=selector+'__native_'+('iu' if fb else core);assert detail['source_arm']==source
                        if detail['valid']:
                            for suffix in ('window','risk'):np.testing.assert_array_equal(a[arm+'__'+suffix],a[source+'__'+suffix])
                    for k in ('valid','decision_valid','fixed_iu_valid'):assert row[k][arm]==detail[k]
                    for k,j in [('predictions','prediction'),('fixed_iu_predictions','fixed_iu_prediction'),('peaks','peak'),('sources','source_arm')]:assert row[k][arm]==detail.get(j)
                    if not detail['valid']:counts['invalid_final_outputs']+=1;continue
                    coverage[arm]+=1;stored=a[arm+'__window'];accum=np.zeros(len(raw));support=np.zeros(len(raw))
                    if selector=='risk_top' and core in affine_cores:
                        original_risk=aa[d.ANCHORS[core]+'__window'];original_step=aa[d.ANCHORS[core]+'__risk']
                        mu,sd0=stored[fi].mean(),stored[fi].std();mu0,sd1=original_risk[fi].mean(),original_risk[fi].std()
                        assert sd0>0 and sd1>0
                        transformed=(a[arm+'__risk']-mu)*(sd1/sd0)+mu0
                        reverse=(original_step-mu0)*(sd0/sd1)+mu
                        affine_index['sample_to_full_scale'][uid]['scores'][arm]=transformed.tolist()
                        affine_index['full_to_sample_scale'][uid]['scores'][arm]=reverse.tolist()
                    for lo,hi,value in zip(starts,ends,stored):accum[lo:hi]+=value;support[lo:hi]+=1
                    assert (support>0).all();token=accum/support;step=np.array([token[lo:hi].max() for lo,hi in zip(ss,ee)])
                    np.testing.assert_allclose(step,a[arm+'__risk'],atol=1e-12,rtol=1e-12);np.testing.assert_array_equal(row['scores'][arm],a[arm+'__risk'])
                    peak=int(np.argmax(a[arm+'__risk']));assert detail['peak']==peak
                    if not replay:
                        samples=stored[fi,None]  # full original grid, NOT selected rows
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore');gms=[GaussianMixture(n_components=k,n_init=3,max_iter=300,reg_covar=1e-4,random_state=2026090705).fit(samples) for k in (1,2)]
                        if all(g.converged_ for g in gms):
                            bic=[g.bic(samples) for g in gms];opened=bic[1]<bic[0] and np.any(a[arm+'__risk']>gms[1].means_.mean())
                            assert detail['decision_valid'] and detail['prediction']==(peak if opened else -1)
                            np.testing.assert_allclose(detail['gate']['bic'],bic,atol=1e-8,rtol=1e-10)
                        else:assert not detail['decision_valid']
                        counts['dense_gmm_replays']+=1
                    reference=om['methods']['moment__iu'];valid=bool(reference['valid'] and reference['decision_valid'])
                    assert detail['fixed_iu_valid']==valid
                    if valid:assert detail['fixed_iu_prediction']==(peak if reference['prediction']!=-1 else -1)
                    counts['step_output_replays']+=1
        if counts['independent_matrices']%20==0:print('Reviewed',counts['independent_matrices'],'/110',flush=True)
    eligible=[r for r in e['rows'] if r['sampling']['eligible']];assert len(eligible)==72
    for subset,name in [(e['rows'],'metrics'),(eligible,'eligible_metrics')]:
        for arm,bundle in e[name].items():
            metrics.check_equal(bundle,dict(prm=metrics.prm(subset,arm),pb=metrics.pb(subset,arm),pb_common_iu_gate=metrics.pb(subset,arm,True)))
            counts['independent_metric_bundles']+=1
    assert len(contrasts['pairs'])==93
    for pair in contrasts['pairs'].values():
        selected=selected_scope(e['rows'],pair);assert pair['selected_ids']==[r['uid'] for r in selected]
        common=[r for r in selected if r['valid'][pair['left']] and r['valid'][pair['right']]]
        for side in ('left','right'):
            metrics.check_equal(pair[side+'_prm'],metrics.prm(common,pair[side]));metrics.check_equal(pair[side+'_pb'],metrics.pb(selected,pair[side]))
        counts['paired_point_scope_checks']+=1
    checks=[]
    for selector,core,scope in [('risk_top','iu','all'),('dufs_transposed','graph010','all'),('window_diffusion','iu','eligible'),('dufs_permuted','graph010','eligible_both_native'),('uniform','graph010','all')]:
        pair=dict(left=d.arm_name(selector,core),right=d.arm_name('full',core),scope=scope)
        actual=contrasts['pairs'][d.key(pair)]['uncertainty'];selected=selected_scope(e['rows'],pair)
        for k,v in metrics.explicit_bootstrap(selected,pair['left'],pair['right']).items():metrics.check_equal(actual[k],v)
        checks.append({**pair,'draws':1000,'status':'MATCH'})
    transitions={}
    for selector in d.SELECTORS[1:]:
        for core in ('iu','graph010'):
            arm=d.arm_name(selector,core);base=d.arm_name('full',core);cells={}
            for cell in sorted({r['cell'] for r in e['rows'] if r['cell'].startswith('pb')}):
                items=[]
                for r in (r for r in e['rows'] if r['cell']==cell):
                    oldok=bool(r['decision_valid'][base] and r['predictions'][base]==r['target']);newok=bool(r['decision_valid'][arm] and r['predictions'][arm]==r['target'])
                    items.append(dict(uid=r['uid'],clean=r['target']==-1,old_correct=oldok,new_correct=newok,
                        prediction_changed=r['predictions'][base]!=r['predictions'][arm],peak_changed=r['peaks'][base]!=r['peaks'][arm]))
                cells[cell]=dict(gained=sum(v['new_correct'] and not v['old_correct'] for v in items),lost=sum(v['old_correct'] and not v['new_correct'] for v in items),
                    predictions_changed=sum(v['prediction_changed'] for v in items),peaks_changed=sum(v['peak_changed'] for v in items),rows=items)
            transitions[arm]=cells
    affine=[]
    for core in affine_cores:
        arm=d.arm_name('risk_top',core);base=d.arm_name('full',core)
        left=metrics.prm(affine_rows['sample_to_full_scale'],arm);right=metrics.prm(affine_rows['full_to_sample_scale'],arm)
        np.testing.assert_allclose(left['within_answer_auc'],e['metrics'][arm]['prm']['within_answer_auc'],atol=1e-12,rtol=0)
        np.testing.assert_allclose(right['within_answer_auc'],e['metrics'][base]['prm']['within_answer_auc'],atol=1e-12,rtol=0)
        affine.append(dict(core=core,full=e['metrics'][base]['prm'],sample=e['metrics'][arm]['prm'],
            sample_ranking_full_scale=left,full_ranking_sample_scale=right))
    save(OUT/'DIAGNOSTICS.json',dict(status='POST_SCORE_DESCRIPTIVE',records=summaries,pb_transitions=transitions,
        affine_diagnostic=affine,affine_scope='Post-evaluation positive affine normalization per answer using all original fit rows. PRMB-only diagnostic, not a candidate or new PB decision. Matching means and standard deviations does not prove causality or fully decompose the nonlinear AUC.',
        evaluation_sha256=sha(OUT/'EVALUATION.json')))
    report=dict(status='PASS',counts=dict(counts),fixed_banks=dict(banks),native_joint_valid=dict(native),coverage=dict(coverage),
        fallback_counts=dict(fallback),failures=dict(failures),K_counts=dict(sizes),maximum_risk_difference=largest,
        group_refits=sorted(group_checks),joint_refits=sorted(fit_checks),dufs_refits=sorted(gate_checks),selector_replays=sorted(selector_checks),bootstrap_checks=checks,
        hashes={str(OUT/n):sha(OUT/n) for n in ('MANIFEST.json','SCORES_FROZEN.json','EVALUATION.json','CONTRASTS.json','DIAGNOSTICS.json')},
        dependencies={str(p):sha(p) for p in [Path(__file__),Path(metrics.__file__),Path(algebra.__file__),Path(orientation.__file__),ROOT/'spectral_utils/prmbench.py']},
        seconds=time.monotonic()-start,scope='Same-session review. Raw annotations/spans/three scalar streams; independent feature, normalization, covariance/inverse/Laplacian and overlap algebra; direct PB/AUC and five explicit source bootstraps. Shared IU/Joint/group discovery, graph, DUFS/diffusion, GMM and source orientation-pruning kernels disclosed. Representative expensive refits, not every optimizer independently implemented. No external review or new model forward.')
    save(OUT/'REVIEW.json',report);print('REVIEW PASS',dict(counts),flush=True)


if __name__=='__main__':main()
