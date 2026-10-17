"""Review new fits, immutable anchors, graph attribution, fallbacks and metrics."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
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
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
driver=module(ROOT/'scripts/run_fusion_prediction_quality_v1.py','quality_review_driver')
OUT,PARENT,AUDIT,ORIGINAL=driver.OUT,driver.PARENT,driver.AUDIT,driver.ORIGINAL
load,sha,save=driver.load,driver.sha,driver.save
from spectral_utils.answer_localization_v2 import prepare_local,JOINT_SEED
from spectral_utils.joint_lsml import covariance_matrix,fit_joint_lsml,discover_loao_consensus_groups,_profiled_jacobian_audit
from spectral_utils.adapted_dufs import adapted_dufs_soft_gates
from spectral_utils.laplacian_upcr import build_graph_from_features,IU_FIT_DEFAULTS
from spectral_utils.upcr import upcr_fit


def inverse(system,loading):
    ev,q=np.linalg.eigh((system+system.T)/2);psd=(q*np.maximum(ev,0.))@q.T
    lo,hi=np.linalg.eigvalsh(psd)[[0,-1]]
    ridge=1. if hi<=1e-14 else max(0.,(hi-100*lo)/99.,hi*1e-10)
    return np.linalg.solve(psd+ridge*np.eye(len(loading)),loading),ridge


def nested_equal(actual,expected):
    """Compare grouping records, including lists of candidate dictionaries."""
    if isinstance(expected,dict):
        assert isinstance(actual,dict) and set(actual)==set(expected)
        for key in expected:nested_equal(actual[key],expected[key])
    elif isinstance(expected,list):
        assert isinstance(actual,list) and len(actual)==len(expected)
        for a,b in zip(actual,expected):nested_equal(a,b)
    elif isinstance(expected,(int,float)) and not isinstance(expected,bool):
        np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)
    else:assert actual==expected


def main():
    start=time.monotonic();m=driver.verify();f,e,c=[load(OUT/n) for n in ('SCORES_FROZEN.json','EVALUATION.json','CONTRASTS.json')]
    assert f['manifest_sha256']==sha(OUT/'MANIFEST.json') and e['scores_sha256']==sha(OUT/'SCORES_FROZEN.json')
    assert c['evaluation_sha256']==sha(OUT/'EVALUATION.json') and c['status']=='COMPLETE'
    assert len(m['arms'])==98 and len(c['pairs'])==74 and not f['labels_decoded']
    for p,h in f['files'].items():assert sha(p)==h,p
    metrics=module(ROOT/'scripts/review_fusion_explicit_fallback_v1.py','prediction_independent_metrics')
    orient=module(ROOT/'scripts/review_fusion_context_bank_v1.py','prediction_independent_orientation')
    parent=load(PARENT/'EVALUATION.json');prevrows={r['uid']:r for r in parent['rows']}
    rows={r['uid']:r for r in e['rows']};assert len(rows)==len(e['rows'])==110
    release=load(driver.SOURCE/'RELEASE_V2.json')
    counts=Counter();coverage=Counter();banks=Counter();native=Counter();fallback=Counter();group_sizes={}
    grouping_checks=set();fit_checks=set();gate_checks=set();largest=0.
    for cell in sorted({r['cell'] for r in m['selected']}):
        with np.load(release['cells'][cell]['label_path'],allow_pickle=False) as labels:
            index={str(v):i for i,v in enumerate(labels['row_ids'])};assert len(index)==len(labels['row_ids'])
            for rec in (r for r in m['selected'] if r['cell']==cell):
                uid=rec['uid'];row=rows[uid];prior=prevrows[uid];i=index[rec['row_id']]
                if cell.startswith('prm'):
                    a,b=labels['step_flag_offsets'][i:i+2];target=labels['step_error_flags'][a:b]
                else:target=int(labels['first_error'][i])
                np.testing.assert_array_equal(row['target'],target);np.testing.assert_array_equal(row['target'],prior['target'])
                assert row['group_id']==rec['group_id']==prior['group_id'];counts['direct_label_group_joins']+=1
                for arm in driver.PARENT_ARMS:
                    for key in ('valid','decision_valid','fixed_iu_valid','predictions','fixed_iu_predictions','peaks','sources'):
                        assert row[key][arm]==prior[key][arm]
                    if prior['valid'][arm]:np.testing.assert_array_equal(row['scores'][arm],prior['scores'][arm])
                    counts['exact_parent_method_rows']+=1
                meta=load(OUT/'scores'/(uid+'.json'));om=load(ORIGINAL/'scores'/(uid+'.json'));am=load(AUDIT/'answers'/(uid+'.json'))
                assert meta['manifest_sha256']==f['manifest_sha256'] and not meta['labels_decoded']
                assert meta['array_sha256']==sha(OUT/'scores'/(uid+'.npz'))
                assert row['routing']==meta['routing']==om['routing']
                route=om['routing']['routes']['dual'];bank='context' if route=='context_joint' else 'moment'
                assert meta['diagnostics']['bank']==row['augmentation']['bank']==bank;banks[bank]+=1
                assert row['augmentation']['original_joint_valid']==om['methods'][bank+'__joint0']['valid']
                with np.load(OUT/'scores'/(uid+'.npz'),allow_pickle=False) as a,np.load(AUDIT/'answers'/(uid+'.npz'),allow_pickle=False) as aa:
                    fi=a['fit_indices'];starts=a['window_starts'];ends=a['window_ends'];ss=a['step_starts'];ee=a['step_ends']
                    for key in ('fit_indices','window_starts','window_ends'):np.testing.assert_array_equal(a[key],aa[key])
                    for kind in driver.KINDS:
                        sh=meta['diagnostics']['variants'][kind];x=a[kind+'__features']
                        names=am['diagnostics']['banks'][bank+'__'+kind]['names']
                        np.testing.assert_array_equal(x,aa[bank+'__'+kind+'__features']);counts['exact_augmented_matrix_replays']+=1
                        if sh.get('preparation_failure'):
                            for core in driver.CORES:assert not meta['methods'][kind+'__'+core]['valid']
                            counts['preparation_failures']+=1;continue
                        z,anchor,normal=prepare_local(x,names,fi)
                        np.testing.assert_array_equal(z,a[kind+'__z'])
                        for key in ('active_features','active_p','rank','anchor_feature'):assert normal[key]==sh[key]
                        for key in ('mean','sd','feature_signs'):np.testing.assert_allclose(normal[key],sh[key],atol=1e-12,rtol=1e-12)
                        cols=[names.index(name) for name in sh['active_features']];selected=x[:,cols];fitx=selected[fi]
                        independent=(selected-fitx.mean(0))/fitx.std(0);independent-=independent[fi].mean(0);independent*=sh['feature_signs']
                        np.testing.assert_allclose(independent,z,atol=1e-12,rtol=1e-12);counts['normalization_orientation_replays']+=1
                        fit=z[fi];p=fit.shape[1];joint_valid=False;rep=(cell,kind)
                        group=sh.get('grouping')
                        if group and rep not in grouping_checks:
                            g=discover_loao_consensus_groups(fit,np.minimum(3,np.arange(len(fit))*4//len(fit)),
                                k_range=(3,4,6,8),seed=JOINT_SEED,minimum_group_size=3,
                                minimum_held_admissible_fraction=.95,use_minimum_ari_tiebreak=True)
                            for key in ('status','K','group_sizes','median_ari','candidates'):
                                nested_equal(driver.safe(g.get(key)),group.get(key))
                            grouping_checks.add(rep)
                        if 'joint' in sh:
                            jd=sh['joint'];groups=np.asarray(jd['groups']);sizes=Counter(groups.tolist())
                            assert min(sizes.values())>=3 and len(groups)==p
                            assert group['status']=='SELECTED' and group['K']==len(sizes)
                            group_sizes[kind+'/'+str(len(sizes))]=group_sizes.get(kind+'/'+str(len(sizes)),0)+1
                            v,u=a[kind+'__v'],a[kind+'__u'];mask=groups[:,None]==groups[None,:]
                            component=np.outer(v,v)+np.outer(u,u)*mask
                            centered=fit-fit.mean(0);observed=centered.T@centered/(len(fit)-1)
                            diagonal=np.maximum(np.diag(observed)-np.diag(component),0.)
                            covariance=component+np.diag(diagonal)
                            np.testing.assert_allclose(covariance,a[kind+'__covariance'],atol=1e-10,rtol=1e-10)
                            counts['independent_covariance_constructions']+=1
                            jac=_profiled_jacobian_audit(v,u,mask)
                            metrics.check_equal(driver.safe(jac),jd['jacobian']);counts['source_jacobian_replays']+=1
                            joint_valid=bool(jd['converged'] and jd['multistart']['status']=='PASS'
                                and jac['full_global_rank'] and np.isfinite(jac['condition_number']) and jac['condition_number']<=1e8)
                            if joint_valid and rep not in fit_checks:
                                fitted=fit_joint_lsml(covariance_matrix(fit),groups,anchor_index=anchor,seed=JOINT_SEED,starts=5,max_sweeps=5000)
                                for expected,actual in [(v,fitted.global_loading),(u,fitted.group_loading),(a[kind+'__covariance'],fitted.model_covariance)]:
                                    np.testing.assert_allclose(expected,actual,atol=1e-10,rtol=1e-10)
                                assert fitted.converged and fitted.multistart_audit['status']=='PASS';fit_checks.add(rep)
                        assert joint_valid==sh['joint_valid']==row['augmentation']['native_joint_valid'][kind]
                        native[kind]+=joint_valid;counts['native_fit_guards']+=1
                        jobs={'equal':(np.ones(p)/p,None)}
                        iu=upcr_fit(fit.T,**dict(IU_FIT_DEFAULTS));counts['source_iu_refits']+=1
                        if not iu.abstained:jobs['iu']=(iu.w,None)
                        else:assert not meta['methods'][kind+'__iu']['valid']
                        graphs={}
                        if 'gates' in sh:
                            gd=sh['gates'];gates=np.asarray(gd['values']);np.testing.assert_array_equal(gates,a[kind+'__gates'])
                            if rep not in gate_checks:
                                gg,_=adapted_dufs_soft_gates(fit.T,seeds=(0,1,2),epochs=120)
                                np.testing.assert_allclose(gg,gates,atol=1e-12,rtol=1e-12);gate_checks.add(rep)
                            graph=build_graph_from_features(fit.T,gates=gates,k=7).toarray()
                            seed=int(hashlib.sha256((m['scoring_namespace']+'/'+cell+'/'+rec['row_id']+'/moments27_local8').encode()).hexdigest()[:8],16)
                            permutation=np.random.default_rng(seed).permutation(len(fit));assert gd['seed']==seed
                            np.testing.assert_array_equal(permutation,gd['permutation']);counts['source_graph_replays']+=1
                            graphs={'graph010':graph,'graph_perm':graph[permutation][:,permutation]}
                        if joint_valid:
                            jobs['joint0']=inverse(a[kind+'__covariance'],a[kind+'__v'])
                        for which,graph in graphs.items():
                            degree=np.maximum(graph.sum(1),1e-12)
                            lap=np.eye(len(fit))-graph/np.sqrt(np.outer(degree,degree))
                            rough=fit.T@lap@fit/len(fit);rough=(rough+rough.T)/2;trace=np.trace(rough)
                            assert np.linalg.eigvalsh(rough).min()>-1e-10;counts['independent_laplacians']+=1
                            simple=np.eye(p)+(.1*p/trace*rough if trace>1e-12 else 0.)
                            jobs['equal_'+which]=inverse(simple,np.ones(p)/p)
                            if joint_valid:
                                cv=a[kind+'__covariance'];system=cv+(.1*np.trace(cv)/trace*rough if trace>1e-12 else 0.)
                                jobs[which]=inverse(system,a[kind+'__v'])
                        for core,(weight,ridge) in jobs.items():
                            arm=kind+'__'+core;detail=meta['methods'][arm]
                            # A recorded numeric failure is not silently turned
                            # into a success by this review.
                            assert detail['valid'] and not detail['fallback_to_augmented_iu'],arm
                            w,rule,flipped=orient.orient(weight,fit,anchor)
                            np.testing.assert_allclose(w,detail['standardized_weights'],atol=1e-10,rtol=1e-10)
                            assert detail['orientation_rule']==rule and detail['flipped']==flipped
                            if ridge is not None:
                                np.testing.assert_allclose(ridge,detail['inverse']['ridge'],atol=1e-10,rtol=1e-10)
                                assert detail['inverse']['target_condition']==100
                            risk=-z@w;stored=a[arm+'__window'];largest=max(largest,float(np.max(np.abs(risk-stored))))
                            np.testing.assert_allclose(risk,stored,atol=1e-10,rtol=1e-10);counts['independent_weight_projections']+=1
                            accum=np.zeros(rec['tokens']);support=np.zeros(rec['tokens'])
                            for lo,hi,value in zip(starts,ends,stored):accum[lo:hi]+=value;support[lo:hi]+=1
                            assert (support>0).all();token=accum/support;steps=np.array([token[lo:hi].max() for lo,hi in zip(ss,ee)])
                            # Source token projection uses difference arrays and
                            # cumsum; the independent direct-overlap sum has a
                            # different floating reduction order. Keep serialized
                            # score replay exact, and use the established 1e-12
                            # reconstruction tolerance only for the independent map.
                            np.testing.assert_allclose(a[arm+'__risk'],steps,atol=1e-12,rtol=1e-12)
                            np.testing.assert_array_equal(row['scores'][arm],a[arm+'__risk'])
                            counts['independent_step_reductions']+=1
                            samples=stored[fi,None]
                            with warnings.catch_warnings():
                                warnings.simplefilter('ignore')
                                gm=[GaussianMixture(n_components=k,n_init=3,max_iter=300,reg_covar=1e-4,random_state=2026090705).fit(samples) for k in (1,2)]
                            if all(g.converged_ for g in gm):
                                bic=[g.bic(samples) for g in gm];opened=bic[1]<bic[0] and np.any(steps>gm[1].means_.mean())
                                assert detail['decision_valid'] and detail['prediction']==(int(np.argmax(steps)) if opened else -1)
                                np.testing.assert_allclose(detail['gate']['bic'],bic,atol=1e-8,rtol=1e-10)
                            else:assert not detail['decision_valid']
                            counts['independent_native_decisions']+=1
                        for core in driver.CORES:
                            arm=kind+'__'+core;d=meta['methods'][arm]
                            expected_fallback=core in ('joint0','graph010','graph_perm') and not joint_valid
                            assert d['fallback_to_augmented_iu']==expected_fallback and d['bank']==bank
                            source='iu' if expected_fallback else core;assert d['source_arm']==kind+'__native_'+source
                            if expected_fallback:
                                fallback[arm]+=1
                                for key in ('valid','decision_valid','fixed_iu_valid','prediction','fixed_iu_prediction','peak'):
                                    assert d.get(key)==meta['methods'][kind+'__iu'].get(key)
                            if d['valid']:
                                coverage[arm]+=1
                                for suffix in ('window','risk'):np.testing.assert_array_equal(a[arm+'__'+suffix],a[kind+'__native_'+source+'__'+suffix])
                                reference=om['methods']['moment__iu'];valid=bool(reference['valid'] and reference['decision_valid'])
                                assert d['fixed_iu_valid']==valid
                                if valid:assert d['fixed_iu_prediction']==(int(np.argmax(a[arm+'__risk'])) if reference['prediction']!=-1 else -1)
                            for key in ('valid','decision_valid','fixed_iu_valid'):assert row[key][arm]==d[key]
                            assert row['predictions'][arm]==d.get('prediction') and row['sources'][arm]==d['source_arm']
                            counts['fixed_bank_fallback_and_row_checks']+=1
        print('Reviewed fits, maps, decisions and anchors:',cell,flush=True)
    for arm,metric in e['metrics'].items():
        metrics.check_equal(metric,{'prm':metrics.prm(e['rows'],arm),'pb':metrics.pb(e['rows'],arm),'pb_common_iu_gate':metrics.pb(e['rows'],arm,True)})
        counts['metric_bundles']+=1
        if arm in driver.PARENT_ARMS:assert metric==parent['metrics'][arm];counts['parent_metric_replays']+=1
    for pair in c['pairs'].values():
        if pair['scope']=='all':selected=e['rows']
        else:
            mode,kind=pair['scope'].split(':')
            selected=[r for r in e['rows'] if r['augmentation']['native_joint_valid'][kind]
                and (mode!='native_and_original' or r['augmentation']['original_joint_valid'])]
        assert pair['selected_ids']==[r['uid'] for r in selected] and pair['selected_answers']==len(selected)
        common=[r for r in selected if r['valid'][pair['left']] and r['valid'][pair['right']]]
        for side in ('left','right'):
            arm=pair[side];metrics.check_equal(pair[side+'_prm'],metrics.prm(common,arm))
            metrics.check_equal(pair[side+'_pb'],metrics.pb(selected,arm))
            metrics.check_equal(pair[side+'_pb_common_iu_gate'],metrics.pb(selected,arm,True))
        counts['paired_point_scope_bundles']+=1
    checks=[]
    for left,right,scope in [('ar1__graph010','dual__cond100_graph010','all'),('ar1__iu','dual__iu','all'),
        ('ar1__graph010','ar1__equal_graph010','all'),('ar1__graph010','last__graph010','all'),
        ('ar1__graph010','ar1__iu','native:ar1'),('ar1__graph010','dual__cond100_graph010','native_and_original:ar1')]:
        rows_for=driver.select_rows(e['rows'],scope)
        actual=c['pairs'][driver.pair_key(dict(left=left,right=right,scope=scope))]['uncertainty']
        explicit=metrics.explicit_bootstrap(rows_for,left,right)
        for key,value in explicit.items():metrics.check_equal(actual[key],value)
        checks.append(dict(left=left,right=right,scope=scope,draws=1000,status='MATCH'))
        print('Explicit intervals match:',left,right,scope,flush=True)
    report={'status':'PASS','counts':dict(counts),'coverage':dict(coverage),'fixed_banks':dict(banks),
        'native_joint_coverage':dict(native),'fallback':dict(fallback),'K_counts':group_sizes,
        'representative_grouping_refits':[list(x) for x in sorted(grouping_checks)],
        'representative_joint_refits':[list(x) for x in sorted(fit_checks)],
        'representative_gate_refits':[list(x) for x in sorted(gate_checks)],
        'bootstrap_checks':checks,'maximum_risk_difference':largest,'seconds':time.monotonic()-start,
        'scope':'Same-session review. Independent covariance, inverse/Laplacian algebra, routing, step/decision and metric reconstruction. Source grouping/Joint/IU/DUFS/graph/Jacobian/GMM kernels reused for specified replays.',
        'hashes':{str(OUT/n):sha(OUT/n) for n in ('MANIFEST.json','SCORES_FROZEN.json','EVALUATION.json','CONTRASTS.json')},
        'review_dependencies':{str(ROOT/'scripts'/n):sha(ROOT/'scripts'/n) for n in ('run_fusion_prediction_quality_v1.py','review_fusion_explicit_fallback_v1.py','review_fusion_context_bank_v1.py')},
        'review_script_sha256':sha(__file__)}
    save(OUT/'REVIEW.json',report);print('Review PASS',dict(counts),flush=True)


if __name__=='__main__':main()
