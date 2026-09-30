"""Independent endpoint/route/graph-head review of checked pair localization."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
from sklearn.mixture import GaussianMixture
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/fusion_pair_quality_v1';PARENT=ROOT/'results/fusion_replication_v1';AUDIT=ROOT/'results/joint_pair_identifiability_audit_v1'
sys.path.insert(0,str(ROOT/'local_cache/short_cycle01_code'))
import spectral_utils
spectral_utils.__path__.append(str(ROOT/'spectral_utils'))
from spectral_utils.laplacian_upcr import build_graph_from_features
from spectral_utils.adapted_dufs import adapted_dufs_soft_gates
from spectral_utils.joint_pair_jacobian import profiled_pair_jacobian


def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(p,name):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def review():
    started=time.monotonic();m,f,e,c=[load(OUT/n) for n in ('MANIFEST.json','SCORES_FROZEN.json','EVALUATION.json','CONTRASTS.json')]
    assert f['manifest_sha256']==sha(OUT/'MANIFEST.json') and e['scores_sha256']==sha(OUT/'SCORES_FROZEN.json')
    assert c['evaluation_sha256']==sha(OUT/'EVALUATION.json') and c['status']=='COMPLETE' and len(c['pairs'])==63
    assert len(m['arms'])==33 and not f['labels_decoded']
    for path,h in {**m['hashes'],**f['files']}.items():assert sha(path)==h,path
    metrics=module(ROOT/'scripts/review_fusion_explicit_fallback_v1.py','independent_quality_metrics')
    inv=module(ROOT/'scripts/review_fusion_context_bank_v1.py','independent_quality_inverse')
    pair_review=module(ROOT/'scripts/review_joint_pairs_v1.py','independent_product_profile')
    previous=load(PARENT/'EVALUATION.json');prevrows={r['uid']:r for r in previous['rows']}
    rows={r['uid']:r for r in e['rows']};assert len(rows)==len(e['rows'])==110
    release=load(ROOT/'results/localization_source_group_audit_v1/RELEASE_V2.json')
    counts=Counter();coverage=Counter();routing=Counter();gates_refit=set();largest=0.
    for cell in sorted({r['cell'] for r in m['selected']}):
        info=release['cells'][cell]
        with np.load(info['label_path'],allow_pickle=False) as labels:
            positions={str(v):i for i,v in enumerate(labels['row_ids'])};assert len(positions)==len(labels['row_ids'])
            for rec in (r for r in m['selected'] if r['cell']==cell):
                uid=rec['uid'];row=rows[uid];idx=positions[rec['row_id']]
                if cell.startswith('prm'):
                    a,b=labels['step_flag_offsets'][idx:idx+2];target=labels['step_error_flags'][a:b]
                else:target=int(labels['first_error'][idx])
                np.testing.assert_array_equal(row['target'],target);np.testing.assert_array_equal(row['target'],prevrows[uid]['target'])
                assert row['group_id']==prevrows[uid]['group_id'];counts['direct_label_group_joins']+=1
                meta=load(OUT/'scores'/(uid+'.json'));parent=load(PARENT/'scores'/(uid+'.json'));audit=load(AUDIT/'rows'/(uid+'.json'))
                assert not meta['labels_decoded']
                with np.load(OUT/'scores'/(uid+'.npz'),allow_pickle=False) as ar,np.load(PARENT/'scores'/(uid+'.npz'),allow_pickle=False) as old,np.load(AUDIT/'rows'/(uid+'.npz'),allow_pickle=False) as oldfit:
                    for key in old.files:np.testing.assert_array_equal(ar[key],old[key]);counts['exact_parent_arrays']+=1
                    for arm,d in parent['methods'].items():
                        assert meta['methods'][arm]==d;counts['exact_parent_metadata']+=1
                        for key in ('valid','decision_valid','fixed_iu_valid','predictions','fixed_iu_predictions','peaks','sources'):
                            assert row[key][arm]==prevrows[uid][key][arm]
                        if d['valid']:np.testing.assert_array_equal(row['scores'][arm],prevrows[uid]['scores'][arm])
                    fi=ar['fit_indices'];starts=ar['window_starts'];ends=ar['window_ends'];ss=ar['step_starts'];ee=ar['step_ends']
                    for bank in ('moment','context'):
                        bd=meta['diagnostics']['banks'][bank];ad=audit['banks'][bank];group=np.array(bd['grouping']['labels'])
                        assert bd['grouping']=={k:ad['grouping'][k] for k in bd['grouping']};counts['exact_audit_groupings']+=1
                        normal=bd['normalization'];names=parent['diagnostics']['banks'][bank]['names'];cols=[names.index(n) for n in normal['active_features']]
                        x=ar[bank+'__features'][:,cols];fitx=x[fi];mu=fitx.mean(0);sd=fitx.std(0)
                        np.testing.assert_allclose(normal['mean'],mu,atol=1e-11);np.testing.assert_allclose(normal['sd'],sd,atol=1e-11)
                        z=(x-mu)/sd;z-=z[fi].mean(0);z*=np.asarray(normal['feature_signs']);fit=z[fi]
                        anchor=normal['active_features'].index(normal['anchor_feature'])
                        for core in ('joint0','graph010','graph_perm'):assert meta['methods']['pair_'+bank+'__'+core]['valid']==ad['valid']
                        if bank+'__covariance' in oldfit.files:
                            for suffix in ('covariance','v','u'):np.testing.assert_allclose(ar['pair_'+bank+'__'+suffix],oldfit[bank+'__'+suffix],atol=1e-10,rtol=1e-10)
                            counts['audit_factor_covariance_replays']+=1
                        if not ad['valid']:continue
                        cv=ar['pair_'+bank+'__covariance'];v=ar['pair_'+bank+'__v'];u=ar['pair_'+bank+'__u']
                        jac=profiled_pair_jacobian(v,u,group)
                        assert jac['full_global_rank'] and jac['condition_number']<=1e8
                        if min(Counter(group).values())==2:
                            ij=pair_review.independent_product_profile(v,u,group);assert ij[0] and ij[1]==jac['rank']
                            np.testing.assert_allclose(ij[2],jac['condition_number'],atol=1e-7,rtol=1e-7);counts['independent_pair_product_jacobians']+=1
                        gates=np.asarray(bd['gates']['values']);oldg=parent['diagnostics']['banks'][bank]['shared'].get('gates')
                        if oldg:np.testing.assert_array_equal(gates,oldg['values']);counts['same_answer_gate_replays']+=1
                        elif (cell,bank) not in gates_refit:
                            recomputed,_=adapted_dufs_soft_gates(fit.T,seeds=(0,1,2),epochs=120)
                            np.testing.assert_allclose(gates,recomputed,atol=1e-12);gates_refit.add((cell,bank))
                        W=build_graph_from_features(fit.T,gates=gates,k=7).toarray()
                        seed=int(hashlib.sha256((m['scoring_namespace']+'/'+cell+'/'+rec['row_id']+'/moments27_local8').encode()).hexdigest()[:8],16)
                        assert seed==bd['gates']['seed'];perm=np.random.default_rng(seed).permutation(len(fit));np.testing.assert_array_equal(perm,bd['gates']['permutation'])
                        for core,lam,graph in [('joint0',0.,W),('graph010',.1,W),('graph_perm',.1,W[perm][:,perm])]:
                            degree=np.maximum(graph.sum(1),1e-12);lap=np.eye(len(graph))-graph/np.sqrt(np.outer(degree,degree))
                            rough=fit.T@lap@fit/len(fit);rough=(rough+rough.T)/2
                            scale=np.trace(cv)/np.trace(rough) if np.trace(rough)>1e-12 else 0.
                            weight,_=inv.project_inverse(cv+lam*scale*rough,v);weight,_,_=inv.orient(weight,fit,anchor)
                            arm='pair_'+bank+'__'+core;detail=meta['methods'][arm]
                            np.testing.assert_allclose(weight,detail['standardized_weights'],atol=1e-8,rtol=1e-8)
                            risk=-z@weight;np.testing.assert_allclose(risk,ar[arm+'__window'],atol=1e-8,rtol=1e-8)
                            largest=max(largest,float(np.max(np.abs(risk-ar[arm+'__window']))));counts['independent_graph_inverse_projections']+=1
                            sums=np.zeros(rec['tokens']);support=np.zeros(rec['tokens'])
                            for a,b,val in zip(starts,ends,ar[arm+'__window']):sums[a:b]+=val;support[a:b]+=1
                            assert (support>0).all();token=sums/support;steps=np.array([token[a:b].max() for a,b in zip(ss,ee)])
                            np.testing.assert_allclose(row['scores'][arm],steps,atol=1e-12);counts['independent_window_step_maps']+=1
                            rr=ar[arm+'__window'][fi,None]
                            models=[GaussianMixture(n_components=k,n_init=3,max_iter=300,reg_covar=1e-4,random_state=2026090705).fit(rr) for k in (1,2)]
                            if all(x.converged_ for x in models):
                                bic=[x.bic(rr) for x in models];open_gate=bic[1]<bic[0] and np.any(steps>models[1].means_.mean())
                                assert detail['decision_valid'] and detail['prediction']==(int(np.argmax(steps)) if open_gate else -1)
                                np.testing.assert_allclose(detail['gate']['bic'],bic,atol=1e-8)
                            else:assert not detail['decision_valid']
                            counts['independent_native_gmm_decisions']+=1
                    mv=meta['methods']['pair_moment__joint0']['valid'];cv=meta['methods']['pair_context__joint0']['valid']
                    routes={'single':'moment_joint' if mv else 'moment_iu','dual':'moment_joint' if mv else ('context_joint' if cv else 'moment_iu')}
                    assert meta['routing']=={'eligibility':{'moment':mv,'context':cv},'routes':routes}
                    for policy,route in routes.items():routing[policy+'/'+route]+=1
                    for arm,d in meta['methods'].items():
                        for field in ('valid','decision_valid','fixed_iu_valid'):assert row[field][arm]==d[field]
                        assert row['predictions'][arm]==d.get('prediction') and row['fixed_iu_predictions'][arm]==d.get('fixed_iu_prediction')
                        if d['valid']:coverage[arm]+=1
                        if arm.startswith(('pair_single__','pair_dual__')):
                            policy,core=arm.removeprefix('pair_').split('__');route=routes[policy];bank='context' if route=='context_joint' else 'moment'
                            source=bank+'__'+core if core in ('equal','iu') else ('moment__iu' if route=='moment_iu' else 'pair_'+bank+'__'+core)
                            assert d['source_arm']==source
                            for key in ('valid','decision_valid','fixed_iu_valid','prediction','fixed_iu_prediction','peak'):assert d.get(key)==meta['methods'][source].get(key)
                            if d['valid']:
                                for suffix in ('window','risk'):np.testing.assert_array_equal(ar[arm+'__'+suffix],ar[source+'__'+suffix])
                            counts['independent_route_inheritances']+=1
                        if arm.startswith('pair_') and d['valid']:
                            ref=meta['methods']['moment__iu'];assert d['fixed_iu_valid']==bool(ref['valid'] and ref['decision_valid'])
                            if d['fixed_iu_valid']:assert d['fixed_iu_prediction']==(int(np.argmax(ar[arm+'__risk'])) if ref['prediction']!=-1 else -1)
        print('Reviewed label joins, graphs, gates and routing:',cell,flush=True)
    for arm,x in e['metrics'].items():
        metrics.check_equal(x,{'prm':metrics.prm(e['rows'],arm),'pb':metrics.pb(e['rows'],arm),'pb_common_iu_gate':metrics.pb(e['rows'],arm,True)})
        counts['metric_bundles']+=1
        if arm in previous['metrics']:assert x==previous['metrics'][arm];counts['historical_parent_metric_replays']+=1
    for pair in c['pairs'].values():
        l,r=pair['left'],pair['right'];common=[row for row in e['rows'] if row['valid'][l] and row['valid'][r]]
        for side,arm in [('left',l),('right',r)]:
            metrics.check_equal(pair[side+'_prm'],metrics.prm(common,arm));metrics.check_equal(pair[side+'_pb'],metrics.pb(e['rows'],arm))
            metrics.check_equal(pair[side+'_pb_common_iu_gate'],metrics.pb(e['rows'],arm,True))
        counts['paired_point_bundles']+=1
    checks=[]
    for left,right in [('pair_single__joint0','single__joint0'),('pair_single__joint0','moment__iu'),
                       ('pair_dual__graph010','dual__graph010'),('pair_dual__graph010','pair_dual__joint0'),
                       ('pair_dual__graph010','pair_dual__graph_perm'),('pair_dual__graph010','pair_dual__iu')]:
        explicit=metrics.explicit_bootstrap(e['rows'],left,right);actual=c['pairs'][left+' minus '+right]['uncertainty']
        for key,value in explicit.items():metrics.check_equal(actual[key],value)
        checks.append({'left':left,'right':right,'draws':1000,'four_intervals_and_counts':'MATCH'})
        print('Explicit bootstrap matches:',left,'minus',right,flush=True)
    report={'status':'PASS','counts':dict(counts),'coverage':dict(coverage),'routes':dict(routing),
        'new_gate_recipe_refits':[list(x) for x in sorted(gates_refit)],'bootstrap_checks':checks,
        'max_reconstructed_risk_difference':largest,'seconds':time.monotonic()-started,
        'scope':'Independent graph-Laplacian/native inverse, step/GMM/routing/metric reconstruction; audited Joint fits, graph-builder and DUFS kernels reused.',
        'hashes':{str(OUT/n):sha(OUT/n) for n in ('MANIFEST.json','SCORES_FROZEN.json','EVALUATION.json','CONTRASTS.json')},
        'review_dependencies':{str(ROOT/'scripts'/n):sha(ROOT/'scripts'/n) for n in ('review_fusion_explicit_fallback_v1.py','review_fusion_context_bank_v1.py','review_joint_pairs_v1.py')},
        'review_script_sha256':sha(__file__)}
    (OUT/'REVIEW.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('status','counts','coverage','routes','seconds')},indent=2),flush=True)


if __name__=='__main__':review()
