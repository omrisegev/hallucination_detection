"""Independent graph, inverse, simple-control and fixed-route quality review."""
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
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/fusion_graph_conditioning_v1'
PARENT=ROOT/'results/fusion_native_conditioning_v1';ORIGINAL=ROOT/'results/fusion_replication_v1'
sys.path.insert(0,str(ROOT/'local_cache/short_cycle01_code'))
import spectral_utils
spectral_utils.__path__.append(str(ROOT/'spectral_utils'))
from spectral_utils.answer_localization_v2 import prepare_local
from spectral_utils.laplacian_upcr import build_graph_from_features
from spectral_utils.adapted_dufs import adapted_dufs_soft_gates

def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(p,name):
    spec=importlib.util.spec_from_file_location(name,p);obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj


def inverse(system,loading,cap):
    ev,q=np.linalg.eigh((system+system.T)/2);ev=np.maximum(ev,0.);psd=(q*ev)@q.T
    lo,hi=np.linalg.eigvalsh(psd)[[0,-1]]
    ridge=1. if hi<=1e-14 else max(0.,(hi-cap*lo)/(cap-1),hi*1e-10)
    return np.linalg.solve(psd+ridge*np.eye(len(loading)),loading),ridge


def review():
    started=time.monotonic();m,f,e,c=[load(OUT/n) for n in ('MANIFEST.json','SCORES_FROZEN.json','EVALUATION.json','CONTRASTS.json')]
    assert f['manifest_sha256']==sha(OUT/'MANIFEST.json') and e['scores_sha256']==sha(OUT/'SCORES_FROZEN.json')
    assert c['evaluation_sha256']==sha(OUT/'EVALUATION.json') and c['status']=='COMPLETE'
    assert len(m['arms'])==77 and len(c['pairs'])==101 and m['conditions']==[30,100,300] and not f['labels_decoded']
    for p,h in {**m['hashes'],**f['files']}.items():assert sha(p)==h,p
    metrics=module(ROOT/'scripts/review_fusion_explicit_fallback_v1.py','graph_condition_review_metrics')
    orientation=module(ROOT/'scripts/review_fusion_context_bank_v1.py','graph_condition_review_orientation')
    previous=load(PARENT/'EVALUATION.json');prevrows={r['uid']:r for r in previous['rows']}
    rows={r['uid']:r for r in e['rows']};assert len(rows)==len(e['rows'])==110
    release=load(ROOT/'results/localization_source_group_audit_v1/RELEASE_V2.json')
    counts=Counter();coverage=Counter();routes=Counter();gate_refits=set();largest=0.;max_control_condition=0.
    for cell in sorted({r['cell'] for r in m['selected']}):
        with np.load(release['cells'][cell]['label_path'],allow_pickle=False) as lab:
            positions={str(v):i for i,v in enumerate(lab['row_ids'])};assert len(positions)==len(lab['row_ids'])
            for rec in (r for r in m['selected'] if r['cell']==cell):
                uid=rec['uid'];row=rows[uid];idx=positions[rec['row_id']]
                if cell.startswith('prm'):
                    a,b=lab['step_flag_offsets'][idx:idx+2];target=lab['step_error_flags'][a:b]
                else:target=int(lab['first_error'][idx])
                np.testing.assert_array_equal(row['target'],target);np.testing.assert_array_equal(row['target'],prevrows[uid]['target'])
                assert row['group_id']==prevrows[uid]['group_id'];counts['direct_label_group_joins']+=1
                meta=load(OUT/'scores'/(uid+'.json'));pm=load(PARENT/'scores'/(uid+'.json'));om=load(ORIGINAL/'scores'/(uid+'.json'))
                assert not meta['labels_decoded'] and meta['diagnostics']['joint_refits']==0
                assert meta['routing']==om['routing']==row['routing']==pm['routing']
                with np.load(OUT/'scores'/(uid+'.npz'),allow_pickle=False) as ar,np.load(PARENT/'scores'/(uid+'.npz'),allow_pickle=False) as pa:
                    for key in pa.files:np.testing.assert_array_equal(ar[key],pa[key]);counts['exact_parent_arrays']+=1
                    for arm,d in pm['methods'].items():
                        assert meta['methods'][arm]==d;counts['exact_parent_metadata']+=1
                        for key in ('valid','decision_valid','fixed_iu_valid','predictions','fixed_iu_predictions','peaks','sources'):assert row[key][arm]==prevrows[uid][key][arm]
                        if d['valid']:np.testing.assert_array_equal(row['scores'][arm],prevrows[uid]['scores'][arm])
                    fi=ar['fit_indices'];starts=ar['window_starts'];ends=ar['window_ends'];ss=ar['step_starts'];ee=ar['step_ends']
                    for bank in ('moment','context'):
                        sh=om['diagnostics']['banks'][bank]['shared'];names=om['diagnostics']['banks'][bank]['names'];bd=meta['diagnostics']['banks'][bank]
                        cols=[names.index(n) for n in sh['active_features']];x=ar[bank+'__features'][:,cols];fitx=x[fi]
                        mu=fitx.mean(0);sd=fitx.std(0);np.testing.assert_allclose(sh['mean'],mu,atol=1e-12);np.testing.assert_allclose(sh['sd'],sd,atol=1e-12)
                        z=(x-mu)/sd;z-=z[fi].mean(0);z*=np.asarray(sh['feature_signs']);fit=z[fi];p=fit.shape[1]
                        anchor=sh['active_features'].index(sh['anchor_feature']);rz,ra,_=prepare_local(ar[bank+'__features'],names,fi)
                        np.testing.assert_allclose(z,rz,atol=1e-12,rtol=1e-12);assert ra==anchor
                        gates=np.asarray(bd['gates']['values']);np.testing.assert_array_equal(gates,ar[bank+'__graph_gates'])
                        if sh.get('gates'):
                            np.testing.assert_array_equal(gates,sh['gates']['values']);counts['original_gate_replays']+=1
                        elif (cell,bank) not in gate_refits:
                            gg,_=adapted_dufs_soft_gates(rz[fi].T,seeds=(0,1,2),epochs=120)
                            np.testing.assert_allclose(gates,gg,atol=1e-12,rtol=1e-12);gate_refits.add((cell,bank))
                        # Reuse source graph builder on exact source-recipe inputs;
                        # Laplacian, trace matching and all solves are independent.
                        graph=build_graph_from_features(rz[fi].T,gates=gates,k=7).toarray();counts['source_graph_reconstructions']+=1
                        seed=int(hashlib.sha256((m['scoring_namespace']+'/'+cell+'/'+rec['row_id']+'/moments27_local8').encode()).hexdigest()[:8],16)
                        permutation=np.random.default_rng(seed).permutation(len(fit));assert bd['seed']==seed
                        np.testing.assert_array_equal(permutation,bd['permutation'])
                        if sh.get('gates'):assert seed==sh['gates']['seed'];np.testing.assert_array_equal(permutation,sh['gates']['permutation'])
                        equal=np.ones(p)/p;ew,_,_=orientation.orient(equal,fit,anchor)
                        np.testing.assert_allclose(-z@ew,ar[bank+'__equal__window'],atol=1e-10,rtol=1e-10);counts['equal_zero_graph_replays']+=1
                        native_valid=om['methods'][bank+'__joint0']['valid'];assert bd['original_joint_valid']==native_valid
                        for kind,wgraph in [('graph010',graph),('graph_perm',graph[permutation][:,permutation])]:
                            degree=np.maximum(wgraph.sum(1),1e-12);lap=np.eye(len(fit))-wgraph/np.sqrt(np.outer(degree,degree))
                            rough=fit.T@lap@fit/len(fit);rough=(rough+rough.T)/2;trace=np.trace(rough)
                            assert np.linalg.eigvalsh(rough).min()>-1e-10;counts['independent_laplacians']+=1
                            control_system=np.eye(p)+(.1*p/trace*rough if trace>1e-12 else 0.)
                            control_weights=[]
                            for cap in (30,100,300,1000):
                                w,ridge=inverse(control_system,equal,cap);control_weights.append(w)
                            for w in control_weights[1:]:np.testing.assert_array_equal(w,control_weights[0])
                            cond=float(np.linalg.cond(control_system));assert cond<=1+.1*p+1e-8
                            max_control_condition=max(max_control_condition,cond);counts['equal_graph_cap_invariance']+=1
                            jobs=[(bank+'__equal_'+kind,control_weights[0],ridge,1000)]
                            if native_valid:
                                cv=ar['original_'+bank+'__covariance'];v=ar['original_'+bank+'__v']
                                system=cv+(.1*np.trace(cv)/trace*rough if trace>1e-12 else 0.)
                                for cap in (1000,30,100,300):
                                    arm=bank+'__'+kind if cap==1000 else f'{bank}__cond{cap}_{kind}'
                                    w,ridge=inverse(system,v,cap);jobs.append((arm,w,ridge,cap))
                            else:
                                for cap in (30,100,300):assert not meta['methods'][f'{bank}__cond{cap}_{kind}']['valid']
                                counts['invalid_native_graph_families_preserved']+=1
                            for arm,w,ridge,cap in jobs:
                                d=meta['methods'][arm];w,rule,flipped=orientation.orient(w,fit,anchor)
                                np.testing.assert_allclose(w,d['standardized_weights'],atol=1e-10,rtol=1e-10)
                                np.testing.assert_allclose(ridge,d['inverse']['ridge'],atol=1e-11,rtol=1e-10)
                                assert d['inverse']['target_condition']==cap and d['inverse']['lambda']==.1
                                assert d['inverse']['condition_after']<=cap*(1+1e-9) and d['orientation_rule']==rule and d['flipped']==flipped
                                risk=-z@w;np.testing.assert_allclose(risk,ar[arm+'__window'],atol=1e-10,rtol=1e-10)
                                largest=max(largest,float(np.max(np.abs(risk-ar[arm+'__window']))));counts['independent_inverse_projections']+=1
                                accum=np.zeros(rec['tokens']);support=np.zeros(rec['tokens'])
                                for a,b,value in zip(starts,ends,ar[arm+'__window']):accum[a:b]+=value;support[a:b]+=1
                                assert (support>0).all();token=accum/support;step=np.array([token[a:b].max() for a,b in zip(ss,ee)])
                                np.testing.assert_allclose(row['scores'][arm],step,atol=1e-12);counts['independent_step_maps']+=1
                                rr=ar[arm+'__window'][fi,None]
                                gm=[GaussianMixture(n_components=k,n_init=3,max_iter=300,reg_covar=1e-4,random_state=2026090705).fit(rr) for k in (1,2)]
                                if all(g.converged_ for g in gm):
                                    bic=[g.bic(rr) for g in gm];opened=bic[1]<bic[0] and np.any(step>gm[1].means_.mean())
                                    assert d['decision_valid'] and d['prediction']==(int(np.argmax(step)) if opened else -1)
                                    np.testing.assert_allclose(d['gate']['bic'],bic,atol=1e-8)
                                else:assert not d['decision_valid']
                                counts['independent_native_gmm_decisions']+=1
                    for family,route in om['routing']['routes'].items():routes[family+'/'+route]+=1
                    for arm,d in meta['methods'].items():
                        for key in ('valid','decision_valid','fixed_iu_valid'):assert row[key][arm]==d[key]
                        assert row['predictions'][arm]==d.get('prediction') and row['fixed_iu_predictions'][arm]==d.get('fixed_iu_prediction')
                        if d['valid']:coverage[arm]+=1
                        if arm in pm['methods']:continue
                        family,suffix=arm.split('__')
                        if family in ('single','dual'):
                            route=om['routing']['routes'][family];bank='context' if route=='context_joint' else 'moment'
                            source=bank+'__'+suffix if suffix.startswith('equal_') or route!='moment_iu' else 'moment__iu'
                            assert d['source_arm']==source and d['route']==route
                            for key in ('valid','decision_valid','fixed_iu_valid','prediction','fixed_iu_prediction','peak'):assert d.get(key)==meta['methods'][source].get(key)
                            if d['valid']:
                                for ending in ('window','risk'):np.testing.assert_array_equal(ar[arm+'__'+ending],ar[source+'__'+ending])
                            counts['fixed_route_inheritances']+=1
                        if d['valid']:
                            ref=meta['methods']['moment__iu'];assert d['fixed_iu_valid']==bool(ref['valid'] and ref['decision_valid'])
                            if d['fixed_iu_valid']:assert d['fixed_iu_prediction']==(int(np.argmax(ar[arm+'__risk'])) if ref['prediction']!=-1 else -1)
        print('Reviewed graphs, native/equal maps, labels and routes:',cell,flush=True)
    for arm,x in e['metrics'].items():
        metrics.check_equal(x,{'prm':metrics.prm(e['rows'],arm),'pb':metrics.pb(e['rows'],arm),'pb_common_iu_gate':metrics.pb(e['rows'],arm,True)})
        counts['metric_bundles']+=1
        if arm in previous['metrics']:assert x==previous['metrics'][arm];counts['parent_metric_replays']+=1
    for pair in c['pairs'].values():
        l,r=pair['left'],pair['right'];common=[row for row in e['rows'] if row['valid'][l] and row['valid'][r]]
        for side,arm in [('left',l),('right',r)]:
            metrics.check_equal(pair[side+'_prm'],metrics.prm(common,arm));metrics.check_equal(pair[side+'_pb'],metrics.pb(e['rows'],arm))
            metrics.check_equal(pair[side+'_pb_common_iu_gate'],metrics.pb(e['rows'],arm,True))
        counts['paired_point_bundles']+=1
    checks=[]
    for left,right in [('dual__cond30_graph010','dual__cond30_graph_perm'),('dual__cond30_graph010','dual__cond30'),
        ('dual__cond30_graph010','dual__iu'),('dual__cond30_graph010','dual__equal_graph010'),
        ('dual__equal_graph010','dual__equal'),('dual__cond100_graph010','dual__iu'),('moment__cond300_graph010','moment__cond300')]:
        explicit=metrics.explicit_bootstrap(e['rows'],left,right);actual=c['pairs'][left+' minus '+right]['uncertainty']
        for key,value in explicit.items():metrics.check_equal(actual[key],value)
        checks.append({'left':left,'right':right,'draws':1000,'four_intervals_and_counts':'MATCH'})
        print('Explicit bootstrap matches:',left,'minus',right,flush=True)
    report={'status':'PASS','counts':dict(counts),'coverage':dict(coverage),'routes':dict(routes),
        'new_equal_control_gate_refits':[list(x) for x in sorted(gate_refits)],'bootstrap_checks':checks,
        'max_reconstructed_risk_difference':largest,'maximum_equal_graph_condition':max_control_condition,'seconds':time.monotonic()-started,
        'scope':'Source graph-builder/DUFS/GMM kernels reused. Independent Laplacians, trace matching, inverse solves, simple-control invariance, step/decision/routing/metric reconstruction. No Joint refits.',
        'hashes':{str(OUT/n):sha(OUT/n) for n in ('MANIFEST.json','SCORES_FROZEN.json','EVALUATION.json','CONTRASTS.json')},
        'review_dependencies':{str(ROOT/'scripts'/n):sha(ROOT/'scripts'/n) for n in ('review_fusion_explicit_fallback_v1.py','review_fusion_context_bank_v1.py')},
        'review_script_sha256':sha(__file__)}
    (OUT/'REVIEW.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('status','counts','routes','seconds')},indent=2),flush=True)


if __name__=='__main__':review()
