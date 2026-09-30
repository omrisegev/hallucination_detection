"""Independent covariance/inverse/decision review of original Joint conditioning."""
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
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/fusion_native_conditioning_v1'
PARENT=ROOT/'results/fusion_pair_quality_v1';ORIGINAL=ROOT/'results/fusion_replication_v1'
sys.path.insert(0,str(ROOT/'local_cache/short_cycle01_code'))
import spectral_utils
spectral_utils.__path__.append(str(ROOT/'spectral_utils'))
from spectral_utils.joint_lsml import fit_joint_lsml,covariance_matrix
from spectral_utils.answer_localization_v2 import prepare_local


def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(p,name):
    spec=importlib.util.spec_from_file_location(name,p);obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj


def review():
    started=time.monotonic();m,f,e,c=[load(OUT/n) for n in ('MANIFEST.json','SCORES_FROZEN.json','EVALUATION.json','CONTRASTS.json')]
    assert f['manifest_sha256']==sha(OUT/'MANIFEST.json') and e['scores_sha256']==sha(OUT/'SCORES_FROZEN.json')
    assert c['evaluation_sha256']==sha(OUT/'EVALUATION.json') and c['status']=='COMPLETE'
    assert len(m['arms'])==45 and len(c['pairs'])==69 and m['conditions']==[30,100,300] and not f['labels_decoded']
    for p,h in {**m['hashes'],**f['files']}.items():assert sha(p)==h,p
    metrics=module(ROOT/'scripts/review_fusion_explicit_fallback_v1.py','conditioning_review_metrics')
    inv=module(ROOT/'scripts/review_fusion_context_bank_v1.py','conditioning_review_orientation')
    parent=load(PARENT/'EVALUATION.json');parentrows={r['uid']:r for r in parent['rows']}
    original=load(ORIGINAL/'EVALUATION.json');originalrows={r['uid']:r for r in original['rows']}
    rows={r['uid']:r for r in e['rows']};assert len(rows)==len(e['rows'])==110
    release=load(ROOT/'results/localization_source_group_audit_v1/RELEASE_V2.json')
    counts=Counter();coverage=Counter();routes=Counter();refits=set();largest=0.;condition_data=[]
    for cell in sorted({r['cell'] for r in m['selected']}):
        with np.load(release['cells'][cell]['label_path'],allow_pickle=False) as lab:
            positions={str(v):i for i,v in enumerate(lab['row_ids'])};assert len(positions)==len(lab['row_ids'])
            for rec in (r for r in m['selected'] if r['cell']==cell):
                uid=rec['uid'];row=rows[uid];idx=positions[rec['row_id']]
                if cell.startswith('prm'):
                    a,b=lab['step_flag_offsets'][idx:idx+2];target=lab['step_error_flags'][a:b]
                else:target=int(lab['first_error'][idx])
                np.testing.assert_array_equal(row['target'],target);np.testing.assert_array_equal(row['target'],parentrows[uid]['target'])
                assert row['group_id']==originalrows[uid]['group_id'];counts['direct_label_group_joins']+=1
                meta=load(OUT/'scores'/(uid+'.json'));pm=load(PARENT/'scores'/(uid+'.json'));om=load(ORIGINAL/'scores'/(uid+'.json'))
                assert not meta['labels_decoded'] and meta['routing']==om['routing']==row['routing']
                with np.load(OUT/'scores'/(uid+'.npz'),allow_pickle=False) as ar,np.load(PARENT/'scores'/(uid+'.npz'),allow_pickle=False) as pa:
                    for key in pa.files:np.testing.assert_array_equal(ar[key],pa[key]);counts['exact_parent_arrays']+=1
                    for arm,d in pm['methods'].items():
                        assert meta['methods'][arm]==d;counts['exact_parent_metadata']+=1
                        for key in ('valid','decision_valid','fixed_iu_valid','predictions','fixed_iu_predictions','peaks','sources'):
                            assert row[key][arm]==parentrows[uid][key][arm]
                        if d['valid']:np.testing.assert_array_equal(row['scores'][arm],parentrows[uid]['scores'][arm])
                    fi=ar['fit_indices'];starts=ar['window_starts'];ends=ar['window_ends'];ss=ar['step_starts'];ee=ar['step_ends']
                    for bank in ('moment','context'):
                        od=om['methods'][bank+'__joint0'];bd=meta['diagnostics']['banks'][bank]
                        for condition in (30,100,300):assert meta['methods'][f'{bank}__cond{condition}']['valid']==od['valid']
                        if not od['valid']:
                            assert bd['status']=='ORIGINAL_INVALID_PRESERVED';counts['original_invalid_banks_preserved']+=1;continue
                        sh=om['diagnostics']['banks'][bank]['shared'];names=om['diagnostics']['banks'][bank]['names']
                        cols=[names.index(n) for n in sh['active_features']];x=ar[bank+'__features'][:,cols];fitx=x[fi]
                        mu=fitx.mean(0);sd=fitx.std(0);np.testing.assert_allclose(sh['mean'],mu,atol=1e-12);np.testing.assert_allclose(sh['sd'],sd,atol=1e-12)
                        z=(x-mu)/sd;z-=z[fi].mean(0);z*=np.asarray(sh['feature_signs']);fit=z[fi]
                        groups=np.array(sh['joint']['groups']);np.testing.assert_array_equal(groups,bd['groups']);assert min(Counter(groups).values())>=3
                        anchor=sh['active_features'].index(sh['anchor_feature']);s=fit.T@fit/(len(fit)-1)
                        v=ar['original_'+bank+'__v'];u=ar['original_'+bank+'__u'];cv=ar['original_'+bank+'__covariance']
                        signal=np.outer(v,v)+np.outer(u,u)*(groups[:,None]==groups)
                        rebuilt=signal+np.diag(np.maximum(np.diag(s)-np.diag(signal),0.))
                        np.testing.assert_allclose(rebuilt,cv,atol=1e-12,rtol=1e-12);counts['independent_covariance_constructions']+=1
                        np.testing.assert_allclose(bd['relative_offdiag_misfit'],sh['joint']['relative_offdiag_misfit'],atol=1e-12,rtol=1e-10)
                        if (cell,bank) not in refits:
                            # Exact source-recipe replay uses its reduction order.
                            # The independent normalization above remains the
                            # input to all algebra/score/decision reconstructions.
                            rz,ra,_=prepare_local(ar[bank+'__features'],names,fi)
                            np.testing.assert_allclose(rz,z,atol=1e-12,rtol=1e-12);assert ra==anchor
                            jf=fit_joint_lsml(covariance_matrix(rz[fi]),groups,anchor_index=anchor,seed=2026090601,starts=5,max_sweeps=5000)
                            for value,expected in [(jf.model_covariance,cv),(jf.global_loading,v),(jf.group_loading,u)]:np.testing.assert_allclose(value,expected,atol=1e-11,rtol=1e-11)
                            assert jf.converged and jf.multistart_audit['status']=='PASS';refits.add((cell,bank))
                        ev,q=np.linalg.eigh((cv+cv.T)/2);ev=np.maximum(ev,0.);psd=(q*ev)@q.T
                        lo,hi=np.linalg.eigvalsh(psd)[[0,-1]];ridge_before=-1.
                        for condition in (1000,300,100,30):
                            arm=bank+'__joint0' if condition==1000 else f'{bank}__cond{condition}'
                            d=meta['methods'][arm];ridge=1. if hi<=1e-14 else max(0.,(hi-condition*lo)/(condition-1),hi*1e-10)
                            assert ridge>=ridge_before-1e-12;ridge_before=ridge
                            system=psd+ridge*np.eye(len(v));w=np.linalg.solve(system,v);w,rule,flipped=inv.orient(w,fit,anchor)
                            np.testing.assert_allclose(w,d['standardized_weights'],atol=1e-10,rtol=1e-10)
                            np.testing.assert_allclose(d['inverse']['ridge'],ridge,atol=1e-12,rtol=1e-10)
                            assert d['inverse']['target_condition']==condition and d['inverse']['lambda']==0.
                            assert d['inverse']['condition_after']<=condition*(1+1e-9)
                            assert d['orientation_rule']==rule and d['flipped']==flipped
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
                            condition_data.append({'uid':uid,'bank':bank,'condition':condition,'ridge':ridge,
                                'condition_after':float(np.linalg.cond(system)),
                                'weight_cosine_to_original':float(w@np.asarray(od['standardized_weights'])/np.linalg.norm(w)/np.linalg.norm(od['standardized_weights']))})
                    for policy,route in om['routing']['routes'].items():routes[policy+'/'+route]+=1
                    for arm,d in meta['methods'].items():
                        for key in ('valid','decision_valid','fixed_iu_valid'):assert row[key][arm]==d[key]
                        assert row['predictions'][arm]==d.get('prediction') and row['fixed_iu_predictions'][arm]==d.get('fixed_iu_prediction')
                        if d['valid']:coverage[arm]+=1
                        if '__cond' not in arm:continue
                        family,dose=arm.split('__')
                        if family in ('single','dual'):
                            route=om['routing']['routes'][family];bank='context' if route=='context_joint' else 'moment'
                            source='moment__iu' if route=='moment_iu' else bank+'__'+dose
                            assert d['source_arm']==source and d['route']==route
                            for key in ('valid','decision_valid','fixed_iu_valid','prediction','fixed_iu_prediction','peak'):assert d.get(key)==meta['methods'][source].get(key)
                            if d['valid']:
                                for suffix in ('window','risk'):np.testing.assert_array_equal(ar[arm+'__'+suffix],ar[source+'__'+suffix])
                            counts['fixed_route_inheritances']+=1
                        if d['valid']:
                            ref=meta['methods']['moment__iu'];assert d['fixed_iu_valid']==bool(ref['valid'] and ref['decision_valid'])
                            if d['fixed_iu_valid']:assert d['fixed_iu_prediction']==(int(np.argmax(ar[arm+'__risk'])) if ref['prediction']!=-1 else -1)
        print('Reviewed original fits, inverse doses, labels and routes:',cell,flush=True)
    for arm,x in e['metrics'].items():
        metrics.check_equal(x,{'prm':metrics.prm(e['rows'],arm),'pb':metrics.pb(e['rows'],arm),'pb_common_iu_gate':metrics.pb(e['rows'],arm,True)})
        counts['metric_bundles']+=1
        if arm in parent['metrics']:assert x==parent['metrics'][arm];counts['parent_metric_replays']+=1
    for pair in c['pairs'].values():
        l,r=pair['left'],pair['right'];common=[row for row in e['rows'] if row['valid'][l] and row['valid'][r]]
        for side,arm in [('left',l),('right',r)]:
            metrics.check_equal(pair[side+'_prm'],metrics.prm(common,arm));metrics.check_equal(pair[side+'_pb'],metrics.pb(e['rows'],arm))
            metrics.check_equal(pair[side+'_pb_common_iu_gate'],metrics.pb(e['rows'],arm,True))
        counts['paired_point_bundles']+=1
    checks=[]
    for left,right in [('dual__cond100','dual__joint0'),('dual__cond100','dual__iu'),('dual__cond100','dual__equal'),
                       ('dual__cond100','dual__graph010'),('context__cond30','context__equal'),('moment__cond300','moment__joint0')]:
        explicit=metrics.explicit_bootstrap(e['rows'],left,right);actual=c['pairs'][left+' minus '+right]['uncertainty']
        for key,value in explicit.items():metrics.check_equal(actual[key],value)
        checks.append({'left':left,'right':right,'draws':1000,'four_intervals_and_counts':'MATCH'})
        print('Explicit bootstrap matches:',left,'minus',right,flush=True)
    report={'status':'PASS','counts':dict(counts),'coverage':dict(coverage),'routes':dict(routes),
        'representative_original_refits':[list(x) for x in sorted(refits)],'bootstrap_checks':checks,
        'max_reconstructed_risk_difference':largest,'condition_diagnostics':condition_data,'seconds':time.monotonic()-started,
        'scope':'Independent covariance, inverse solves, window/step/GMM decisions, fixed routing and metrics. Original optimizer and sklearn GMM kernels reused.',
        'review_notes':['The first representative refit used independently normalized inputs and differed by up to 1.42e-11 in covariance. Exact source-recipe replay now preserves the original normalization/covariance reduction order; independent input/algebra checks remain. No frozen scorer, prediction, metric or tolerance changed.'],
        'hashes':{str(OUT/n):sha(OUT/n) for n in ('MANIFEST.json','SCORES_FROZEN.json','EVALUATION.json','CONTRASTS.json')},
        'review_dependencies':{str(ROOT/'scripts'/n):sha(ROOT/'scripts'/n) for n in ('review_fusion_explicit_fallback_v1.py','review_fusion_context_bank_v1.py')},
        'review_script_sha256':sha(__file__)}
    (OUT/'REVIEW.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('status','counts','routes','seconds')},indent=2),flush=True)


if __name__=='__main__':review()
