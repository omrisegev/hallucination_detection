"""Independent selection, input, score, gate, routing and metric review."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.signal import lfilter
from sklearn.mixture import GaussianMixture

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/fusion_replication_v1';SOURCE=ROOT/'results/localization_source_group_audit_v1'
sys.path.insert(0,str(ROOT/'local_cache/short_cycle01_code'))
import spectral_utils
spectral_utils.__path__.append(str(ROOT/'spectral_utils'))
from spectral_utils.answer_localization_v2 import prepare_local,JOINT_SEED
from spectral_utils.joint_lsml import fit_joint_lsml,covariance_matrix
from spectral_utils.laplacian_upcr import build_graph_from_features

def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def module(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def review():
    started=time.monotonic();m,f,e,c=[load(OUT/n) for n in ('MANIFEST.json','SCORES_FROZEN.json','EVALUATION.json','CONTRASTS.json')]
    assert f['manifest_sha256']==sha(OUT/'MANIFEST.json') and e['scores_sha256']==sha(OUT/'SCORES_FROZEN.json')
    assert c['status']=='COMPLETE' and len(c['pairs'])==38 and c['evaluation_sha256']==sha(OUT/'EVALUATION.json')
    for p,h in {**m['hashes'],**f['files']}.items():assert sha(p)==h,p
    release=load(SOURCE/'RELEASE_V2.json');old=load(SOURCE/'EVALUATION_V2.json')
    metrics=module(ROOT/'scripts/review_fusion_explicit_fallback_v1.py','independent_metrics')
    inverse=module(ROOT/'scripts/review_fusion_context_bank_v1.py','independent_inverse')
    counts=Counter();selected=[];used=set(m['excluded_components']);support=[]
    # Reconstruct selection independently of the production selector.
    htext=lambda s:hashlib.sha256(s.encode()).hexdigest()
    for cell in ('prmbench_qwen3_8b','pb_gsm8k_q8','pb_math_q8','pb_olympiadbench_q8','pb_omnimath_q8'):
        for lo,hi in ((64,255),(256,1023),(1024,2048)):
            pool=[r for r in release['cells'][cell]['rows'] if lo<=r['tokens']<=hi and r['group_id'] not in used]
            ordered=sorted(pool,key=lambda r:(htext(m['selection_namespace']+'/'+cell+'/'+r['group_id']),htext(r['row_id'])))
            n=0
            for r in ordered:
                if r['group_id'] in used:continue
                selected.append((cell,r['row_id'],r['group_id'],[lo,hi]));used.add(r['group_id']);n+=1
                if n==8:break
            support.append({'cell':cell,'length_bin':[lo,hi],'eligible_groups_at_bin_entry':len({r['group_id'] for r in pool}),
                            'selected':n,'quota':8,'shortfall':8-n})
    assert selected==[(r['cell'],r['row_id'],r['group_id'],r['length_bin']) for r in m['selected']]
    assert support==m['length_support'];assert len(selected)==110
    assert len({r[2] for r in selected})==110 and not {r[2] for r in selected}&set(m['excluded_components'])
    assert not {r[2] for r in selected}&{r['group_id'] for r in old['rows']}
    counts['independent_selection_rows']=len(selected)
    rows={r['uid']:r for r in e['rows']};assert len(rows)==len(e['rows'])==len(selected)
    primitives=[1,15,19,23,24,25,26,27,28];coverage=Counter();routes=Counter();failures={};geometry=[]
    max_feature=max_projection=max_inverse=0.;refits=[]
    for cell in dict.fromkeys(r['cell'] for r in m['selected']):
        info=release['cells'][cell];refit_done=False
        with np.load(info['telemetry_path'],allow_pickle=False) as original, np.load(info['label_path'],allow_pickle=False) as labels:
            raw=original['raw'];ids=original['row_ids'].astype(str);positions={v:i for i,v in enumerate(ids)}
            lpos={str(v):i for i,v in enumerate(labels['row_ids'])};assert len(lpos)==len(labels['row_ids'])
            for rec in (r for r in m['selected'] if r['cell']==cell):
                uid=rec['uid'];row=rows[uid];j=positions[rec['row_id']];assert j==rec['row']
                a,b=original['token_offsets'][j:j+2];u,v=original['step_row_offsets'][j:j+2]
                with np.load(OUT/'inputs'/(uid+'.npz'),allow_pickle=False) as z:
                    values=z['raw'].copy();ss=z['step_starts'];ee=z['step_ends']
                    np.testing.assert_array_equal(values,raw[a:b]);np.testing.assert_array_equal(ss,original['step_starts'][u:v]-a)
                    np.testing.assert_array_equal(ee,original['step_ends'][u:v]-a)
                counts['raw_input_and_span_joins']+=1
                i=lpos[rec['row_id']]
                if cell.startswith('prm'):
                    x,y=labels['step_flag_offsets'][i:i+2];target=labels['step_error_flags'][x:y]
                else:target=int(labels['first_error'][i])
                np.testing.assert_array_equal(row['target'],target);counts['direct_label_joins']+=1
                meta=load(OUT/'scores'/(uid+'.json'));assert meta['labels_decoded'] is False
                with np.load(OUT/'scores'/(uid+'.npz'),allow_pickle=False) as ar:
                    full=np.arange(0,len(values)-7,8);starts=np.unique(np.r_[full,len(values)-8]);ends=starts+8
                    fit_indices=np.searchsorted(starts,full)
                    for key,actual in [('window_starts',starts),('window_ends',ends),('fit_indices',fit_indices),('step_starts',ss),('step_ends',ee)]:
                        np.testing.assert_array_equal(ar[key],actual)
                    base=values[:,primitives];t=np.linspace(-.5,.5,8)
                    moment=np.asarray([np.column_stack((chunk.mean(0),np.sqrt(((chunk-chunk.mean(0))**2).mean(0)),
                        t@(chunk-chunk.mean(0))/(t@t))).ravel() for chunk in (base[a:b] for a,b in zip(starts,ends))])
                    em=[]
                    for span in (8,32):
                        alpha=2/(span+1);smoothed,_=lfilter([alpha],[1,-(1-alpha)],base,axis=0,zi=(1-alpha)*base[:1]);em.append(smoothed)
                    context=np.asarray([np.column_stack((base[a:b].mean(0),em[0][a:b].mean(0),em[1][a:b].mean(0))).ravel() for a,b in zip(starts,ends)])
                    for bank,features in [('moment',moment),('context',context)]:
                        np.testing.assert_allclose(ar[bank+'__features'],features,atol=1e-11,rtol=1e-12)
                        max_feature=max(max_feature,float(np.max(np.abs(ar[bank+'__features']-features))));counts['raw_feature_banks']+=1
                        bd=meta['diagnostics']['banks'][bank]
                        if bd['status']!='SCORED':continue
                        sh=bd['shared'];cols=[bd['names'].index(n) for n in sh['active_features']]
                        fit=features[fit_indices][:,cols];mean=fit.mean(0);sd=fit.std(0)
                        np.testing.assert_allclose(sh['mean'],mean,atol=1e-11,rtol=1e-12)
                        np.testing.assert_allclose(sh['sd'],sd,atol=1e-11,rtol=1e-12)
                        zz=(features[:,cols]-mean)/sd;zz-=zz[fit_indices].mean(0);zz*=np.array(sh['feature_signs'])
                        geometry.append({'uid':uid,'bank':bank,'active_p':sh['active_p'],'n_fit':len(fit),'participation_rank':sh['participation_rank']})
                        counts['normalization_reconstructions']+=1
                        for core in ('equal','iu','joint0','graph010','graph_perm'):
                            arm=bank+'__'+core;d=meta['methods'][arm]
                            if d['valid']:
                                projected=-zz@np.asarray(d['standardized_weights'])
                                np.testing.assert_allclose(ar[arm+'__window'],projected,atol=1e-9,rtol=1e-9)
                                max_projection=max(max_projection,float(np.max(np.abs(ar[arm+'__window']-projected))));counts['weight_projections']+=1
                        joint=sh.get('joint')
                        if joint:
                            jac=joint['jacobian'];valid=bool(joint['converged'] and joint['multistart']['status']=='PASS' and jac['full_global_rank']
                                and np.isfinite(jac['condition_number']) and jac['condition_number']<=1e8)
                            assert min(Counter(joint['groups']).values())>=3
                            for core in ('joint0','graph010','graph_perm'):assert meta['methods'][bank+'__'+core]['valid']==valid
                            counts['joint_validity_records']+=1
                    # Refit the first moment-Joint-valid answer per cell, then
                    # independently reconstruct its three native inverse heads.
                    if not refit_done and meta['methods']['moment__joint0']['valid']:
                        bd=meta['diagnostics']['banks']['moment'];sh=bd['shared']
                        z,anchor,_=prepare_local(ar['moment__features'],bd['names'],fit_indices);zfit=z[fit_indices]
                        joint=fit_joint_lsml(covariance_matrix(zfit),sh['joint']['groups'],anchor_index=anchor,seed=JOINT_SEED,starts=5,max_sweeps=5000)
                        C,V=joint.model_covariance,joint.global_loading;gates=np.asarray(sh['gates']['values'])
                        graph=build_graph_from_features(zfit.T,gates=gates,k=7);W=graph.toarray();perm=np.asarray(sh['gates']['permutation'])
                        expected_seed=int(hashlib.sha256((m['scoring_namespace']+'/'+cell+'/'+rec['row_id']+'/moments27_local8').encode()).hexdigest()[:8],16)
                        assert sh['gates']['seed']==expected_seed;np.testing.assert_array_equal(perm,np.random.default_rng(expected_seed).permutation(len(zfit)))
                        for core,lam,ww in [('joint0',0.,W),('graph010',.1,W),('graph_perm',.1,W[perm][:,perm])]:
                            degree=ww.sum(1);scale=np.zeros_like(degree);np.divide(1.,np.sqrt(degree),out=scale,where=degree>0)
                            L=np.eye(len(ww))-scale[:,None]*ww*scale[None,:]
                            R=zfit.T@L@zfit/len(zfit)
                            if np.trace(R)>0:R*=np.trace(C)/np.trace(R)
                            w,_=inverse.project_inverse(C+lam*R,V);w,_,_=inverse.orient(w,zfit,anchor)
                            expected=meta['methods']['moment__'+core]['standardized_weights']
                            np.testing.assert_allclose(w,expected,atol=1e-8,rtol=1e-8)
                            max_inverse=max(max_inverse,float(np.max(np.abs(w-expected))));counts['refitted_independent_inverse_heads']+=1
                        refits.append(uid);refit_done=True
                    for arm,d in meta['methods'].items():
                        for key in ('valid','decision_valid','fixed_iu_valid'):assert row[key][arm]==d[key]
                        assert row['predictions'][arm]==d.get('prediction') and row['fixed_iu_predictions'][arm]==d.get('fixed_iu_prediction')
                        if not d['valid']:
                            failures.setdefault(arm,Counter())[d.get('reason',d.get('status','invalid'))]+=1;continue
                        coverage[arm]+=1;risk=ar[arm+'__window'];sums=np.zeros(len(values));weights=np.zeros(len(values))
                        for a,b,value in zip(starts,ends,risk):sums[a:b]+=value;weights[a:b]+=1
                        token=sums/weights;steps=np.asarray([token[a:b].max() for a,b in zip(ss,ee)])
                        np.testing.assert_allclose(row['scores'][arm],steps,atol=1e-12,rtol=1e-12);counts['step_maps']+=1
                        if arm.startswith(('single__','dual__')):
                            source=d['source_arm'];np.testing.assert_array_equal(risk,ar[source+'__window'])
                            for key in ('valid','decision_valid','fixed_iu_valid','prediction','fixed_iu_prediction','peak'):assert d.get(key)==meta['methods'][source].get(key)
                            counts['fallback_inheritance']+=1
                        else:
                            models=[GaussianMixture(n_components=k,n_init=3,max_iter=300,reg_covar=1e-4,random_state=2026090705).fit(risk[fit_indices,None]) for k in (1,2)]
                            if all(mm.converged_ for mm in models):
                                bic=[mm.bic(risk[fit_indices,None]) for mm in models];opened=bic[1]<bic[0] and np.any(steps>models[1].means_.mean())
                                pred=int(np.argmax(steps)) if opened else -1
                                assert d['decision_valid'] and d['prediction']==pred;np.testing.assert_allclose(d['gate']['bic'],bic,atol=1e-9)
                            else:assert not d['decision_valid']
                            counts['independent_gmm_decisions']+=1
                        ref=meta['methods']['moment__iu'];fv=bool(d['valid'] and ref['valid'] and ref['decision_valid'])
                        assert d['fixed_iu_valid']==fv
                        if fv:assert d['fixed_iu_prediction']==(int(np.argmax(steps)) if ref['prediction']!=-1 else -1)
                    mv=meta['methods']['moment__joint0']['valid'];cv=meta['methods']['context__joint0']['valid']
                    for policy in ('single','dual'):
                        expected='moment_joint' if mv else ('context_joint' if policy=='dual' and cv else 'moment_iu')
                        assert meta['routing']['routes'][policy]==expected;routes[policy+'/'+expected]+=1
                        for core in ('joint0','graph010','graph_perm'):
                            source='moment__iu' if expected=='moment_iu' else expected.split('_')[0]+'__'+core
                            assert meta['methods'][policy+'__'+core]['source_arm']==source
                    for core in ('equal','iu'):
                        bank='context' if not mv and cv else 'moment';assert meta['methods']['dual__'+core]['source_arm']==bank+'__'+core
                    counts['independent_routing_rows']+=1
        print('Reviewed raw rows, features, gates and routing:',cell,flush=True)
    for arm,me in e['metrics'].items():
        metrics.check_equal(me,{'prm':metrics.prm(e['rows'],arm),'pb':metrics.pb(e['rows'],arm),'pb_common_iu_gate':metrics.pb(e['rows'],arm,True)})
        assert e['previous_cohort_metrics'][arm]==old['metrics'][arm];counts['metric_bundles_and_historical_replays']+=1
    for pair in c['pairs'].values():
        left,right=pair['left'],pair['right'];common=[r for r in e['rows'] if r['valid'][left] and r['valid'][right]]
        for side,arm in [('left',left),('right',right)]:
            metrics.check_equal(pair[side+'_prm'],metrics.prm(common,arm));metrics.check_equal(pair[side+'_pb'],metrics.pb(e['rows'],arm))
            metrics.check_equal(pair[side+'_pb_common_iu_gate'],metrics.pb(e['rows'],arm,True))
        counts['paired_point_bundles']+=1
    bootstrap=[]
    for left,right in [('single__joint0','moment__iu'),('single__joint0','context__equal'),('dual__graph010','dual__joint0'),
                       ('dual__iu','moment__iu'),('dual__graph010','dual__graph_perm')]:
        explicit=metrics.explicit_bootstrap(e['rows'],left,right);actual=c['pairs'][left+' minus '+right]['uncertainty']
        for key,value in explicit.items():metrics.check_equal(actual[key],value)
        bootstrap.append({'left':left,'right':right,'draws':1000,'four_intervals_and_counts':'MATCH'})
        print('Explicit bootstrap matches:',left,'minus',right,flush=True)
    result={'status':'PASS','counts':dict(counts),'coverage':dict(coverage),'routes':dict(routes),
        'failure_reasons':{k:dict(v) for k,v in failures.items()},'geometry':geometry,'refit_answer_ids':refits,
        'max_feature_difference':max_feature,'max_projection_difference':max_projection,'max_refitted_inverse_difference':max_inverse,
        'bootstrap_checks':bootstrap,'seconds':time.monotonic()-started,
        'scope':'Independent raw/score/gate/metric reconstruction; original Joint and graph-builder kernels reused for five representative refits.',
        'hashes':{str(OUT/n):sha(OUT/n) for n in ('MANIFEST.json','SCORES_FROZEN.json','EVALUATION.json','CONTRASTS.json')},
        'review_dependencies':{str(ROOT/'scripts'/n):sha(ROOT/'scripts'/n) for n in ('review_fusion_explicit_fallback_v1.py','review_fusion_context_bank_v1.py')},
        'review_script_sha256':sha(__file__)}
    (OUT/'REVIEW.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','counts','coverage','routes','seconds')},indent=2),flush=True)


if __name__=='__main__':review()
