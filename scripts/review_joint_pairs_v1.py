"""Independent algebra, grouping summaries and native-map review; no labels."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='1'
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
from sklearn.metrics import adjusted_rand_score

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/joint_pair_identifiability_audit_v1';PARENT=ROOT/'results/fusion_replication_v1'
sys.path.insert(0,str(ROOT/'local_cache/short_cycle01_code'))
import spectral_utils
spectral_utils.__path__.append(str(ROOT/'spectral_utils'))
from spectral_utils.joint_pair_extension import pair_model_covariance, fit_joint_pairs
from spectral_utils.joint_pair_jacobian import profiled_pair_jacobian
from spectral_utils.joint_lsml import _profiled_jacobian_audit


def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')


def native(c,v):
    val,vec=np.linalg.eigh((c+c.T)/2);psd=(vec*np.maximum(val,0))@vec.T
    lo,hi=np.linalg.eigvalsh(psd)[[0,-1]];ridge=1. if hi<=1e-14 else max(0.,(hi-1000*lo)/999,hi*1e-10)
    return np.linalg.solve(psd+ridge*np.eye(len(v)),v)


def independent_product_profile(v,u,labels):
    """Eliminate pair rows outright, then profile larger-group nuisances."""
    p=len(v);sizes=Counter(labels);jv=[];ju=[]
    for i in range(p):
        for j in range(i+1,p):
            if labels[i]==labels[j] and sizes[labels[i]]==2:continue
            a=np.zeros(p);b=np.zeros(p);a[i]=v[j];a[j]=v[i]
            if labels[i]==labels[j]:b[i]=u[j];b[j]=u[i]
            jv.append(a);ju.append(b)
    jv,ju=np.asarray(jv),np.asarray(ju);active=np.linalg.norm(ju,axis=0)>1e-12;ju=ju[:,active]
    projected=jv-ju@np.linalg.lstsq(ju,jv,rcond=None)[0] if ju.shape[1] else jv
    norms=np.linalg.norm(projected,axis=0);active=norms>1e-12;z=projected[:,active]/norms[active]
    singular=np.linalg.svd(z,compute_uv=False);rank=int(np.linalg.matrix_rank(z))
    return bool(active.sum()==p and rank==p),rank,float(singular[0]/max(singular[-1],1e-12))


def algebra():
    v=np.linspace(.3,.55,6);u=np.linspace(.2,.45,6);u[1]*=-1;groups=np.repeat(np.arange(3),2)
    same=groups[:,None]==groups[None,:];mask=same.astype(float)-np.eye(6)
    s=np.outer(v,v)+same*np.outer(u,u);np.fill_diagonal(s,1.)
    family=[];reference=native(s,v)
    for t in (.03,.2,1.,4.,10.,30.):
        changed=u.copy();changed[0]*=t;changed[1]/=t
        component=np.outer(v,v)+same*np.outer(changed,changed)
        legacy=component+np.diag(np.maximum(np.diag(s)-np.diag(component),0))
        fixed,_,_=pair_model_covariance(s,groups,v,changed)
        off=~np.eye(6,dtype=bool)
        np.testing.assert_allclose(legacy[off],s[off],atol=1e-15)
        np.testing.assert_allclose(fixed,s,atol=1e-14)
        np.testing.assert_allclose(native(fixed,v),reference,atol=1e-14)
        jac=_profiled_jacobian_audit(v,changed,mask);assert jac['full_global_rank']
        family.append({'pair_scale':t,'pair_loadings':changed[:2].tolist(),'legacy_diagonal':np.diag(legacy).tolist(),
            'legacy_weights':native(legacy,v).tolist(),'pair_covariance_weights':native(fixed,v).tolist(),
            'offdiagonal_max_difference':float(np.max(np.abs(legacy[off]-s[off]))),'global_jacobian_pass':True})
    # Independent central differences expose the nuisance null directions.
    indices=np.triu_indices(6,1);theta=np.r_[v,u]
    def prediction(t):return (np.outer(t[:6],t[:6])+same*np.outer(t[6:],t[6:]))[indices]
    jac=np.column_stack([(prediction(theta+1e-6*d)-prediction(theta-1e-6*d))/2e-6 for d in np.eye(12)])
    rank=int(np.linalg.matrix_rank(jac,tol=1e-7));assert rank==9
    result={'status':'PASS','fixture_covariance':s.tolist(),'global_loadings':v.tolist(),'group_loadings':u.tolist(),
        'groups':groups.tolist(),'full_offdiag_jacobian_rank':rank,'latent_parameter_count':12,
        'pair_scale_null_directions':3,'family':family,
        'finding':'Same off-diagonal objective and passing global Jacobian do not guarantee an invariant clipped native covariance.'}
    save(OUT/'ALGEBRA.json',result);return result


def review():
    started=time.monotonic();m,f,summary=[load(OUT/n) for n in ('MANIFEST.json','FROZEN.json','SUMMARY.json')]
    assert not m['labels_decoded'] and not f['labels_decoded'] and not summary['labels_decoded']
    assert f['manifest_sha256']==sha(OUT/'MANIFEST.json') and summary['frozen_sha256']==sha(OUT/'FROZEN.json')
    for p,h in {**m['hashes'],**f['files']}.items():assert sha(p)==h,p
    amendment=load(OUT/'JACOBIAN_AMENDMENT.json')
    for path,digest in amendment['hashes'].items():assert sha(path)==digest,path
    counts=Counter();bankrows={b:[] for b in ('moment','context')};refits=[];seen=set();maxdiff=0.;guard_changes=[];zero_products=0
    spec=importlib.util.spec_from_file_location('old_orientation_review',ROOT/'scripts/review_fusion_context_bank_v1.py')
    oldreview=importlib.util.module_from_spec(spec);spec.loader.exec_module(oldreview)
    for rec in m['selected']:
        r=load(OUT/'rows'/(rec['uid']+'.json'));p=load(PARENT/'scores'/(rec['uid']+'.json'))
        assert not r['labels_decoded'] and r['manifest_sha256']==sha(OUT/'MANIFEST.json')
        with np.load(OUT/'rows'/(rec['uid']+'.npz'),allow_pickle=False) as arrays,np.load(PARENT/'scores'/(rec['uid']+'.npz'),allow_pickle=False) as source:
            fi=source['fit_indices']
            for bank,b in r['banks'].items():
                bankrows[bank].append(b);before=p['diagnostics']['banks'][bank]
                assert b['old_valid']==p['methods'][bank+'__joint0']['valid']
                normal=b['normalization'];cols=[before['names'].index(s) for s in normal['active_features']]
                x=source[bank+'__features'][fi][:,cols];np.testing.assert_allclose(x.mean(0),normal['mean'],atol=1e-11)
                np.testing.assert_allclose(x.std(0),normal['sd'],atol=1e-11)
                z=(x-x.mean(0))/x.std(0);z-=z.mean(0);z*=np.array(normal['feature_signs'])
                s=z.T@z/(len(z)-1);counts['feature_normalization_covariances']+=1
                g=b['grouping'];admissible=[]
                for candidate in g['candidates']:
                    if 'labels' not in candidate:continue
                    labels=np.asarray(candidate['labels']);sizes=list(Counter(labels).values())
                    assert sorted(sizes)==sorted(candidate['group_sizes'])
                    held=candidate['held_answer_labels'];hs=[list(Counter(a).values()) for a in held]
                    okay=float(np.mean([len(v)==candidate['K'] and min(v)>=2 for v in hs]))
                    assert okay==candidate['held_admissible_fraction']
                    expected=len(sizes)==candidate['K'] and min(sizes)>=2 and okay>=.95
                    assert candidate['admissible']==expected
                    ari=[adjusted_rand_score(labels,a) for a in held]
                    for key,value in [('median_ari',np.median(ari)),('mean_ari',np.mean(ari)),('minimum_ari',np.min(ari))]:
                        np.testing.assert_allclose(candidate[key],value,atol=1e-12)
                    if expected:admissible.append(candidate)
                    counts['group_candidate_guards_and_ari']+=1
                if not admissible:
                    assert g['status']!='SELECTED' and not b['valid'];continue
                winner=min(admissible,key=lambda x:(-x['median_ari'],-x['mean_ari'],-x['minimum_ari'],x['K']))
                assert g['K']==winner['K'];np.testing.assert_array_equal(g['labels'],winner['labels']);counts['grouping_selections']+=1
                if b['status'].startswith('PAIR_'):
                    d=b['failure_detail'];bi,bj=d['residual_variance_budgets'];product=abs(d['original_product'])
                    if 'NEGATIVE' in b['status']:assert min(bi,bj)<-d['tolerance']
                    else:assert product>np.sqrt(max(0,bi)*max(0,bj))+d['tolerance']
                    counts['independent_infeasibility_checks']+=1;continue
                vv,uu=arrays[bank+'__v'],arrays[bank+'__u'];labels=np.array(g['labels']);same=labels[:,None]==labels[None,:]
                component=np.outer(vv,vv)+same*np.outer(uu,uu);d=np.diag(s)-np.diag(component)
                c=component+np.diag(np.maximum(d,0));np.testing.assert_allclose(c,arrays[bank+'__covariance'],atol=1e-11)
                off=~np.eye(len(vv),dtype=bool);np.testing.assert_allclose(c[off],arrays[bank+'__fitted_offdiag'][off],atol=2e-10)
                assert np.linalg.eigvalsh(c)[0]>=-1e-9
                for pair in b.get('pairs',{}).get('pairs',[]):
                    i,j=pair['indices'];budgets=np.diag(s)[[i,j]]-vv[[i,j]]**2
                    np.testing.assert_allclose(c[[i,j],[i,j]],np.diag(s)[[i,j]],atol=2e-10)
                    np.testing.assert_allclose(uu[i]*uu[j],pair['original_product'],atol=2e-10)
                    if np.min(budgets)>1e-10:np.testing.assert_allclose(uu[i]**2/budgets[0],uu[j]**2/budgets[1],atol=1e-10)
                    counts['pair_product_and_variance_reconstructions']+=1
                weight=native(c,vv);difference=float(np.max(np.abs(weight-arrays[bank+'__weight'])))
                maxdiff=max(maxdiff,difference);np.testing.assert_allclose(weight,arrays[bank+'__weight'],atol=1e-9,rtol=1e-9)
                counts['independent_native_inverse_maps']+=1
                jac=b['jacobian'];valid=bool(b['converged'] and b['multistart']['status']=='PASS' and jac['full_global_rank'] and
                    np.isfinite(jac['condition_number']) and jac['condition_number']<=1e8);assert b['valid']==valid
                corrected=profiled_pair_jacobian(vv,uu,labels)
                if min(Counter(labels).values())==2:
                    independent=independent_product_profile(vv,uu,labels)
                    assert independent[0]==corrected['full_global_rank'] and independent[1]==corrected['rank']
                    np.testing.assert_allclose(independent[2],corrected['condition_number'],atol=1e-7,rtol=1e-7)
                    counts['independent_product_profile_checks']+=1;zero_products+=corrected['zero_pair_products']
                amended_valid=bool(b['converged'] and b['multistart']['status']=='PASS' and corrected['full_global_rank'] and
                    np.isfinite(corrected['condition_number']) and corrected['condition_number']<=1e8)
                if amended_valid!=valid:guard_changes.append({'uid':rec['uid'],'bank':bank,'old_valid':valid,'amended_valid':amended_valid})
                counts['amended_jacobian_eligibility_checks']+=1
                if b['same_partition_as_parent'] and b['old_valid']:
                    expected=p['methods'][bank+'__joint0']['standardized_weights'];anchor=normal['active_features'].index(normal['anchor_feature'])
                    oriented,_,_=oldreview.orient(weight,z,anchor);np.testing.assert_allclose(oriented,expected,atol=1e-9,rtol=1e-9)
                    counts['unchanged_partition_parent_map_replays']+=1
                key=(rec['cell'],bank)
                if b['valid'] and b['pairs'].get('pair_count',0)>0 and key not in seen:
                    anchor=normal['active_features'].index(normal['anchor_feature']);refit=fit_joint_pairs(s,labels,anchor_index=anchor)
                    np.testing.assert_allclose(refit.joint.model_covariance,c,atol=1e-8,rtol=1e-8)
                    assert refit.joint.multistart_audit['status']=='PASS'
                    refits.append({'uid':rec['uid'],'bank':bank});seen.add(key)
        counts['source_answer_rows']+=1
    recomputed={}
    for bank,rr in bankrows.items():
        recomputed[bank]={'status_counts':dict(Counter(r['status'] for r in rr)),
            'old_valid':sum(r['old_valid'] for r in rr),'new_valid':sum(r['valid'] for r in rr),
            'rescued':sum(r['valid'] and not r['old_valid'] for r in rr),'lost':sum(r['old_valid'] and not r['valid'] for r in rr),
            'selected_K':{str(k):v for k,v in Counter(r['grouping']['K'] for r in rr if r['grouping']['status']=='SELECTED').items()},
            'valid_K':{str(k):v for k,v in Counter(r['grouping']['K'] for r in rr if r['valid']).items()},
            'same_partition_as_parent':sum(r.get('same_partition_as_parent',False) for r in rr),
            'valid_pair_fits':sum(r['valid'] and r.get('pairs',{}).get('pair_count',0)>0 for r in rr)}
    assert recomputed==summary['banks']
    algebra()
    result={'status':'PASS','counts':dict(counts),'representative_refits':refits,'max_native_weight_difference':maxdiff,
        'labels_decoded':False,'seconds':time.monotonic()-started,
        'jacobian_amendment':{'guard_changes':guard_changes,'exact_zero_pair_products':zero_products,
            'canonical_entry_point':'spectral_utils.joint_pair_jacobian.fit_joint_pairs_checked'},
        'scope':'Independent pair algebra, covariance/inverse and grouping guard/ARI reconstruction; original clustering labels and optimizer reused.',
        'hashes':{str(OUT/n):sha(OUT/n) for n in ('MANIFEST.json','FROZEN.json','SUMMARY.json','ALGEBRA.json','JACOBIAN_AMENDMENT.json')},
        'dependencies':{str(ROOT/'scripts/review_fusion_context_bank_v1.py'):sha(ROOT/'scripts/review_fusion_context_bank_v1.py')},
        'review_script_sha256':sha(__file__)}
    save(OUT/'REVIEW.json',result);print(json.dumps({k:result[k] for k in ('status','counts','representative_refits','seconds')},indent=2),flush=True)


if __name__=='__main__':review()
