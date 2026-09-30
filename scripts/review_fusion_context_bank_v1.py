"""Independent raw-feature, group, inverse, gate and benchmark review."""
import os
for option in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[option]='1'
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.signal import lfilter
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score,roc_auc_score
from sklearn.mixture import GaussianMixture

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/fusion_context_bank_pilot_v1'
PARENT=ROOT/'results/answer_localization_representation_pilot_v1'
sys.path.insert(0,str(ROOT/'local_cache/short_cycle01_code'))
import spectral_utils
spectral_utils.__path__.append(str(ROOT/'spectral_utils'))
# Reuse the unchanged, source-bound nearest-neighbor graph and IU kernel;
# feature reconstruction, Laplacian, native inverse and metrics are independent.
from spectral_utils.laplacian_upcr import build_graph_from_features,IU_FIT_DEFAULTS
from spectral_utils.upcr import upcr_fit


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')


def project_inverse(c,v):
    c=(c+c.T)/2;e,u=np.linalg.eigh(c);e=np.maximum(e,0);psd=(u*e)@u.T
    lo,hi=np.linalg.eigvalsh(psd)[[0,-1]]
    ridge=1. if hi<=1e-14 else max(0.,(hi-1000*lo)/999,hi*1e-10)
    return np.linalg.solve(psd+ridge*np.eye(len(v)),v),ridge


def orient(w,z,anchor):
    score=z@w;sd=score.std();row=z.mean(axis=1)
    corr=np.corrcoef(score,row)[0,1] if row.std() else np.nan
    rule='rowmean_pearson'
    if not np.isfinite(corr) or abs(corr)<.02:
        corr=spearmanr(score,z[:,anchor]).statistic;rule='entropy_spearman_fallback'
    return w/sd*(-1 if corr<0 else 1),rule,bool(corr<0)


def pb(rows,arm,fixed=False):
    values=[]
    for cell in sorted({r['cell'] for r in rows if r['cell'].startswith('pb_')}):
        correct={True:[],False:[]}
        for r in (r for r in rows if r['cell']==cell):
            valid=r['fixed_iu_valid'][arm] if fixed else r['decision_valid'][arm]
            prediction=r['fixed_iu_predictions'][arm] if fixed else r['predictions'][arm]
            correct[r['target']==-1].append(int(valid and prediction==r['target']))
        if not correct[True] or not correct[False]:return None
        ca,ea=np.mean(correct[True]),np.mean(correct[False]);values.append(2*ca*ea/(ca+ea) if ca+ea else 0.)
    return float(np.mean(values))


def auc(records,arm):
    selected=[r for r in records if r['cell'].startswith('prm') and r['valid'][arm]]
    if not selected:return None,None
    y=np.concatenate([r['target'] for r in selected]);x=np.concatenate([r['scores'][arm] for r in selected])
    within=[roc_auc_score(r['target'],r['scores'][arm]) for r in selected if len(set(r['target']))==2]
    return float(roc_auc_score(y,x)) if len(set(y))==2 else None,float(np.mean(within)) if within else None


def core_review():
    started=time.monotonic();manifest,frozen,e=[load(OUT/n) for n in ('MANIFEST.json','SCORES_FROZEN.json','EVALUATION.json')]
    for p,h in {**manifest['hashes'],**frozen['files']}.items():assert sha(p)==h,p
    assert frozen['manifest_sha256']==sha(OUT/'MANIFEST.json') and e['scores_sha256']==sha(OUT/'SCORES_FROZEN.json')
    rows=e['rows'];by_id={r['uid']:r for r in rows};assert len(by_id)==58
    release=load(PARENT/'RELEASE.json');counts=Counter();coverage=Counter();failure={};group_counts={}
    max_feature=max_weight=max_span=0.;geometry={};peak_hits=Counter();gate_agreement=Counter()
    for cell in sorted({r['cell'] for r in rows}):
        info=release['cells'][cell];assert sha(info['label_path'])==info['label_opaque_sha256']
        with np.load(info['label_path'],allow_pickle=False) as lab:
            index={str(k):i for i,k in enumerate(lab['row_ids'])};assert len(index)==len(lab['row_ids'])
            for r in (r for r in rows if r['cell']==cell):
                i=index[r['row_id']]
                if cell.startswith('prm'):
                    a,b=lab['step_flag_offsets'][i:i+2];target=lab['step_error_flags'][a:b]
                else:target=int(lab['first_error'][i])
                np.testing.assert_array_equal(target,r['target']);counts['direct_label_joins']+=1
    for rec in manifest['selected']:
        uid=rec['uid'];row=by_id[uid];meta=load(OUT/'scores'/f'{uid}.json');diag=meta['diagnostics']
        old=load(PARENT/'scores'/f'{uid}.json')['report'];old_iu=old['methods']['moments27_local8__iu']
        common_open=old_iu['readout']['prediction']!=-1;assert common_open==diag['common_iu_gate_open']
        seed=int(hashlib.sha256((manifest['release_id']+'/'+rec['cell']+'/'+rec['row_id']+'/moments27_local8').encode()).hexdigest()[:8],16)
        perm=np.random.default_rng(seed).permutation(rec['tokens']//8)
        np.testing.assert_array_equal(perm,diag['graph_permutation']);counts['permutation_identity_checks']+=1
        with np.load(PARENT/'inputs'/f'{uid}.npz',allow_pickle=False) as raw:token=raw['raw'].copy()
        base=token[:,[1,15,19,23,24,25,26,27,28]];smooth=[]
        for span in (8,32):
            alpha=2./(span+1);beta=1-alpha
            smooth.append(lfilter([alpha],[1.,-beta],base,axis=0,zi=(beta*base[0])[None,:])[0])
        with np.load(OUT/'scores'/f'{uid}.npz',allow_pickle=False) as arrays,np.load(PARENT/'scores'/f'{uid}.npz',allow_pickle=False) as parent:
            starts,ends,fit=arrays['window_starts'],arrays['window_ends'],arrays['fit_indices']
            features=np.asarray([np.column_stack((base[a:b].mean(axis=0),smooth[0][a:b].mean(axis=0),smooth[1][a:b].mean(axis=0))).ravel() for a,b in zip(starts,ends)])
            difference=np.max(np.abs(features-arrays['context__features']));max_feature=max(max_feature,float(difference))
            np.testing.assert_allclose(features,arrays['context__features'],atol=1e-11,rtol=1e-11);counts['raw_context_reconstructions']+=1
            np.testing.assert_array_equal(arrays['context__features'][:,::3],parent['moments27_local8__features'][:,::3]);counts['exact_level_column_replays']+=1
            normalized={};graphs={}
            for bank,binfo in diag['banks'].items():
                if binfo['status']!='OK':continue
                normal=binfo['normalization'];columns=[binfo['names'].index(s) for s in normal['active_features']]
                values=arrays[bank+'__features'][:,columns];z=(values-normal['mean'])/normal['sd'];z-=z[fit].mean(axis=0);z*=normal['feature_signs']
                np.testing.assert_allclose(z,arrays[bank+'__normalized'],atol=1e-12,rtol=1e-12)
                np.testing.assert_allclose(z[fit].std(axis=0),1.,atol=1e-9);counts['normalization_reconstructions']+=1
                normalized[bank]=z;c=z[fit].T@z[fit]
                participation=float(np.trace(c)**2/np.sum(c*c));assert abs(participation-normal['participation_rank'])<1e-9
                geometry.setdefault(bank,[]).append({'uid':uid,'active_p':normal['active_p'],'rank':normal['rank'],
                    'participation_rank':participation,'n_fit':len(fit)})
                if bank+'__gates' in arrays.files:
                    graphs[bank]=build_graph_from_features(z[fit].T,gates=arrays[bank+'__gates'],k=7).toarray()
            for family,g in diag['groupings'].items():
                ks=tuple(g['candidate_k']);bank='context' if family.startswith('context') else 'moment'
                p=normalized[bank].shape[1]
                assert ks==((3,4,6,8) if family=='context' else tuple(range(3,p//3+1)))
                eligible=[]
                for candidate in g['candidates']:
                    labels=np.asarray(candidate['labels']);sizes=[int(np.sum(labels==k)) for k in np.unique(labels)]
                    assert sizes==candidate['group_sizes'] and len(labels)==p
                    ari=[adjusted_rand_score(labels,k) for k in candidate['held_answer_labels']]
                    np.testing.assert_allclose(ari,candidate['ari_to_consensus'],atol=1e-12)
                    held_sizes=[[int(np.sum(np.asarray(k)==j)) for j in np.unique(k)] for k in candidate['held_answer_labels']]
                    admissible=(len(sizes)==candidate['K'] and min(sizes)>=3 and
                        np.mean([len(s)==candidate['K'] and min(s)>=3 for s in held_sizes])>=.95)
                    assert bool(admissible)==candidate['admissible']
                    for key,value in [('median_ari',np.median(ari)),('mean_ari',np.mean(ari)),('minimum_ari',np.min(ari))]:assert abs(candidate[key]-value)<1e-12
                    if admissible:eligible.append(candidate)
                    counts['group_candidate_contracts']+=1
                if eligible:
                    chosen=sorted(eligible,key=lambda d:(-d['median_ari'],-d['mean_ari'],-d['minimum_ari'],d['K']))[0]
                    assert chosen['K']==g['K'];np.testing.assert_array_equal(chosen['labels'],g['labels'])
                else:assert g['status']=='BLOCKED_NO_ADMISSIBLE_PARTITION'
                group_counts.setdefault(family,Counter())[str(g.get('K',g['status']))]+=1
            for family,j in diag['joint_fits'].items():
                if j['valid']:
                    assert j['converged'] and j['multistart']['status']=='PASS'
                    assert j['jacobian']['full_global_rank'] and j['jacobian']['condition_number']<=1e8
                    counts['joint_validity_checks']+=1
            for arm,d in meta['methods'].items():
                if not d['valid']:
                    failure.setdefault(arm,Counter())[d['reason']]+=1
                    assert not row['valid'][arm] and not row['fixed_iu_valid'][arm];continue
                coverage[arm]+=1;risk=arrays[arm+'__window'];steps=arrays[arm+'__risk'];assert abs(risk[fit].mean())<1e-9 and abs(risk[fit].std()-1)<1e-9
                if d.get('parent_replay'):
                    np.testing.assert_array_equal(risk,parent[d['source']+'__window']);np.testing.assert_array_equal(steps,parent[d['source']+'__step'])
                    counts['exact_parent_replays']+=1
                else:
                    family,core=arm.split('__');bank='context' if family.startswith('context') else 'moment';z=normalized[bank];zz=z[fit]
                    normal=diag['banks'][bank]['normalization'];anchor=normal['active_features'].index(normal['anchor_feature'])
                    if core=='equal':w=np.ones(z.shape[1])/z.shape[1]
                    elif core=='iu':
                        iu=upcr_fit(zz.T,**dict(IU_FIT_DEFAULTS));assert not iu.abstained;w=iu.w
                    else:
                        c=arrays[family+'__model_covariance'];v=arrays[family+'__global_loading'];system=c.copy()
                        if core!='joint0':
                            g=graphs[bank];g=g[np.ix_(perm,perm)] if core=='graph_perm' else g
                            deg=g.sum(axis=1);inv=np.where(deg>1e-12,1/np.sqrt(np.maximum(deg,1e-12)),0.)
                            lap=np.eye(len(g))-inv[:,None]*g*inv[None,:]
                            rough=zz.T@lap@zz/len(zz);rough=(rough+rough.T)/2
                            scale=np.trace(c)/np.trace(rough) if np.trace(rough)>1e-12 else 0.;system=c+.1*scale*rough
                        w,ridge=project_inverse(system,v);assert abs(ridge-d['inverse']['ridge'])<1e-8
                    w,rule,flipped=orient(w,zz,anchor);assert rule==d['orientation_rule'] and flipped==d['flipped']
                    error=float(np.max(np.abs(w-arrays[arm+'__weights'])));max_weight=max(max_weight,error)
                    np.testing.assert_allclose(w,arrays[arm+'__weights'],atol=1e-8,rtol=1e-8)
                    np.testing.assert_allclose(-z@w,risk,atol=1e-8,rtol=1e-8);counts['independent_weight_reconstructions']+=1
                    if bank=='moment' and diag['joint_fits'][family]['same_parent_partition']:
                        np.testing.assert_allclose(risk,arrays['moment__'+core+'__window'],atol=1e-10,rtol=1e-10);counts['same_partition_parent_replays']+=1
                totals=np.zeros(rec['tokens']);cover=np.zeros(rec['tokens'])
                for a,b,v in zip(starts,ends,risk):totals[a:b]+=v;cover[a:b]+=1
                assert np.all(cover>0);mapped=np.asarray([(totals/cover)[a:b].max() for a,b in zip(arrays['step_starts'],arrays['step_ends'])])
                max_span=max(max_span,float(np.max(np.abs(mapped-steps))));np.testing.assert_allclose(mapped,steps,atol=1e-12,rtol=1e-12)
                counts['span_maps']+=1
                models=[GaussianMixture(n_components=k,n_init=3,max_iter=300,reg_covar=1e-4,random_state=2026090705).fit(risk[fit,None]) for k in (1,2)]
                if d['decision_valid']:
                    assert all(m.converged_ for m in models);bic=[m.bic(risk[fit,None]) for m in models]
                    split=bic[1]<bic[0];threshold=float(models[1].means_.mean()) if split else None
                    opened=bool(split and np.any(steps>threshold));peak=int(np.argmax(steps))
                    assert d['prediction']==(peak if opened else -1);np.testing.assert_allclose(bic,d['gate']['bic'],atol=1e-9)
                    counts['independent_gmm_gates']+=1
                assert d['fixed_iu_prediction']==(int(np.argmax(steps)) if common_open else -1)
                np.testing.assert_array_equal(steps,row['scores'][arm])
                if rec['cell'].startswith('pb_') and row['target']!=-1:peak_hits[arm]+=int(np.argmax(steps)==row['target'])
                if rec['cell'].startswith('pb_') and d['decision_valid']:gate_agreement[arm]+=int((d['prediction']!=-1)==common_open)
    for arm,metric in e['metrics'].items():
        aa,within=auc(rows,arm)
        assert abs(aa-metric['prm']['auroc'])<1e-12 and abs(within-metric['prm']['within_answer_auc'])<1e-12
        assert abs(pb(rows,arm)-metric['pb']['macro_f1'])<1e-12
        assert abs(pb(rows,arm,True)-metric['pb_common_iu_gate']['macro_f1'])<1e-12;counts['arm_endpoint_checks']+=1
    transitions={}
    for task in ('prm','pb_','all'):
        subset=[r for r in rows if task=='all' or r['cell'].startswith(task)]
        transitions[task]={key:sum(bool(r['valid']['moment__joint0'])==a and bool(r['valid']['context__joint0'])==b for r in subset)
            for key,a,b in [('both_valid',True,True),('rescued',False,True),('lost',True,False),('both_invalid',False,False)]}
    simple={}
    for bank in ('moment','context'):
        common=[r for r in rows if r['cell'].startswith('prm') and r['valid'][bank+'__joint0']]
        simple[bank]={'common_prm_answers':len(common),'auc':{bank+'__'+core:auc(common,bank+'__'+core)[0]
            for core in ('equal','iu','joint0','graph010','graph_perm')}}
    result={'status':'NUMERICAL_PASS_BOOTSTRAP_PENDING','counts':dict(counts),'coverage':dict(coverage),
        'failures':{k:dict(v) for k,v in failure.items()},'group_counts':{k:dict(v) for k,v in group_counts.items()},
        'geometry':geometry,'pb_error_peak_hits':dict(peak_hits),'pb_native_common_gate_agreements':dict(gate_agreement),
        'coverage_transitions':transitions,'descriptive_matched_simple_controls':simple,
        'max_raw_feature_error':max_feature,'max_weight_error':max_weight,'max_span_error':max_span,
        'evaluation_sha256':sha(OUT/'EVALUATION.json'),'review_script_sha256':sha(__file__),'seconds':time.monotonic()-started}
    save(OUT/'REVIEW.json',result);print(json.dumps({k:result[k] for k in ('status','counts','coverage_transitions','descriptive_matched_simple_controls','seconds')},indent=2),flush=True)


def bootstrap_review():
    result=load(OUT/'REVIEW.json');assert result['review_script_sha256']==sha(__file__)
    e=load(OUT/'EVALUATION.json');assert result['evaluation_sha256']==sha(OUT/'EVALUATION.json')
    contrasts=load(OUT/'CONTRASTS.json');assert contrasts['state']=='COMPLETE' and len(contrasts['pairs'])==31
    spec=importlib.util.spec_from_file_location('explicit_parent',ROOT/'scripts/run_answer_localization_v2.py')
    parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
    cases=[('context__iu','moment__iu'),('context__joint0','moment__joint0'),('context__graph010','context__joint0')]
    checked=[]
    for left,right in cases:
        started=time.monotonic();old=parent.paired_intervals(e['rows'],left,right,draws=1000)
        exact=contrasts['pairs'][left+' minus '+right]['uncertainty']
        for key in ('prm_common_valid_ci95','pb_all_population_ci95'):np.testing.assert_allclose(old[key],exact[key],atol=1e-12,rtol=1e-12)
        assert old['prm_valid_draws']==exact['prm_common_valid_valid_draws'] and old['pb_valid_draws']==exact['pb_all_population_valid_draws']
        checked.append({'left':left,'right':right,'draws':1000,'status':'MATCH','explicit_seconds':time.monotonic()-started})
        print('Explicit 1000-draw bootstrap matches:',left,'minus',right,flush=True)
    result.update(status='PASS',bootstrap_reference_checks=checked,contrasts_sha256=sha(OUT/'CONTRASTS.json'))
    save(OUT/'REVIEW.json',result);print('Full review PASS.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('core','bootstrap'),required=True)
    args=parser.parse_args();{'core':core_review,'bootstrap':bootstrap_review}[args.phase]()
