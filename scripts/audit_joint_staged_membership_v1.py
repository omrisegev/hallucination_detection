"""Independent score, membership, calibration and covariance audit."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,runpy
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_joint_staged_membership_v1 import OUT,inputs,KINDS,training_inputs,bank,references,PAIRS
from scripts.audit_joint_selection_results import metrics
from scripts.run_digitfree_broad50_v1 import sha
from scripts.run_lsml_gate_locator_research_v1 import dump
from spectral_utils.joint_structured_stress import structured_augmentation
from spectral_utils.joint_sparse_membership import alias_coordinates,initial_loadings
from spectral_utils.digitfree_broad50 import ANCHOR


def factor_information(c,a):
    regularized=(1-1e-4)*c+1e-4*np.diag(np.maximum(np.diag(c),1e-12))
    return np.sum(a*np.linalg.solve(regularized,a),axis=0)


def audit_model(model,x,offsets,sources):
    for ids in model['aliases']:
        for i in ids:np.testing.assert_array_equal(x[:,i],x[:,ids[0]])
    z=alias_coordinates(x,model['aliases']);membership=model.get('membership');rounds=0
    if membership:
        active=np.arange(z.shape[1]);row_sources=np.repeat(sources,np.diff(offsets))
        _,source_index=np.unique(row_sources,return_inverse=True)
        for r in membership['rounds']:
            np.testing.assert_array_equal(active,r['active_before']);cov=np.cov(np.ascontiguousarray(z[:,active]).T)
            cal=r['calibration'];np.testing.assert_allclose(cal['penalty'],max(np.quantile(cal['null_maxima'],.95),1e-8))
            rng=np.random.default_rng(cal['seed']);signs=rng.choice(np.array([-1.,1.]),size=(source_index.max()+1,len(active)))
            null=np.cov(z[:,active]*signs[source_index],rowvar=False);np.fill_diagonal(null,0.)
            v,u,mask=initial_loadings(cov,np.asarray(r['discovery']['labels']),start=0,seed=cal['seed'])
            maximum=max(np.max(np.abs(null@v)),np.max(np.abs((null*mask)@u)))
            np.testing.assert_allclose(maximum,cal['null_maxima'][0],atol=1e-11,rtol=1e-10)
            converged=[]
            for s in r['sparse']['starts']:
                assert np.all(np.diff(s['objective_trace'])<=1e-9)
                v=np.asarray(s['v']);u=np.asarray(s['u']);assert np.all(v*v+u*u<=np.diag(cov)+1e-12)
                residual=cov-np.outer(v,v)-mask*np.outer(u,u)
                loss=.5*np.square(residual[np.triu_indices(len(v),1)]).sum()+cal['penalty']*(np.abs(v).sum()+np.abs(u).sum())
                np.testing.assert_allclose(loss,s['objective'],atol=1e-10)
                np.testing.assert_array_equal(np.flatnonzero((v!=0)|(u!=0)),s['support'])
                if s['converged']:converged.append(s['support'])
            union=np.unique(np.concatenate(converged))
            labs=np.asarray(r['discovery']['labels']);detail=r['sparse']['signal_membership']
            if len(union)<len(labs):
                keep=union
                assert detail['stage']=='zero_rows'
                np.testing.assert_array_equal(np.setdiff1d(np.arange(len(labs)),union),detail['removed_zero_rows'])
            else:
                signal_groups=[]
                for group in np.unique(labs):
                    if any(any(np.asarray(s['v'])[labs==group]!=0) for s in r['sparse']['starts'] if s['converged']):signal_groups.append(group)
                keep=np.array([i for i in union if labs[i] in signal_groups],dtype=int)
                assert detail['stage']=='global_groups'
                np.testing.assert_array_equal(signal_groups,detail['signal_groups'])
                np.testing.assert_array_equal(np.setdiff1d(np.unique(labs),signal_groups),detail['nuisance_groups'])
            np.testing.assert_array_equal(keep,detail['retained'])
            active=active[keep];np.testing.assert_array_equal(active,r['active_after']);rounds+=1
        np.testing.assert_array_equal(active,membership['active'])
    if not model['valid']:return 0.,rounds
    fit=model['refinement'];assert fit['converged_starts']>=4 and fit['multistart']['status']=='PASS'
    assert fit['jacobian']['full_global_rank'] and fit['jacobian']['condition_number']<=1e8
    selection=fit['selection'];chosen=selection['automatic'];index=selection['automatic_index']
    assert chosen==selection['path'][index] and chosen['minimum_retention']>=.95
    if index+1<len(selection['path']):assert selection['path'][index+1]['minimum_retention']<.95
    initial=np.asarray(model['initial_active']);active=np.asarray(model['active'])
    np.testing.assert_array_equal(initial,membership['active']);np.testing.assert_array_equal(active,initial[np.asarray(chosen['active'])])
    c0=np.asarray(membership['observed_covariance']);v0=np.asarray(membership['global_loading']);u0=np.asarray(membership['group_loading'])
    labels0=np.asarray(membership['final_labels']);a0=np.zeros((len(v0),1+len(np.unique(labels0))));a0[:,0]=v0
    for j,g in enumerate(np.unique(labels0)):a0[labels0==g,j+1]=u0[labels0==g]
    info=factor_information(c0,a0);np.testing.assert_allclose(info,selection['initial_factor_information'],atol=1e-8,rtol=1e-8)
    for step in selection['path']:
        ids=np.asarray(step['active']);after=factor_information(c0[np.ix_(ids,ids)],a0[ids])
        ratios=np.divide(after,info,out=np.ones_like(after),where=info>1e-8)
        np.testing.assert_allclose(ratios,step['retention'],atol=1e-8,rtol=1e-8)
    cov=np.asarray(fit['observed_covariance']);v=np.asarray(fit['global_loading']);u=np.asarray(fit['group_loading']);labels=np.asarray(fit['labels'])
    np.testing.assert_allclose(cov,np.cov(z[:,active],rowvar=False),atol=1e-12)
    component=np.outer(v,v)+(labels[:,None]==labels[None,:])*np.outer(u,u)
    c=component+np.diag(np.maximum(np.diag(cov)-np.diag(component),0));np.testing.assert_allclose(c,fit['model_covariance'],atol=1e-12)
    local=np.zeros(len(v))
    for g in np.unique(labels):
        vg=np.where(labels==g,v,0.);var=float(vg@c@vg)
        local[labels==g]=v[labels==g]*(vg@vg)/var if var>1e-14 else 0.
    expected=np.zeros(z.shape[1]);expected[active]=local
    rho=spearmanr(z@expected,x[:,ANCHOR]).statistic
    if np.isfinite(rho) and rho<0:expected*=-1
    expected/=np.abs(expected).sum();w=np.asarray(model['canonical_weights'])
    np.testing.assert_allclose(expected,w,atol=1e-12,rtol=1e-12)
    return float(np.max(np.abs(expected-w))),rounds


def independent_intervals(data,scores,gate,pairs=PAIRS):
    """Rebuild source sufficient statistics from raw peaks/pair comparisons."""
    names=sorted(set(k for pair in pairs for k in pair));cells=data['cells'].astype(str)
    cellnames=sorted(c for c in set(cells) if c.startswith('pb_'))
    target=data['target'];_,groups=np.unique(data['groups'],return_inverse=True);ng=groups.max()+1
    counts=np.zeros((ng,len(cellnames),2));success=np.zeros((ng,len(cellnames),2,len(names)))
    auc_total=np.zeros((ng,len(names)));auc_count=np.zeros(ng)
    for i,(a,b) in enumerate(zip(data['offsets'][:-1],data['offsets'][1:])):
        g=groups[i]
        if cells[i] in cellnames:
            c=cellnames.index(cells[i]);kind=int(target[i]>=0);counts[g,c,kind]+=1
            for j,k in enumerate(names):
                pred=int(np.argmax(scores[k][a:b])) if gate[i] else -1
                success[g,c,kind,j]+=int(pred==target[i])
        elif cells[i].startswith('prmbench'):
            y=data['labels'][a:b];pos=y==1;neg=y==0
            if pos.any() and neg.any():
                auc_count[g]+=1
                for j,k in enumerate(names):
                    s=scores[k][a:b];delta=s[pos,None]-s[None,neg]
                    auc_total[g,j]+=np.mean((delta>0)+.5*(delta==0))
    assert auc_count.sum()==6030
    rng=np.random.default_rng(400001);draws={p:{e:[] for e in ('pb','within')} for p in pairs}
    for _ in range(100):
        sample=rng.multinomial(ng,np.full(ng,1/ng),size=100)
        den=(sample@counts.reshape(ng,-1)).reshape(100,len(cellnames),2)
        num=(sample@success.reshape(ng,-1)).reshape(100,len(cellnames),2,len(names))
        rates=num/den[:,:,:,None];ca=rates[:,:,0];ea=rates[:,:,1]
        pb=np.divide(2*ca*ea,ca+ea,out=np.zeros_like(ca),where=ca+ea>0).mean(axis=1)
        auc=(sample@auc_total)/(sample@auc_count)[:,None]
        for pair in pairs:
            left,right=map(names.index,pair)
            draws[pair]['pb'].extend(pb[:,left]-pb[:,right]);draws[pair]['within'].extend(auc[:,left]-auc[:,right])
    q=.05/(2*len(pairs))/2
    return {a+' minus '+b:{e:np.quantile(v[e],[q,1-q]) for e in v} for (a,b),v in draws.items()}


def audit_checkpoint(path,model,args):
    cachepath=OUT/'CHECKPOINT_AUDIT.json'
    version=sha(Path(__file__));manifest=sha(OUT/'MANIFEST.json')
    cache=json.loads(cachepath.read_text()) if cachepath.exists() else {}
    if cache.get('auditor_sha256')!=version or cache.get('manifest_sha256')!=manifest:
        cache=dict(auditor_sha256=version,manifest_sha256=manifest,models={})
    key=path.name;digest=sha(path);old=cache['models'].get(key,{})
    if old.get('sha256')==digest:return old['weight_error'],old['rounds']
    assert 'membership' in model,model.get('failure')
    error,n=audit_model(model,args[0],args[1],args[2])
    cache['models'][key]=dict(sha256=digest,weight_error=error,rounds=n)
    dump(cachepath,cache);return error,n


def audit_near_stage(data,base,uids):
    diagnostic=json.loads((OUT/'NEAR_STAGE_DIAGNOSTIC.json').read_text())
    for p,h in json.loads((OUT/'NEAR_STAGE_MANIFEST.json').read_text())['hashes'].items():assert sha(p)==h,p
    with np.load(OUT/'NEAR_STAGE_SCORES.npz') as z:scores={k:z[k] for k in z.files}
    gate=scores.pop('gate');np.testing.assert_array_equal(gate,data['gate']);error=0.
    previous=ROOT/'results/joint_structured_stress_v1'
    with np.load(previous/'SCORES.npz') as z:
        np.testing.assert_array_equal(scores['base_refined'],z['base__refined'])
        np.testing.assert_array_equal(scores['near_refined'],z['near_copies__joint'])
    for kind in ('base','near'):
        x=base if kind=='base' else bank(data,base,uids,'near_copies')
        for outer in range(5):
            prefix='base_replay' if kind=='base' else 'near_copies'
            model=json.loads((previous/f'{prefix}_fold{outer}.json').read_text())['model']
            rows,args=training_inputs(data,x,outer);z=alias_coordinates(x,model['aliases']);m=model['membership']
            v=np.asarray(m['global_loading']);c=np.asarray(m['model_covariance']);lab=np.asarray(m['final_labels'])
            w=np.zeros(len(v))
            for group in np.unique(lab):
                vg=np.where(lab==group,v,0.);variance=vg@c@vg
                if variance>1e-14:w[lab==group]=v[lab==group]*(vg@vg)/variance
            expanded=np.zeros(z.shape[1]);expanded[np.asarray(m['active'])]=w
            rho=spearmanr(z[rows]@expanded,x[rows,ANCHOR]).statistic
            if np.isfinite(rho) and rho<0:expanded*=-1
            expanded/=np.abs(expanded).sum()
            np.testing.assert_allclose(expanded,diagnostic['weights'][f'{kind}_{outer}']['weights'],atol=1e-12)
            expected=z[~rows]@expanded;actual=scores[kind+'_full'][~rows]
            np.testing.assert_allclose(expected,actual,atol=1e-12);error=max(error,float(np.max(np.abs(expected-actual))))
    for k,s in scores.items():
        pb,auc,n=metrics(s,gate,data);m=diagnostic['metrics'][k]
        np.testing.assert_allclose([pb,auc],[m['pb'],m['within']],atol=1e-12,rtol=0);assert n==6030
    pairs=[('near_full','base_full'),('near_refined','base_refined'),
        ('base_refined','base_full'),('near_refined','near_full')]
    intervals=independent_intervals(data,scores,gate,pairs)
    for key,endpoints in intervals.items():
        for e,limits in endpoints.items():
            actual=diagnostic['primary_contrasts'][key][e]
            assert actual['confidence']==1-.05/8 and actual['draws']==10000
            np.testing.assert_allclose(limits,[actual['low'],actual['high']],atol=1e-12,rtol=0)
    return dict(status='PASS',metrics=4,fold_readouts=10,bootstrap_endpoints=8,
        score_replay_max_error=error,scope='separate post-evaluation diagnostic family')


def main():
    for p,h in json.loads((OUT/'MANIFEST.json').read_text())['hashes'].items():assert sha(p)==h,p
    test=runpy.run_path(str(ROOT/'tests/test_joint_staged_membership.py'));passed=[]
    for name,fn in test.items():
        if name.startswith('test_'):fn();passed.append(name)
    data,base,uids=inputs();result=json.loads((OUT/'RESULTS.json').read_text())
    with np.load(OUT/'SCORES.npz') as z:scores={k:z[k] for k in z.files}
    gate=scores.pop('gate');np.testing.assert_array_equal(gate,data['gate'])
    for name,s in references(data).items():np.testing.assert_array_equal(scores[name],s)
    native={};rounds=0;max_weight=0.;max_score=0.;changed_peaks={}
    for kind in KINDS:
        x=bank(data,base,uids,kind);native[kind]=0
        for outer in range(5):
            d=json.loads((OUT/f'{kind}_fold{outer}.json').read_text());m=d['model'];rows,args=training_inputs(data,x,outer)
            error,n=audit_checkpoint(OUT/f'{kind}_fold{outer}.json',m,args)
            max_weight=max(max_weight,error);rounds+=n
            if m['valid']:
                native[kind]+=int(np.sum(data['folds']==outer))
                expected=alias_coordinates(x[~rows],m['aliases'])@np.asarray(m['canonical_weights'])
                np.testing.assert_allclose(expected,x[~rows]@np.asarray(m['expanded_weights']),atol=1e-12)
                assert d['selected_added']==sum(i>=51 for i in m['active_original'])
                assert d['bocpd_retained']==(50 in m['active_original'])
            else:expected=x[~rows,ANCHOR]
            np.testing.assert_allclose(expected,scores[kind+'__staged'][~rows],atol=1e-12,rtol=1e-12)
            max_score=max(max_score,float(np.max(np.abs(expected-scores[kind+'__staged'][~rows]))))
            print('MODEL_PASS',kind,outer,flush=True)
        new=scores[kind+'__staged'];old=scores['base__staged']
        changed_peaks[kind]=int(sum(np.argmax(new[a:b])!=np.argmax(old[a:b]) for a,b in zip(data['offsets'][:-1],data['offsets'][1:])))
    assert native==result['native_answers']
    for name,s in scores.items():
        assert np.isfinite(s).all();pb,auc,n=metrics(s,gate,data);m=result['metrics'][name]
        np.testing.assert_allclose([pb,auc],[m['pb'],m['within']],atol=1e-12,rtol=0);assert n==m['within_n']==6030
    print('METRICS_PASS',len(scores),flush=True)
    intervals=independent_intervals(data,scores,gate)
    for key,endpoints in intervals.items():
        for e,limits in endpoints.items():
            actual=result['primary_contrasts'][key][e]
            assert actual['draws']==10000 and actual['confidence']==1-.05/12
            np.testing.assert_allclose(limits,[actual['low'],actual['high']],atol=1e-12,rtol=0)
    for k,passed_margin in result['practical_preservation'].items():
        c=result['primary_contrasts'][k+'__staged minus base__staged']
        assert passed_margin==(native[k]==13769 and c['pb']['low']>-.01 and c['within']['low']>-.002)
    error=float(np.max(np.abs(scores['base__staged']-scores['base__old'])))
    assert error==result['baseline_replay_max_error']
    assert result['baseline_preserved']==(native['base']==13769 and error<=1e-11)
    diagnostic_audit=audit_near_stage(data,base,uids)
    audit=dict(status='PASS',answers=len(uids),steps=len(base),metrics_checked=len(scores),
        sparse_rounds_checked=rounds,models_checked=25,native_answers=native,tests=passed,
        independent_bootstrap_endpoints=12,bootstrap_draws=10000,weight_replay_max_error=max_weight,
        score_replay_max_error=max_score,changed_peaks_vs_new_base=changed_peaks,
        near_stage_diagnostic=diagnostic_audit,score_sha256=sha(OUT/'SCORES.npz'),checks=['frozen hashes and reference scores',
        'zero-row removal before stable global-group support','null covariance and sparse objective',
        'full information-refinement path','final factor covariance and weights',
        'held scores and native coverage','independent metrics and full bootstrap replay'])
    dump(OUT/'AUDIT.json',audit);print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':
    if '--checkpoints-only' in sys.argv:
        data,base,uids=inputs();count=0
        for kind in KINDS:
            paths=[OUT/f'{kind}_fold{f}.json' for f in range(5)]
            if not any(p.exists() for p in paths):continue
            x=bank(data,base,uids,kind)
            for outer,path in enumerate(paths):
                if not path.exists():continue
                d=json.loads(path.read_text());_,args=training_inputs(data,x,outer)
                audit_checkpoint(path,d['model'],args);count+=1;print('CHECKPOINT_PASS',path.name,flush=True)
        print('CHECKPOINTS_CHECKED',count,flush=True)
    else:main()

