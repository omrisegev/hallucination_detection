"""Independent score, membership, calibration and covariance audit."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,runpy
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_joint_mass_membership_v1 import OUT,inputs,KINDS,training_inputs,bank,references,PAIRS
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
    assert index==len(selection['path'])-1
    assert all(s['minimum_retention']>=.95 for s in selection['path'])
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


def audit_mass_discovery(values,folds,stored):
    from spectral_utils.joint_mass_groups import discover_mass_groups
    replay=discover_mass_groups(values,folds,seed=stored['seed'])
    assert replay['status']==stored['status']
    arrays=[values]+[values[folds!=f] for f in np.unique(folds)]
    records=[stored['mass_audit']]+stored['fold_mass_audits']
    for x,record in zip(arrays,records):
        c=np.cov(x,rowvar=False);scale=np.sqrt(np.maximum(np.diag(c),1e-12))
        kernel=np.clip(c/scale[:,None]/scale[None,:],-1,1)**2
        raw=np.asarray(record['raw_mass']);mass=np.asarray(record['mass'])
        assert np.isfinite(raw).all() and np.all(raw>=0)
        np.testing.assert_allclose(mass,raw/raw.sum(),atol=1e-12)
        gradient=kernel@raw-1;projected=np.where(raw>0,gradient,np.minimum(gradient,0))
        assert np.max(np.abs(projected))<=1e-6
        np.testing.assert_allclose(.5*raw@kernel@raw-raw.sum(),record['objective'],atol=1e-10)
    mass=np.asarray(stored['mass_audit']['mass'])
    for candidate,expected in zip(stored['candidates'],replay['candidates']):
        assert candidate['K']==expected['K'] and candidate['valid']==expected['valid']
        if 'labels' not in candidate:
            assert candidate['reason']==expected['reason'];continue
        np.testing.assert_array_equal(candidate['labels'],expected['labels'])
        np.testing.assert_array_equal(candidate['parts'],expected['parts'])
        left=np.asarray(candidate['labels'])
        for right,value in zip(candidate['parts'],candidate['stability']):
            right=np.asarray(right);table=np.zeros((left.max()+1,right.max()+1))
            np.add.at(table,(left,right),mass)
            a=table.sum(axis=1);b=table.sum(axis=0);expectation=a[:,None]*b[None,:]
            ok=table>0;mi=np.sum(table[ok]*np.log2(table[ok]/expectation[ok]))
            ha=-np.sum(a[a>0]*np.log2(a[a>0]));hb=-np.sum(b[b>0]*np.log2(b[b>0]))
            nmi=mi/np.sqrt(ha*hb) if min(ha,hb)>0 else float(ha==hb)
            np.testing.assert_allclose(nmi,value,atol=1e-10)
    return replay


def audit_checkpoint(path,model,args):
    cachepath=OUT/'CHECKPOINT_AUDIT.json'
    version=sha(Path(__file__));manifest=sha(OUT/'MANIFEST.json')
    cache=json.loads(cachepath.read_text()) if cachepath.exists() else {}
    if cache.get('auditor_sha256')!=version or cache.get('manifest_sha256')!=manifest:
        cache=dict(auditor_sha256=version,manifest_sha256=manifest,models={})
    key=path.name;digest=sha(path);old=cache['models'].get(key,{})
    if old.get('sha256')==digest:return old['weight_error'],old['rounds']
    if not model['valid']:
        from spectral_utils.joint_mass_membership import fit_mass_joint
        replay=fit_mass_joint(*args,anchor_index=ANCHOR,seed=model['seed'])
        assert not replay['valid'] and replay['failure']==model['failure']
    error,n=audit_model(model,args[0],args[1],args[2])
    z=alias_coordinates(args[0],model['aliases']);rowfold=np.repeat(args[3],np.diff(args[1]))
    if 'membership' in model:
        for r in model['membership']['rounds']:
            audit_mass_discovery(z[:,r['active_before']],rowfold,r['discovery'])
        if 'final_discovery' in model['membership']:
            audit_mass_discovery(z[:,model['membership']['active']],rowfold,model['membership']['final_discovery'])
    dynamic=audit_regroup_path(model,args[0],args[1],args[3])
    cache['models'][key]=dict(sha256=digest,weight_error=error,rounds=n,regrouping=dynamic)
    dump(cachepath,cache);return error,n


def audit_regroup_path(model,x,offsets,folds):
    from spectral_utils.joint_mass_groups import discover_mass_groups
    from spectral_utils.joint_feature_selection import checked_fit
    from spectral_utils.joint_pair_jacobian import profiled_pair_jacobian
    if not model['valid']:return dict(status='NON_NATIVE',failure=model['failure'])
    z=alias_coordinates(x,model['aliases']);initial=np.asarray(model['initial_active']);values=z[:,initial]
    rowfold=np.repeat(folds,np.diff(offsets));sel=model['refinement']['selection'];path=sel['path']
    c=np.cov(np.ascontiguousarray(values).T);discoveries=0;changed=0;rejections=0;budget_rejections=0
    for step,state in enumerate(path):
        ids=np.asarray(state['active']);labels=np.asarray(state['labels'])
        v=np.asarray(state['global_loading']);u=np.asarray(state['group_loading'])
        assert state['converged_starts']>=4 and state['multistart']['status']=='PASS'
        jac=profiled_pair_jacobian(v,u,labels)
        assert jac['full_global_rank'] and jac['condition_number']<=1e8
        np.testing.assert_allclose(jac['condition_number'],state['jacobian']['condition_number'],rtol=1e-8)
        if step>=len(sel['deletion_audit']):continue
        deletion=sel['deletion_audit'][step]
        groups=np.unique(labels);a=np.zeros((len(ids),1+len(groups)));a[:,0]=v
        for j,g in enumerate(groups):a[labels==g,j+1]=u[labels==g]
        sub=c[np.ix_(ids,ids)];reg=(1-1e-4)*sub+1e-4*np.diag(np.maximum(np.diag(sub),1e-12))
        precision=np.linalg.solve(reg,np.eye(len(ids)));transformed=precision@a
        info=np.sum(a*transformed,axis=0);loss=transformed**2/np.diag(precision)[:,None]
        importance=np.mean(loss[:,info>1e-8]/info[info>1e-8],axis=1)
        candidates=[int(i) for i in np.argsort(importance,kind='stable') if sum(labels==labels[i])>2][:3]
        attempts=list(deletion['rejected_candidates'])
        if 'removed' in deletion:attempts.append(dict(deletion,feature=deletion['removed']))
        assert len(attempts)<=len(candidates)
        for attempt,index in zip(attempts,candidates):
            assert attempt['feature']==int(ids[index]);proposal=np.delete(ids,index)
            np.testing.assert_array_equal(proposal,attempt['proposed_active'])
            reference_a=np.asarray(sel['initial_factor_matrix']);reference_info=np.asarray(sel['initial_factor_information'])
            relevant=reference_info>1e-8
            budget_info=factor_information(c[np.ix_(proposal,proposal)],reference_a[proposal])
            retention=np.divide(budget_info,reference_info,out=np.ones_like(budget_info),where=relevant)
            np.testing.assert_allclose(retention,attempt['information_retention'],atol=1e-8,rtol=1e-8)
            if attempt.get('reason')=='INFORMATION_BUDGET':
                assert np.any(retention[relevant]<.95) and attempt['discovery'] is None
                budget_rejections+=1
                continue
            assert np.all(retention[relevant]>=.95)
            replay=audit_mass_discovery(values[:,proposal],rowfold,attempt['discovery'])
            stored=attempt['discovery'];assert replay['status']==stored['status'];discoveries+=1
            if replay['status']=='SELECTED':
                np.testing.assert_array_equal(replay['labels'],stored['labels'])
                np.testing.assert_allclose(replay['stability'],stored['stability'],atol=1e-12)
            if 'reason' in attempt:
                rejections+=1
                if replay['status']!='SELECTED':assert attempt['reason']=='NO_ADMISSIBLE_REGROUPING'
                else:
                    try:checked_fit(c[np.ix_(proposal,proposal)],np.asarray(replay['labels']),sel['fit_seed']+step+1)
                    except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:assert str(exc)==attempt['reason']
                    else:raise AssertionError('REJECTED_FIT_NOW_VALID')
            else:
                following=path[step+1];np.testing.assert_array_equal(following['active'],proposal)
                np.testing.assert_array_equal(following['labels'],replay['labels'])
                kept=np.delete(labels,index)
                if not np.array_equal(kept[:,None]==kept[None,:],np.asarray(replay['labels'])[:,None]==np.asarray(replay['labels'])[None,:]):changed+=1
                np.testing.assert_allclose(importance[index],attempt['conditional_information_loss'],atol=1e-9,rtol=1e-7)
        if 'stopped' in deletion:assert len(attempts)==len(candidates)
    np.testing.assert_array_equal(model['refinement']['labels'],sel['automatic']['labels'])
    return dict(status='PASS',discoveries=discoveries,changed_partitions=changed,rejections=rejections,budget_rejections=budget_rejections)


def main():
    for p,h in json.loads((OUT/'MANIFEST.json').read_text())['hashes'].items():assert sha(p)==h,p
    test=runpy.run_path(str(ROOT/'tests/test_joint_mass_refinement.py'));test.update(runpy.run_path(str(ROOT/'tests/test_joint_mass_groups.py')));passed=[]
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
            np.testing.assert_allclose(expected,scores[kind+'__mass'][~rows],atol=1e-12,rtol=1e-12)
            max_score=max(max_score,float(np.max(np.abs(expected-scores[kind+'__mass'][~rows]))))
            print('MODEL_PASS',kind,outer,flush=True)
        new=scores[kind+'__mass'];old=scores['base__mass']
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
        c=result['primary_contrasts'][k+'__mass minus base__mass']
        assert passed_margin==(native[k]==13769 and c['pb']['low']>-.01 and c['within']['low']>-.002)
    error=float(np.max(np.abs(scores['base__mass']-scores['base__old'])))
    assert error==result['baseline_replay_max_error']
    ci=result['primary_contrasts']['base__mass minus base__previous']
    assert result['baseline_preserved']==(native['base']==13769 and ci['pb']['low']>-.01 and ci['within']['low']>-.002)
    audit=dict(status='PASS',answers=len(uids),steps=len(base),metrics_checked=len(scores),
        sparse_rounds_checked=rounds,models_checked=25,native_answers=native,tests=passed,
        independent_bootstrap_endpoints=12,bootstrap_draws=10000,weight_replay_max_error=max_weight,
        score_replay_max_error=max_score,changed_peaks_vs_new_base=changed_peaks,
        score_sha256=sha(OUT/'SCORES.npz'),checks=['frozen hashes and reference scores',
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
