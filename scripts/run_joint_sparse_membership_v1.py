"""Full-population null-row selection through sparse Joint factor loadings."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_joint_fixed_readout_v1 import inputs,BANKS
from scripts.run_joint_feature_selection_bocpd_v1 import bootstrap
from scripts.run_digitfree_broad50_v1 import sha
from scripts.run_lsml_gate_locator_research_v1 import dump,score_locator,scalar_metrics
from spectral_utils.joint_redundancy_stress import augment
from spectral_utils.joint_sparse_membership import exact_aliases,alias_coordinates,source_grams,calibrate_penalty,sparse_membership
from spectral_utils.joint_block_balanced import discover_sourcefold_groups
from spectral_utils.joint_feature_selection import checked_fit
from spectral_utils.joint_lsml import covariance_matrix
from spectral_utils.joint_group_reliability import group_reliability_weights
from spectral_utils.lsml_gate_locator_research import _orient
from spectral_utils.digitfree_broad50 import ANCHOR
PREVIOUS=ROOT/'results/joint_group_reliability_v1';OUT=ROOT/'results/joint_sparse_membership_v1'


def state(status,**kwargs):
    dump(OUT/'RUN_STATE.json',dict(status=status,time=time.strftime('%Y-%m-%dT%H:%M:%S'),**kwargs))
    print(status,kwargs,flush=True)


def select_fit(train,rowfold,grams,nrows,outer):
    active=np.arange(train.shape[1]);rounds=[]
    for round_index in range(3):
        sub=train[:,active];cov=covariance_matrix(sub)
        discovery=discover_sourcefold_groups(sub,rowfold,seed=399170+100+outer)
        if discovery['status']!='SELECTED':raise ValueError('NO_ADMISSIBLE_SPARSE_STRUCTURE')
        penalty,calibration=calibrate_penalty(grams[:,active][:,:,active],nrows,cov,
            np.asarray(discovery['labels']),seed=404170+outer+100*round_index)
        state('SPARSE_FIT',outer=outer,round=round_index,features=len(active),penalty=penalty)
        local,audit=sparse_membership(cov,np.asarray(discovery['labels']),penalty=penalty,
            seed=399170+200+outer+round_index)
        selected=active[local]
        rounds.append(dict(round=round_index,active_before=active,active_after=selected,
            discovery=discovery,calibration=calibration,sparse=audit))
        state('MEMBERSHIP',outer=outer,round=round_index,kept=len(selected),
            converged=audit['converged_starts'],support_agreement=audit['supports_agree'])
        if len(selected)<6:raise ValueError('TOO_FEW_ACTIVE_MEASUREMENTS')
        if np.array_equal(active,selected):break
        active=selected
    sub=train[:,active];cov=covariance_matrix(sub)
    discovery=discover_sourcefold_groups(sub,rowfold,seed=399170+100+outer)
    candidates=sorted([c for c in discovery['candidates'] if c['valid']],
        key=lambda c:(-c['median_ari'],-c['mean_ari'],-c['minimum_ari'],c['K']))
    attempts=[]
    for candidate in candidates:
        try:
            checked=checked_fit(cov,np.asarray(candidate['labels']),seed=399170+200+outer)
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
            attempts.append(dict(K=candidate['K'],valid=False,reason=str(exc)));continue
        j=checked.joint;local,readout=group_reliability_weights(j.model_covariance,j.global_loading,candidate['labels'])
        w=np.zeros(train.shape[1]);w[active]=local
        attempts.append(dict(K=candidate['K'],valid=True))
        return dict(valid=True,active=active,unoriented_weights=w,rounds=rounds,attempts=attempts,
            final_discovery=discovery,final_labels=candidate['labels'],readout=readout,
            observed_covariance=cov,model_covariance=j.model_covariance,global_loading=j.global_loading,
            group_loading=j.group_loading,diagonal_audit=j.diagonal_audit,
            converged_starts=j.converged_starts,multistart=j.multistart_audit,jacobian=j.jacobian_audit)
    return dict(valid=False,rounds=rounds,active=active,attempts=attempts,failure='NO_VALID_DEBIASED_STRUCTURE')


def main():
    OUT.mkdir(exist_ok=True)
    files=[Path(__file__),ROOT/'spectral_utils/joint_sparse_membership.py',
        ROOT/'spectral_utils/joint_group_reliability.py',ROOT/'spectral_utils/joint_block_balanced.py',
        ROOT/'spectral_utils/joint_feature_selection.py',ROOT/'docs/experiments/JOINT_SPARSE_MEMBERSHIP_V1.md',
        PREVIOUS/'SCORES.npz',PREVIOUS/'AUDIT.json',PREVIOUS/'MANIFEST.json']
    manifest=dict(schema='joint-sparse-membership-v1',hashes={str(p):sha(p) for p in files})
    mp=OUT/'MANIFEST.json'
    if mp.exists() and json.loads(mp.read_text())!=manifest:raise ValueError('MANIFEST_DRIFT')
    dump(mp,manifest);state('LOADING');data,base,uids=inputs()
    rowfold=np.repeat(data['folds'],np.diff(data['offsets']))
    with np.load(PREVIOUS/'SCORES.npz') as z:
        scores={k:z[k] for k in z.files if k!='gate'};np.testing.assert_array_equal(z['gate'],data['gate'])
    native={};fit_cache={};all_meta=[]
    for bank in BANKS:
        x=base if bank=='base' else augment(base,data['offsets'],uids,bank)
        scores[bank+'__sparse']=np.full(len(x),np.nan);native[bank+'__sparse']=0
        cached_groups=None;grams=None
        for outer in range(5):
            started=time.perf_counter();held=rowfold==outer;path=OUT/f'{bank}_fold{outer}.json'
            if path.exists():
                meta=json.loads(path.read_text());groups=meta['aliases'];z=alias_coordinates(x,groups)
                fit_cache[(outer,meta['training_digest'])]=meta['fit']
            else:
                state('CANONICALIZE',bank=bank,outer=outer)
                groups=exact_aliases(x[~held]);z=alias_coordinates(x,groups);train=z[~held]
                digest=hashlib.sha256(np.ascontiguousarray(train).tobytes()).hexdigest();key=(outer,digest)
                if key in fit_cache:
                    fit=fit_cache[key];reused=True
                else:
                    if groups!=cached_groups:
                        grams,sizes,sourcefold=source_grams(z,data['offsets'],data['groups'],data['folds']);cached_groups=groups
                    try:
                        fit=select_fit(train,rowfold[~held],grams[sourcefold!=outer],int(sizes[sourcefold!=outer].sum()),outer)
                    except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                        fit=dict(valid=False,failure=f'{type(exc).__name__}: {exc}')
                    fit_cache[key]=fit;reused=False
                if fit['valid']:
                    extended=np.column_stack((train,x[~held,ANCHOR]))
                    w,orientation=_orient(extended,np.r_[fit['unoriented_weights'],0.],train.shape[1]);w=w[:-1]
                    expanded=np.zeros(x.shape[1])
                    for j,ids in enumerate(groups):expanded[ids]=w[j]/len(ids)
                else:
                    w=np.zeros(z.shape[1]);expanded=np.eye(x.shape[1])[ANCHOR];orientation=None
                meta=dict(bank=bank,outer=outer,aliases=groups,training_digest=digest,fit=fit,
                    canonical_weights=w,expanded_weights=expanded,orientation=orientation,
                    reused_canonical_fit=reused,seconds=time.perf_counter()-started)
                if fit['valid']:
                    active_original=[i for j in fit['active'] for i in groups[j]]
                    meta.update(active_original=active_original,selected_added=int(sum(i>=51 for i in active_original)),
                        bocpd_retained=50 in active_original)
                dump(path,meta)
            if meta['fit']['valid']:
                scores[bank+'__sparse'][held]=z[held]@np.asarray(meta['canonical_weights'])
                native[bank+'__sparse']+=int(np.sum(data['folds']==outer))
            else:scores[bank+'__sparse'][held]=x[held,ANCHOR]
            all_meta.append(meta);state('FOLD_COMPLETE',bank=bank,outer=outer,valid=meta['fit']['valid'],
                selected_added=meta.get('selected_added'),seconds=meta['seconds'])
        # Alias-only control is exact reuse of the same canonical full bank.
        control='base' if bank=='duplicates' else bank
        if bank=='duplicates':np.testing.assert_array_equal(alias_coordinates(x,exact_aliases(x)),base)
        scores[bank+'__alias_control']=scores[control+'__reliability_full'].copy()
        del grams
    state('EVALUATING');metrics={k:score_locator(s,data['gate'],data) for k,s in scores.items()}
    pairs=[(b+'__sparse',b+'__reliability_auto') for b in BANKS]
    pairs += [(b+'__sparse','base__sparse') for b in ('duplicates','noise')]
    ci=bootstrap(data,metrics,pairs)
    preservation={b:bool(native[b+'__sparse']==13769 and ci[b+'__sparse minus base__sparse']['pb']['low']>-.01 and
        ci[b+'__sparse minus base__sparse']['within']['low']>-.002) for b in ('duplicates','noise')}
    result=dict(status='COMPLETE',metrics={k:scalar_metrics(m) for k,m in metrics.items()},primary_contrasts=ci,
        native_answers=native,practical_preservation=preservation,
        exact_copy_max_score_difference=float(np.max(np.abs(scores['base__sparse']-scores['duplicates__sparse']))),
        exact_copy_peak_changes=int(np.sum(metrics['base__sparse']['peaks']!=metrics['duplicates__sparse']['peaks'])),
        selected_counts={b:[len(m['fit'].get('active',[])) for m in all_meta if m['bank']==b] for b in BANKS},
        selected_added={b:[m.get('selected_added') for m in all_meta if m['bank']==b] for b in BANKS},
        bocpd_retained={b:[m.get('bocpd_retained') for m in all_meta if m['bank']==b] for b in BANKS})
    dump(OUT/'RESULTS.json',result);np.savez_compressed(OUT/'SCORES.npz',**scores,gate=data['gate'])
    lines=['# Sparse Joint loading membership','','| Method | PB % | Within AUC | Native |','|---|---:|---:|---:|']
    for k,m in result['metrics'].items():lines.append(f"| {k} | {100*m['pb']:.4f} | {m['within']:.6f} | {native.get(k,'reference/control')} |")
    lines+=['','Practical preservation: '+str(preservation),'Selected counts: '+str(result['selected_counts']),
        'Selected additions: '+str(result['selected_added']),'','```json',json.dumps(ci,indent=2),'```']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8');state('COMPLETE',practical_preservation=preservation)
    print('\n'.join(lines[33:43]),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:state('FAILED',error=str(exc));raise
