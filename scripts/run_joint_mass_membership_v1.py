"""One frozen mass refinement intervention on five full banks."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_joint_fixed_readout_v1 import inputs
from scripts.run_joint_structured_stress_v1 import training_inputs
from scripts.run_joint_feature_selection_bocpd_v1 import bootstrap
from scripts.run_digitfree_broad50_v1 import sha
from scripts.run_lsml_gate_locator_research_v1 import dump,score_locator,scalar_metrics
from spectral_utils.joint_mass_membership import fit_mass_joint,score_noise_aware_joint
from spectral_utils.joint_redundancy_stress import augment
from spectral_utils.joint_structured_stress import structured_augmentation
from spectral_utils.digitfree_broad50 import ANCHOR
OUT=ROOT/'results/joint_mass_membership_v1'
OLD=ROOT/'results/joint_sparse_refinement_v1'
HARD=ROOT/'results/joint_structured_stress_v1'
PREVIOUS=ROOT/'results/joint_staged_membership_v1'
ROBUST=ROOT/'results/joint_regroup_membership_v1'
LAST=ROOT/'results/joint_feasible_membership_v1'
MINIMAX=ROOT/'results/joint_minimax_membership_v1'
KINDS=('base','duplicates','noise','near_copies','structured_noise')
PAIRS=[('near_copies__mass','near_copies__feasible_reference'),('base__mass','base__previous')]+[
    (k+'__mass','base__mass') for k in KINDS[1:]]


def bank(data,base,uids,kind):
    if kind=='base':return base
    if kind in ('duplicates','noise'):return augment(base,data['offsets'],uids,kind)
    return structured_augmentation(base,data['offsets'],uids,kind)


def references(data):
    scores={}
    with np.load(OLD/'SCORES.npz') as z:
        np.testing.assert_array_equal(z['gate'],data['gate'])
        for k in KINDS[:3]:
            scores[k+'__old']=z[k+'__refined'];scores[k+'__continuous']=z[k+'__continuous']
            scores[k+'__equal']=z[k+'__equal_full']
        for k in ('innovation5','historical_bocpd'):scores[k]=z[k]
    with np.load(HARD/'SCORES.npz') as z:
        np.testing.assert_array_equal(z['gate'],data['gate'])
        for k in KINDS[3:]:
            for arm,old in [('old','joint'),('continuous','continuous'),('equal','equal')]:scores[k+'__'+arm]=z[k+'__'+old]
    with np.load(PREVIOUS/'SCORES.npz') as z:
        np.testing.assert_array_equal(z['gate'],data['gate'])
        for k in KINDS:scores[k+'__previous']=z[k+'__staged']
    with np.load(ROBUST/'SCORES.npz') as z:
        np.testing.assert_array_equal(z['gate'],data['gate'])
        for k in KINDS:scores[k+'__regroup_reference']=z[k+'__regroup']
    with np.load(LAST/'SCORES.npz') as z:
        np.testing.assert_array_equal(z['gate'],data['gate'])
        for k in KINDS:scores[k+'__feasible_reference']=z[k+'__feasible']
    with np.load(MINIMAX/'SCORES.npz') as z:
        np.testing.assert_array_equal(z['gate'],data['gate'])
        for k in KINDS:scores[k+'__minimax_reference']=z[k+'__minimax']
    return scores


def state(status,**detail):
    dump(OUT/'RUN_STATE.json',dict(status=status,time=time.strftime('%Y-%m-%dT%H:%M:%S'),**detail))
    print(status,detail,flush=True)


def main():
    OUT.mkdir(exist_ok=True)
    modules=('joint_mass_groups','joint_mass_structure','joint_mass_membership','joint_mass_refinement','joint_staged_membership','joint_sparse_membership','joint_sparse_refinement',
        'joint_feature_selection','joint_block_balanced','joint_group_reliability',
        'joint_structured_stress','joint_redundancy_stress','joint_pair_jacobian','joint_lsml')
    files=[Path(__file__),ROOT/'docs/experiments/JOINT_MASS_REFINEMENT_V1.md',ROOT/'tests/test_joint_mass_groups.py',ROOT/'tests/test_joint_mass_refinement.py',
        *[ROOT/f'spectral_utils/{m}.py' for m in modules],
        *[p/name for p in (OLD,HARD,PREVIOUS,ROBUST,LAST,MINIMAX) for name in ('SCORES.npz','MANIFEST.json','AUDIT.json')]]
    manifest=dict(schema='joint-mass-membership-v1',hashes={str(p):sha(p) for p in files})
    path=OUT/'MANIFEST.json'
    if path.exists() and json.loads(path.read_text())!=manifest:raise ValueError('MANIFEST_DRIFT')
    dump(path,manifest);state('LOADING');data,base,uids=inputs();scores=references(data)
    native={};rosters={};base_error=None
    for kind in KINDS:
        x=bank(data,base,uids,kind);scores[kind+'__mass']=np.full(len(x),np.nan)
        native[kind]=0;rosters[kind]=[]
        for outer in range(5):
            rows,args=training_inputs(data,x,outer);path=OUT/f'{kind}_fold{outer}.json'
            if path.exists():meta=json.loads(path.read_text());model=meta['model']
            else:
                started=time.perf_counter()
                def notify(stage,**detail):state(stage,bank=kind,outer=outer,**detail)
                model=fit_mass_joint(*args,anchor_index=ANCHOR,seed=399170+outer,notify=notify)
                selected=model.get('active_original',[])
                meta=dict(model=model,kind=kind,outer=outer,seconds=time.perf_counter()-started,
                    selected_original=sum(i<51 for i in selected),selected_added=sum(i>=51 for i in selected),
                    bocpd_retained=50 in selected)
                dump(path,meta)
            scores[kind+'__mass'][~rows]=score_noise_aware_joint(model,x[~rows],allow_anchor_fallback=True)
            if model['valid']:native[kind]+=int(np.sum(data['folds']==outer))
            rosters[kind].append({k:meta[k] for k in ('selected_original','selected_added','bocpd_retained','seconds')})
            state('FOLD_COMPLETE',bank=kind,outer=outer,valid=model['valid'],
                kept=meta['selected_original']+meta['selected_added'],seconds=meta['seconds'])
        if kind=='base':
            base_error=float(np.max(np.abs(scores['base__mass']-scores['base__old'])))
            state('BASE_REPLAY',max_error=base_error)
    state('EVALUATING');metrics={k:score_locator(s,data['gate'],data) for k,s in scores.items()}
    ci=bootstrap(data,metrics,PAIRS)
    preservation={k:bool(native[k]==13769 and ci[k+'__mass minus base__mass']['pb']['low']>-.01 and
        ci[k+'__mass minus base__mass']['within']['low']>-.002) for k in KINDS[1:]}
    result=dict(status='COMPLETE',metrics={k:scalar_metrics(m) for k,m in metrics.items()},
        native_answers=native,rosters=rosters,primary_contrasts=ci,practical_preservation=preservation,
        fresh_fits=25,equivalent_replays=0,baseline_replay_max_error=base_error,baseline_preserved=bool(native['base']==13769 and ci['base__mass minus base__previous']['pb']['low']>-.01 and ci['base__mass minus base__previous']['within']['low']>-.002))
    dump(OUT/'RESULTS.json',result);np.savez_compressed(OUT/'SCORES.npz',**scores,gate=data['gate'])
    lines=['# Information-mass refinement: full matched evaluation','','| Method | PB % | Within AUC |','|---|---:|---:|']
    for k,m in result['metrics'].items():lines.append(f"| {k} | {100*m['pb']:.4f} | {m['within']:.6f} |")
    lines+=['','Native: '+str(native),'Preservation: '+str(preservation),'','```json',json.dumps(ci,indent=2),'```']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    state('COMPLETE',preservation=preservation,baseline_preserved=result['baseline_preserved'])


if __name__=='__main__':
    try:main()
    except Exception as exc:state('FAILED',error=str(exc));raise
