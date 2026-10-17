"""Unlabeled, source-disjoint DUFS selection from the already registered 31 streams."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from scripts import run_temporal_research_baseline as base
from spectral_utils.renyi_alpha_sweep import sweep_matrix,VIEW_NAMES,escort_varentropy
from spectral_utils.renyi_view_fusion import head_distribution
from spectral_utils.direct_probability_fusion import step_top_mean
from spectral_utils.adapted_dufs import adapted_dufs_soft_gates
from spectral_utils.temporal_research_features import BASELINE,SUBSETS


def prepare(baseout,records,joined,out):
    dest=out/'PREPARED.npz'
    if dest.exists():
        with np.load(dest,allow_pickle=False) as f:return {k:f[k] for k in f.files}
    steps=np.full((int(joined['offsets'][-1]),31),np.nan)
    samples=[];owners=[]
    con=sqlite3.connect('file:'+str(baseout/'CHECKPOINT.sqlite')+'?mode=ro',uri=True)
    try:
        for i,blob in con.execute('SELECT idx,payload FROM answers ORDER BY idx'):
            with np.load(io.BytesIO(blob),allow_pickle=False) as f:lp=f['logprobs15'].astype(float);spans=f['spans']
            matrix,names=sweep_matrix(lp)
            if tuple(names)!=tuple(VIEW_NAMES) or matrix.shape[1]!=31:raise ValueError('registered stream roster drift')
            q,_=head_distribution(lp,15);anchor=escort_varentropy(q,1.)
            centered=matrix-matrix.mean(axis=0);ac=anchor-anchor.mean()
            signs=np.where(centered.T@ac<0,-1.,1.);matrix*=signs
            scale=matrix.std(axis=0);z=(matrix-matrix.mean(axis=0))/np.where(scale>1e-10,scale,1.)
            # Deterministic evenly spaced training observations, with all answers available.
            pick=np.unique(np.linspace(0,len(matrix)-1,min(32,len(matrix))).astype(int))
            samples.append(z[pick].astype(np.float32));owners.extend([i]*len(pick))
            sl=slice(joined['offsets'][i],joined['offsets'][i+1])
            steps[sl]=np.column_stack([step_top_mean(matrix[:,j],spans[:,0],spans[:,1],10) for j in range(31)])
            if (i+1)%1000==0:print('[prepare31]',i+1,len(records),flush=True)
    finally:con.close()
    arrays=dict(steps=steps,samples=np.concatenate(samples),owners=np.array(owners,int))
    with dest.with_suffix('.tmp').open('wb') as f:np.savez_compressed(f,**arrays)
    dest.with_suffix('.tmp').replace(dest)
    return arrays


def fit_selection(samples,owners,metadata,cell,excluded):
    """Only unlabeled samples and {cell,group_id,fold} metadata enter selection."""
    if any(set(m)!={'cell','group_id','fold'} for m in metadata.values()):
        raise ValueError('selector metadata accepts only cell/group_id/fold; no correctness labels')
    train=[i for i,m in metadata.items() if m['cell']==cell and m['fold'] not in excluded]
    held_groups={m['group_id'] for m in metadata.values() if m['fold'] in excluded}
    train_groups={metadata[i]['group_id'] for i in train}
    if held_groups&train_groups:raise ValueError('source group leaked into selector fitting')
    rows_by_answer={i:np.flatnonzero(owners==i) for i in train}
    answers_by_group={}
    for i in train:answers_by_group.setdefault(metadata[i]['group_id'],[]).append(i)
    groups=sorted(answers_by_group)
    if not groups:raise ValueError('no DUFS training groups')
    key=cell+'__'+','.join(map(str,excluded))
    seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:4],'little')
    rng=np.random.default_rng(seed)
    sampled=[]
    for _ in range(8192):
        group=groups[int(rng.integers(len(groups)))];answers=answers_by_group[group]
        answer=answers[int(rng.integers(len(answers)))];rows=rows_by_answer[answer]
        sampled.append(rows[int(rng.integers(len(rows)))])
    gates,diagnostics=adapted_dufs_soft_gates(samples[sampled].T,seeds=(0,1,2),epochs=120)
    if not np.isfinite(gates).all():raise ValueError('DUFS nonfinite fit')
    order=np.lexsort((np.arange(31),-gates))
    return dict(cell=cell,excluded_folds=list(excluded),training_groups=groups,training_answers=len(train),
                sampling='8192 draws: uniform source group -> answer -> one of <=32 evenly spaced tokens',
                gates=gates,selected={str(k):order[:k].tolist() for k in (2,3,4)},diagnostics=diagnostics)


def run(source,baseout,out):
    if base.read_json(ROOT/'results/temporal_historical_pb_repair_v2/REPAIR_REVIEW.json')['status']!='PASS':
        raise ValueError('complete historical PB repair must pass before selector experiment')
    records,joined=base.load_contract(source)
    folds=base.read_json(base.evaluator.old.FOLDS)['outer']
    metadata={i:dict(cell=r['cell'],group_id=r['group_id'],fold=int(folds[r['group_id']])) for i,r in enumerate(records)}
    manifest=dict(schema='temporal-dufs31-v1',features=list(VIEW_NAMES),sizes=[2,3,4],seeds=[0,1,2],epochs=120,
                  fit_labels=False,selection_data='development diagnostics permitted; fitting unlabeled',
                  source_freeze=base.read_json(baseout/'SCORE_FREEZE.json'),code_sha256=base.common.sha256_file(Path(__file__)))
    if (out/'MANIFEST.json').exists() and base.read_json(out/'MANIFEST.json')!=manifest:raise ValueError('checkpoint manifest drift')
    base.common.atomic_json(out/'MANIFEST.json',manifest)
    arrays=prepare(baseout,records,joined,out)
    fitpath=out/'SELECTORS.json';fits=base.read_json(fitpath) if fitpath.exists() else {}
    for cell in sorted({r['cell'] for r in records}):
        ids=[i for i,r in enumerate(records) if r['cell']==cell]
        for excluded in base.common.excluded_sets(metadata,ids,cell):
            key=base.common.model_key(cell,excluded)
            if key in fits:continue
            fits[key]=base.common.json_ready(fit_selection(arrays['samples'],arrays['owners'],metadata,cell,excluded))
            base.common.atomic_json(fitpath,fits)
            print('[dufs]',key,{k:[VIEW_NAMES[j] for j in s] for k,s in fits[key]['selected'].items()},flush=True)
    total=int(joined['offsets'][-1]);scores={f'dufs31_k{k}':np.full(total,np.nan) for k in (2,3,4)}
    nested={(k,h):np.full(total,np.nan) for k in (2,3,4) for h in range(5)}
    for i,r in enumerate(records):
        fold=metadata[i]['fold'];sl=slice(joined['offsets'][i],joined['offsets'][i+1]);views=arrays['steps'][sl]
        selection=fits[base.common.model_key(r['cell'],(fold,))]['selected']
        for k in (2,3,4):scores[f'dufs31_k{k}'][sl]=views[:,selection[str(k)]].mean(axis=1)
        if not r['cell'].startswith('pb_'):
            for held in range(5):
                if held==fold:continue
                selection=fits[base.common.model_key(r['cell'],tuple(sorted((fold,held))))]['selected']
                for k in (2,3,4):nested[(k,held)][sl]=views[:,selection[str(k)]].mean(axis=1)
    thresholds={n:{} for n in scores}
    for held in range(5):
        train=[i for i,r in enumerate(records) if not r['cell'].startswith('pb_') and metadata[i]['fold']!=held]
        for k in (2,3,4):
            values=np.concatenate([nested[(k,held)][joined['offsets'][i]:joined['offsets'][i+1]] for i in train])
            if not np.isfinite(values).all():raise ValueError('nested calibration gap')
            thresholds[f'dufs31_k{k}'][str(held)]=float(np.quantile(values,.8))
    with np.load(baseout/'SCORES_FROZEN.npz',allow_pickle=False) as f:
        gate=f['gate_percentile']>=.33
        references={n:f['steps__'+n] for n in SUBSETS}
        references['append_innovation__H0lim']=f['steps__append_innovation__H0lim']
    metrics,per=base.evaluator.evaluate_arrays(records,joined,scores,fold_auc=True,calibration_thresholds=thresholds,pb_gate_open=gate)
    refmetrics,refper=base.evaluator.evaluate_arrays(records,joined,references,fold_auc=True,pb_gate_open=gate)
    metrics.update(refmetrics);per.update(refper);scores.update(references)
    pairs=[(f'dufs31_k{k}',BASELINE) for k in (2,3,4)]
    contrasts=base.evaluator.paired_bootstrap(records,joined,per,draws=10000,pairs=pairs,primary_pairs=set(pairs),primary_ci=1-.05/3)
    for a,b in pairs:contrasts[a+'_minus_'+b]['pb_delta']=metrics[a]['pb_all8']-metrics[b]['pb_all8']
    np.savez_compressed(out/'SCORES_FROZEN.npz',**{'steps__'+n:s for n,s in scores.items()})
    base.common.atomic_json(out/'METRICS.json',dict(metrics=metrics,contrasts=contrasts,development_only=True))
    names=list(metrics)
    frontier=[n for n in names if not any(metrics[m]['pb_all8']>=metrics[n]['pb_all8'] and metrics[m]['prm_within']>=metrics[n]['prm_within']
              and (metrics[m]['pb_all8']>metrics[n]['pb_all8'] or metrics[m]['prm_within']>metrics[n]['prm_within']) for m in names)]
    size=lambda n:len(SUBSETS[n]) if n in SUBSETS else (int(n[-1]) if n.startswith('dufs31_k') else 5)
    pb_best=min(names,key=lambda n:(-metrics[n]['pb_all8'],size(n),-metrics[n]['prmscore_q08']))
    within_best=min(names,key=lambda n:(-metrics[n]['prm_within'],size(n),-metrics[n]['prmscore_q08']))
    base.common.atomic_json(out/'BANK_SELECTION.json',dict(frontier=frontier,banks=list(dict.fromkeys([BASELINE,pb_best,within_best])),
        labels_used_for_configuration_selection=True,labels_used_for_selector_fit=False,
        note='DUFS candidates are source-excluded selection policies, not globally fixed rosters.'))
    base.common.atomic_json(out/'RUN_STATE.json',dict(status='COMPLETE_REVIEWED',models=len(fits),answers=len(records)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--baseline',type=Path,default=ROOT/'results/temporal_research_baseline_v1');p.add_argument('--out',type=Path,default=ROOT/'results/temporal_dufs31_v1')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    try:
        with threadpool_limits(limits=1):run(a.source_root,a.baseline,a.out)
    except BaseException as e:
        base.common.atomic_json(a.out/'RUN_STATE.json',dict(status='FAILED',error=f'{type(e).__name__}: {e}'));raise


if __name__=='__main__':main()
