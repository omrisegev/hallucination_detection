"""Independent metric audit, matched RBM, and innovation mechanism controls."""
from __future__ import annotations
import argparse
import io
import json
from pathlib import Path
import sqlite3
import sys
import hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from scripts import run_temporal_research_baseline as base
from spectral_utils.temporal_research_features import BASELINE,FEATURES,prefix_innovation
from spectral_utils.direct_probability_fusion import step_top_mean


def independent_pb(target,cells,pred,valid):
    """Scalar-loop calculation independent of the evaluator's vectorized PB code."""
    result={}
    for cell in sorted(set(cells)):
        if not cell.startswith('pb_'):continue
        clean_total=error_total=clean_hits=error_hits=0
        for t,c,p,v in zip(target,cells,pred,valid):
            if c!=cell:continue
            if t==-1:
                clean_total+=1;clean_hits+=int(v and p==-1)
            else:
                error_total+=1;error_hits+=int(v and p==t)
        clean=clean_hits/clean_total;error=error_hits/error_total
        result[cell]=dict(clean=clean_total,erroneous=error_total,clean_hits=clean_hits,error_hits=error_hits,
                          f1=2*clean*error/(clean+error) if clean+error else 0.)
    return result,float(np.mean([v['f1'] for v in result.values()]))


def run(source,baseout,out):
    records,joined=base.load_contract(source)
    old=base.read_json(baseout/'METRICS.json')['metrics']
    if base.read_json(baseout/'BASELINE_REPLAY.json')['status']!='PASS':raise ValueError('baseline not verified')
    cells=np.array([r['cell'] for r in records]);pb=np.char.startswith(cells,'pb_')
    with np.load(baseout/'SCORES_FROZEN.npz',allow_pickle=False) as f:
        names=[BASELINE,'append_innovation__H0lim','append_innovation__VE075','innovation__H0lim']
        scores={n:f['steps__'+n] for n in names}
        gate=f['gate_raw'];percentile=f['gate_percentile']
        h0=f['steps__mean__H0lim']
    with np.load(baseout/'PREDICTIONS.npz',allow_pickle=False) as f:
        audit={}
        for n in old:
            table,macro=independent_pb(joined['target'],cells,f['prediction__'+n],f['decision_valid__'+n])
            np.testing.assert_allclose(macro,old[n]['pb_all8'],rtol=0,atol=1e-14)
            for cell,v in table.items():
                np.testing.assert_allclose(v['f1'],old[n]['pb_cells'][cell]['f1'],rtol=0,atol=1e-14)
                if v['clean']!=old[n]['pb_cells'][cell]['clean'] or v['erroneous']!=old[n]['pb_cells'][cell]['erroneous']:
                    raise ValueError('PB denominator mismatch')
            audit[n]=dict(macro=macro,cells=table)
    base.common.atomic_json(out/'INDEPENDENT_PB_AUDIT.json',dict(status='PASS',methods=len(audit),results=audit))
    rbm_paths=[source/'.worktrees/rbm-supervision-matched-v1/results/rbm_supervision_matched_v1/SCORES.npz',
               source/'.worktrees/rbm-literature-completion-v1/results/rbm_literature_completion_v1/stability/SCORES.npz']
    rbm=[]
    for p in rbm_paths:
        manifest=base.read_json(p.parent/'MANIFEST.json')
        contract_hashes={k:v for k,v in manifest['hashes'].items() if k.replace('\\','/').endswith('/evaluation/JOINED.json')}
        if len(contract_hashes)!=1 or next(iter(contract_hashes.values()))!=base.common.sha256_file(base.evaluator.old.BENCH/'evaluation/JOINED.json'):
            raise ValueError('RBM frozen roster not bound to current benchmark')
        with np.load(p,allow_pickle=False) as f:rbm.append(f['steps__rbm12__logit_old'])
    np.testing.assert_array_equal(rbm[0],rbm[1])
    scores['RBM12_logit']=rbm[0]
    base.common.atomic_json(out/'RBM_PROVENANCE.json',dict(status='PASS',two_archives_equal=True,
        files={str(p):base.common.sha256_file(p) for p in rbm_paths},fit='historical answer-local unsupervised RBM12 logit',
        scope='saved score replay, not a model refit'))
    # Separate original scale/duplication from chronology; no correctness labels.
    controls=['append_duplicate_H0lim','append_centered_H0lim','append_shuffled_prefix_H0lim']
    for n in controls:scores[n]=np.full(int(joined['offsets'][-1]),np.nan)
    con=sqlite3.connect('file:'+str(baseout/'CHECKPOINT.sqlite')+'?mode=ro',uri=True)
    lengths=np.zeros(int(joined['offsets'][-1]),int)
    delta_peak={}
    try:
        for i,blob in con.execute('SELECT idx,payload FROM answers ORDER BY idx'):
            with np.load(io.BytesIO(blob),allow_pickle=False) as f:
                matrix=f['features'];spans=f['spans']
            sl=slice(joined['offsets'][i],joined['offsets'][i+1]);b=scores[BASELINE][sl]
            x=matrix[:,0];n=len(x)
            seed=int.from_bytes(hashlib.sha256(('temporal-innovation-control/'+records[i]['uid']).encode()).digest()[:8],'little')
            permutation=np.random.default_rng(seed).permutation(n)
            residual,_=prefix_innovation(x[permutation]);shuffled=np.empty(n);shuffled[permutation]=residual
            read=lambda v:step_top_mean(v,spans[:,0],spans[:,1],10)
            # Check the persisted true-prefix arm independently using explicit running sums.
            total=0.;independent=np.zeros(n)
            for t,value in enumerate(x):
                independent[t]=value-total/t if t else 0.;total+=value
            np.testing.assert_allclose((4*b+read(independent))/5,scores['append_innovation__H0lim'][sl],rtol=0,atol=2e-12)
            scores[controls[0]][sl]=(4*b+h0[sl])/5
            scores[controls[1]][sl]=(4*b+h0[sl]-x.mean())/5
            scores[controls[2]][sl]=(4*b+read(shuffled))/5
            lengths[sl]=spans[:,1]-spans[:,0]
            if (i+1)%1000==0:print('[controls]',i+1,len(records),flush=True)
    finally:con.close()
    np.savez_compressed(out/'SCORES_FROZEN.npz',**{'steps__'+n:s for n,s in scores.items()})
    metrics,per=base.evaluator.evaluate_arrays(records,joined,scores,fold_auc=True,pb_gate_open=percentile>=.33)
    # Frozen q applied to CDFs from other source folds, explicitly different calibration access.
    folds=base.read_json(base.evaluator.old.FOLDS)['outer'];fold=np.array([int(folds[r['group_id']]) for r in records])
    cross_rank=np.full(len(records),np.nan)
    for cell in sorted(set(cells[pb])):
        for f in sorted(set(fold[cells==cell])):
            train=np.flatnonzero((cells==cell)&(fold!=f));test=np.flatnonzero((cells==cell)&(fold==f))
            if not len(train):raise ValueError('no external gate calibration population')
            ordered=np.sort(gate[train]);query=gate[test]
            cross_rank[test]=(np.searchsorted(ordered,query,'left')+np.searchsorted(ordered,query,'right'))/(2*len(ordered))
    cross_metrics,cross_per=base.evaluator.evaluate_arrays(records,joined,{BASELINE:scores[BASELINE]},fold_auc=True,pb_gate_open=cross_rank>=.33)
    metrics['baseline_crossfold_gate']=cross_metrics[BASELINE];per['baseline_crossfold_gate']=cross_per[BASELINE]
    primary=[('append_innovation__H0lim','append_centered_H0lim'),('append_innovation__H0lim','append_shuffled_prefix_H0lim')]
    pairs=primary+[('RBM12_logit',BASELINE),('baseline_crossfold_gate',BASELINE),('append_innovation__H0lim',BASELINE)]
    contrasts=base.evaluator.paired_bootstrap(records,joined,per,draws=10000,pairs=pairs,primary_pairs=set(primary),primary_ci=.975)
    for a,b in pairs:contrasts[a+'_minus_'+b]['pb_delta']=metrics[a]['pb_all8']-metrics[b]['pb_all8']
    base.common.atomic_json(out/'METRICS.json',dict(metrics=metrics,contrasts=contrasts,development_only=True,
        design='Mechanism controls registered after observing initial innovation gain; not untouched confirmation.'))
    # Location/length strata are diagnostics, never a label-trained router.
    error=pb&(joined['target']>=0)
    step_counts=np.array([r['steps'] for r in records]);token_counts=np.array([r['tokens'] for r in records])
    error_position=np.where(error,joined['target']/np.maximum(step_counts-1,1),np.nan)
    first_is_longest=np.zeros(len(records),bool)
    for i in np.flatnonzero(error):
        sl=slice(joined['offsets'][i],joined['offsets'][i+1])
        first_is_longest[i]=int(np.argmax(lengths[sl]))==joined['target'][i]
    strata={'all_errors':error,'first_error_early':error&(error_position<1/3),
            'first_error_middle':error&(error_position>=1/3)&(error_position<2/3),
            'first_error_late':error&(error_position>=2/3),'error_is_longest':error&first_is_longest,
            'error_not_longest':error&~first_is_longest}
    for lower,upper in ((0,256),(256,512),(512,1024),(1024,1000000)):
        strata[f'answer_tokens_{lower}_{upper}']=error&(token_counts>=lower)&(token_counts<upper)
    diagnostics={}
    for label,mask in strata.items():
        data={'answers':int(mask.sum()),'methods':{}}
        for name,p in per.items():
            hit=p['decision_valid']&(p['prediction']==joined['target'])
            b=per[BASELINE]['decision_valid']&(per[BASELINE]['prediction']==joined['target'])
            data['methods'][name]=dict(exact=int((mask&hit).sum()),gained=int((mask&hit&~b).sum()),lost=int((mask&~hit&b).sum()),
                early=int((mask&(p['peak']<joined['target'])).sum()),late=int((mask&(p['peak']>joined['target'])).sum()))
        diagnostics[label]=data
    base.common.atomic_json(out/'STRATA.json',diagnostics)
    lines=['# Matched controls and innovation mechanism analysis','','All 13,769 development answers; 10,000 source-group bootstrap draws.','',
           '| Method | PB % | Within | PRMScore |','|---|---:|---:|---:|']
    for n,m in metrics.items():lines.append(f"| {n} | {100*m['pb_all8']:.4f} | {m['prm_within']:.6f} | {m['prmscore_q08']:.6f} |")
    lines+=['','True-prefix innovation was independently reconstructed with a scalar running-sum loop for every token.',
            'Centered/duplicated H0lim isolate reweighting and answer offset. The shuffled-prefix control breaks chronology.',
            'Control design follows the initial development finding. This is mechanism evidence, not independent confirmation.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    base.common.atomic_json(out/'RUN_STATE.json',dict(status='COMPLETE_REVIEWED',independent_pb_audit='PASS',full_program_complete=False))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--baseline',type=Path,default=ROOT/'results/temporal_research_baseline_v1')
    p.add_argument('--out',type=Path,default=ROOT/'results/temporal_research_mechanism_v1');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    try:
        with threadpool_limits(limits=1):run(a.source_root,a.baseline,a.out)
    except BaseException as e:
        base.common.atomic_json(a.out/'RUN_STATE.json',dict(status='FAILED',error=f'{type(e).__name__}: {e}'));raise


if __name__=='__main__':main()
