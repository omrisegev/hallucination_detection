"""Separated source fitting/calibration; never opens external labels or scores."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','LOKY_MAX_CPU_COUNT'):
    os.environ[key]='1'
import argparse, json, sys, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spectral_utils.external_generalization.fusion import fit_weights, standardize, answer_z, partition_equal, local_scores, top10, ARMS
from spectral_utils.external_generalization.artifacts import atomic_json, file_hash


def local_one(args):
    i, matrix, spans = args
    scores, diagnostics = local_scores(matrix,spans)
    return i, {k:v.tolist() for k,v in scores.items()},diagnostics


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--limit',type=int,default=0); args=parser.parse_args()
    out=ROOT/'results/lsml_external_generalization_v1/evaluation/source'
    out.mkdir(parents=True,exist_ok=True)
    paths={
      'joined':ROOT/'results/localization_full_benchmark_v3/evaluation/JOINED.json',
      'arrays':ROOT/'results/localization_full_benchmark_v3/evaluation/JOINED.npz',
      'folds':ROOT/'results/localization_source_group_audit_v1/FOLDS_V2.json',
      'tokens':ROOT/'.worktrees/token-probability-fusion-v1/results/token_probability_fusion_v1/TOKEN_MATRICES.npz',
      'level':ROOT/'.worktrees/token-probability-fusion-v1/results/token_probability_fusion_v1/DERIVATIVE_CHANNELS.npz',
      'ct7':ROOT/'.worktrees/token-probability-fusion-v1/results/chosen_token_calibration_v1/CT7_DEV_SCORES.npz'}
    records=json.loads(paths['joined'].read_text())['records']; n=len(records)
    off=np.load(paths['arrays'])['offsets']; foldmap=json.loads(paths['folds'].read_text())['outer']
    folds=np.array([foldmap[r['group_id']] for r in records]); sf=np.repeat(folds,np.diff(off))
    level=np.load(paths['level'])['level']; step=np.empty_like(level)
    ct7=np.load(paths['ct7'])['step_scores']; ct7z=np.empty_like(ct7)
    for i,(a,b) in enumerate(zip(off[:-1],off[1:])):
        step[a:b]=standardize(level[a:b]);ct7z[a:b]=answer_z(ct7[a:b])
    z=np.load(paths['tokens']); tokens=z['tokens']; toff=z['token_offsets'];spans=z['step_spans']
    # Full-population feature/readout replay, before new fitting or target scores.
    maxerr=0.
    for i,(a,b) in enumerate(zip(off[:-1],off[1:])):
        ta,tb=toff[i:i+2]
        maxerr=max(maxerr,float(np.max(np.abs(top10(tokens[ta:tb],spans[a:b])-level[a:b]))))
    if maxerr>1e-6:raise AssertionError(('source Top10 replay',maxerr))
    atomic_json(out/'INPUTS.json',{'paths':{k:{'path':str(p),'sha256':file_hash(p)} for k,p in paths.items()},'answers':n,'steps':int(off[-1]),'top10_max_error':maxerr})
    checkpoint=out/'local_records';checkpoint.mkdir(exist_ok=True)
    pending=[]
    for i in range(args.limit or n):
        if not (checkpoint/(str(i)+'.json')).exists():pending.append(i)
    def work():
        for i in pending:
            a,b=off[i:i+2];ta,tb=toff[i:i+2]
            yield i,tokens[ta:tb],spans[a:b]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for j,(i,s,d) in enumerate(pool.map(local_one,work(),chunksize=8)):
            atomic_json(checkpoint/(str(i)+'.json'),{'index':i,'scores':s,'diagnostics':d})
            if j%100==0:print('local',j+1,'/',len(pending),flush=True)
    if args.limit:return
    scores={k:np.empty(off[-1]) for k in ARMS if k.startswith('local_')};native=[]
    for i in range(n):
        d=json.loads((checkpoint/(str(i)+'.json')).read_text());a,b=off[i:i+2]
        for k in scores:scores[k][a:b]=d['scores'][k]
        native.append(d['diagnostics']['native'])
    scores['ct7']=ct7z
    def frozen(fit):
        raw={'frozen_lsml':step@np.array(fit['weights']), 'frozen_equal':step.mean(1),
             'frozen_partition_equal':step@partition_equal(fit['groups'])}
        for values in raw.values():
            for a,b in zip(off[:-1],off[1:]): values[a:b]=answer_z(values[a:b])
        return raw
    fits=[];pred={k:np.empty(off[-1],dtype=np.int8) for k in (*ARMS,'ct7')}
    oof={k:np.empty(off[-1]) for k in (*ARMS,'ct7')}
    for test in range(5):
        calibration=(test+1)%5;fit=fit_weights(step[(sf!=test)&(sf!=calibration)])
        current={**scores,**frozen(fit)}
        thresholds={k:float(np.quantile(v[sf==calibration],.8,method='linear')) for k,v in current.items()}
        for k,v in current.items():
            oof[k][sf==test]=v[sf==test];pred[k][sf==test]=(v[sf==test]<thresholds[k])
        fits.append({'test':test,'calibration':calibration,'fit':fit,'thresholds':thresholds})
        print('source fold',test,flush=True)
    fit=fit_weights(step[sf<4]);current={**scores,**frozen(fit)}
    thresholds={k:float(np.quantile(v[sf==4],.8,method='linear')) for k,v in current.items()}
    bundle={'fit':fit,'thresholds':thresholds,'fit_folds':[0,1,2,3],'calibration_fold':4,
            'population':'PB+PRMB existing13769, unlabeled fit and calibration',
            'inputs_sha256':file_hash(out/'INPUTS.json'),'arms':list(current),'source_native_answers':sum(native)}
    atomic_json(out/'BUNDLE.json',bundle);atomic_json(out/'VALIDATION_FITS.json',fits)
    np.savez_compressed(out/'VALIDATION.npz',**{k+'_score':v for k,v in oof.items()},
                        **{k+'_pred':v for k,v in pred.items()},offsets=off,folds=folds,native=native)
    print('SOURCE COMPLETE',flush=True)

if __name__=='__main__': main()
