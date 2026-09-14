"""One resumable source-excluded TCN/FM/DiFlo fit; no correctness labels loaded."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from spectral_utils.context_training import FeatureBundle
from spectral_utils.temporal_context_models import (ConditionalFlow,TelemetryTCN,flow_condition,
    flow_objective,probability_pgd,gaussian_prediction_loss,flow_dot)

STOP=False
def stop(*_):
    global STOP
    STOP=True


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def atomic_torch(path,value):
    temp=path.with_suffix('.tmp');torch.save(value,temp);temp.replace(path)


def loss_for(model,bundle,batch,method):
    if method=='tcn':
        mean,variance=model(batch['history'],batch['mask'],batch['position'])
        return gaussian_prediction_loss(mean,variance,batch['target'])
    target=batch['target'];noise=torch.randn_like(target);tau=torch.rand(len(target),1,device=target.device)
    state=(1-tau)*noise+tau*target;velocity=target-noise
    condition=flow_condition(batch['history'],batch['mask'],batch['position'])
    negative=None
    if method=='diflo':
        negative,_=probability_pgd(model,state,tau,velocity,batch['source_logits'],
            lambda lp:bundle.condition_from_probabilities(batch,lp),epsilon=.1,iterations=3)
    return flow_objective(model,state,tau,velocity,condition,negative,
                          repel_weight=.1 if method=='diflo' else 0.,curve_weight=.1 if method=='diflo' else 0.)[0]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--method',choices=['tcn','fm','diflo'],required=True)
    p.add_argument('--bank',choices=['original4','innovation5'],default='original4')
    p.add_argument('--excluded-folds',default='0');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--steps',type=int,default=50000);p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--smoke',action='store_true')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if not 1<=a.steps<=50000:raise ValueError('invalid training cap')
    if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable; no silent CPU job')
    torch.set_num_threads(1);torch.manual_seed(a.seed);rng=np.random.default_rng(a.seed)
    excluded=tuple(sorted(set(int(x) for x in a.excluded_folds.split(','))))
    if len(excluded) not in (1,2) or any(x not in range(5) for x in excluded):raise ValueError('invalid held source folds')
    bundle=FeatureBundle(a.data,a.bank);train,validation,held=bundle.split(excluded)
    dimensions=len(bundle.columns)
    model=(TelemetryTCN(dimensions=dimensions) if a.method=='tcn' else ConditionalFlow(17*dimensions+18,dimensions=dimensions)).to(a.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-3)
    validation_every=min(500,a.steps);validation_batches=2 if a.smoke else 8
    manifest=dict(schema='unlabeled-context-training-v1',method=a.method,bank=a.bank,excluded_folds=excluded,
        seed=a.seed,max_updates=a.steps,batch_size=a.batch_size,device=a.device,smoke=a.smoke,
        normalization='whole-answer mean/std; offline; base score retained in downstream readout',
        loss='Gaussian telemetry prediction' if a.method=='tcn' else 'FM plus repel/curve' if a.method=='diflo' else 'FM',
        training_groups=sorted({bundle.metadata[i]['group_id'] for i in train}),
        validation_groups=sorted({bundle.metadata[i]['group_id'] for i in validation}),
        held_groups=sorted({bundle.metadata[i]['group_id'] for i in held}),
        data_manifest_sha256=hashlib.sha256((a.data/'MANIFEST.json').read_bytes()).hexdigest(),
        code_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in
            (Path(__file__),ROOT/'spectral_utils/context_training.py',ROOT/'spectral_utils/temporal_context_models.py')},
        early_stop='6 consecutive held-validation checkpoints without improvement; training objective, no task labels')
    if (a.out/'MANIFEST.json').exists():
        old=json.loads((a.out/'MANIFEST.json').read_text());current=json.loads(json.dumps(manifest))
        if old!=current:raise ValueError('training checkpoint manifest changed')
    atomic_json(a.out/'MANIFEST.json',manifest)
    start=0;best=float('inf');stale=0;history=[]
    checkpoint=a.out/'LATEST.pt'
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location=a.device,weights_only=False)
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        start=saved['step'];best=saved['best'];stale=saved['stale'];history=saved['history']
        rng.bit_generator.state=saved['numpy_rng'];torch.set_rng_state(saved['torch_rng'].cpu())
        if a.device=='cuda':torch.cuda.set_rng_state_all(saved['cuda_rng'])
    signal.signal(signal.SIGTERM,stop)
    began=time.perf_counter();flow=a.method!='tcn'
    step=start
    def save_current():
        atomic_torch(checkpoint,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,best=best,stale=stale,
            history=history,numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if a.device=='cuda' else None))
    try:
        for step in range(start+1,a.steps+1):
            model.train();answers,positions=bundle.sampler(train,rng,a.batch_size,flow=flow)
            batch=bundle.batch(answers,positions,flow=flow,device=a.device)
            loss=loss_for(model,bundle,batch,a.method)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite training loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            optimizer.step()
            if step%validation_every==0 or step==a.steps:
                model.eval();vrng=np.random.default_rng(20260915)
                # Fixed validation observations and FM noises, without disturbing training RNG.
                state=torch.get_rng_state();cuda_state=torch.cuda.get_rng_state_all() if a.device=='cuda' else None
                torch.manual_seed(20260915);values=[]
                for _ in range(validation_batches):
                    ids,pos=bundle.sampler(validation,vrng,a.batch_size,flow=flow)
                    vb=bundle.batch(ids,pos,flow=flow,device=a.device)
                    # DiFlo PGD needs input gradients on validation, but never updates model weights.
                    with torch.set_grad_enabled(a.method=='diflo'):v=loss_for(model,bundle,vb,a.method)
                    values.append(float(v.detach()))
                torch.set_rng_state(state)
                if cuda_state is not None:torch.cuda.set_rng_state_all(cuda_state)
                value=float(np.mean(values))
                if not np.isfinite(value):raise FloatingPointError('nonfinite validation objective')
                if value<best:
                    best=value;stale=0;atomic_torch(a.out/'BEST.pt',dict(model=model.state_dict(),step=step,validation_loss=value))
                else:stale+=1
                history.append(dict(step=step,train_loss=float(loss.detach()),validation_loss=value,gradient_norm=float(gradient),seconds=time.perf_counter()-began))
                save_current();atomic_json(a.out/'TRAINING.json',history)
                atomic_json(a.out/'RUN_STATE.json',dict(status='TRAINING',step=step,best_validation=best,stale=stale,smoke=a.smoke))
                print(json.dumps(history[-1]),flush=True)
                if stale>=6:break
            if STOP:
                save_current();atomic_json(a.out/'RUN_STATE.json',dict(status='STOPPED_CHECKPOINTED',step=step));return
        best_state=torch.load(a.out/'BEST.pt',map_location=a.device,weights_only=False);model.load_state_dict(best_state['model']);model.eval()
        ids,pos=bundle.sampler(held,np.random.default_rng(42),a.batch_size,flow=False)
        begin=time.perf_counter()
        if flow:
            batch=bundle.batch(ids,pos,flow=True,device=a.device,with_target=False)
            c=flow_condition(batch['history'],batch['mask'],batch['position'])
            noise=torch.randn(a.batch_size,dimensions,device=a.device)
            endpoint,dot=flow_dot(model,c,noise,50)
            finite=bool(torch.isfinite(endpoint).all() and torch.isfinite(dot).all())
            reconstructed=bundle.condition_from_probabilities(batch,batch['source_logits'])
            reconstruction_max=float((c-reconstructed).abs().max())
        else:
            batch=bundle.batch(ids,pos,flow=False,device=a.device)
            with torch.no_grad():mean,var=model(batch['history'],batch['mask'],batch['position'])
            finite=bool(torch.isfinite(mean).all() and torch.isfinite(var).all());reconstruction_max=None
        if a.device=='cuda':torch.cuda.synchronize()
        elapsed=time.perf_counter()-begin
        if not finite:raise FloatingPointError('nonfinite model inference')
        atomic_json(a.out/'RUN_STATE.json',dict(status='SMOKE_COMPLETE' if a.smoke else 'TRAINED',steps=step,best_step=best_state['step'],
            finite_inference=finite,inference_seconds=elapsed,inference_tokens=a.batch_size,
            probability_condition_reconstruction_max=reconstruction_max,quality_evaluation=False,
            runtime_note='Timing includes feature preparation; extrapolation is a feasibility estimate only.'))
    except BaseException as e:
        save_current();atomic_json(a.out/'RUN_STATE.json',dict(status='FAILED',step=step,error=f'{type(e).__name__}: {e}'));raise


if __name__=='__main__':main()
