"""Resumable held-source scoring for a trained telemetry TCN, FM or DiFlo model."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import signal
import sqlite3
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from spectral_utils.context_training import FeatureBundle
from spectral_utils.context_readout import prediction_readouts
from spectral_utils.direct_probability_fusion import step_top_mean
from spectral_utils.temporal_context_models import ConditionalFlow,TelemetryTCN,flow_condition,flow_dot,residual_step_score
from spectral_utils.temporal_research_features import BASELINE
from scripts.train_temporal_context_model import atomic_json

STOP=False
def stop(*_):
    global STOP
    STOP=True


def readouts(bundle,i,model,method,seed,bankbase,device,batch_size=512):
    m=bundle.metadata[i];n=m['tokens'];start=m['offset'];d=len(bundle.columns)
    rng=np.random.default_rng(int.from_bytes(hashlib.sha256(('neural-context/'+m['uid']+'/'+str(seed)).encode()).digest()[:8],'little'))
    noise=torch.tensor(rng.normal(size=(n,d)),dtype=torch.float32,device=device)
    means={k:[] for k in ('real','shuffled','zero')};variances={k:[] for k in means};dots={k:[] for k in means}
    with torch.no_grad():
        for first in range(0,n,batch_size):
            positions=np.arange(first,min(n,first+batch_size));ids=np.full(len(positions),i)
            batch=bundle.batch(ids,positions,flow=method!='tcn',device=device,with_target=False)
            history=batch['history'];shuffled=history.clone();zero=history.clone()
            past=history.shape[1]-int(method!='tcn')
            zero[:,:past]=0
            for row in range(len(positions)):
                visible=torch.nonzero(batch['mask'][row,:past],as_tuple=False).flatten()
                order=torch.tensor(rng.permutation(len(visible)),device=device)
                shuffled[row,visible]=history[row,visible[order]]
            for name,values in [('real',history),('shuffled',shuffled),('zero',zero)]:
                if method=='tcn':
                    mu,var=model(values,batch['mask'],batch['position'])
                    means[name].append(mu.cpu().numpy());variances[name].append(var.cpu().numpy())
                else:
                    condition=flow_condition(values,batch['mask'],batch['position'])
                    endpoint,dot=flow_dot(model,condition,noise[first:first+len(positions)],steps=50)
                    means[name].append(endpoint.cpu().numpy());dots[name].append(dot.cpu().numpy())
    raw=np.asarray(bundle.features[start:start+n])[:,bundle.columns].astype(float)
    z=(raw-bundle.mean[i])/bundle.scale[i]
    spans=np.asarray(bundle.spans[m['step_start']:m['step_stop']])-start
    output={};tokens={}
    for name in means:
        mu=np.concatenate(means[name])
        if not np.isfinite(mu).all():raise FloatingPointError('nonfinite model prediction')
        if method=='tcn':
            var=np.concatenate(variances[name])
            output.update({name+'__'+k:v for k,v in prediction_readouts(raw,z,mu,spans,bankbase,var).items()})
            tokens[name+'__mean']=mu.astype(np.float32);tokens[name+'__variance']=var.astype(np.float32)
            if name=='real':
                static_var=np.broadcast_to(var.mean(0),var.shape)
                output['static_variance_fusion']=prediction_readouts(raw,z,mu,spans,bankbase,static_var)['variance_fusion']
        else:
            dot=np.concatenate(dots[name]);tokens[name+'__dot']=dot.astype(np.float32)
            auxiliary=step_top_mean(dot,spans[:,0],spans[:,1],10);output[name+'__dot']=auxiliary
            for gamma in (.25,1.):output[name+f'__dot_residual_{gamma:g}']=residual_step_score(bankbase,auxiliary,gamma)
            # Explicit observed-innovation control: next-token errors are scored at
            # that observed next token. First token has no previous prediction.
            error=np.zeros(n);error[1:]=np.square(z[1:]-mu[:-1]).mean(1)
            tokens[name+'__observed_error']=error.astype(np.float32)
            auxiliary=step_top_mean(error,spans[:,0],spans[:,1],10)
            output[name+'__observed_error']=auxiliary
            for gamma in (.25,1.):output[name+f'__error_residual_{gamma:g}']=residual_step_score(bankbase,auxiliary,gamma)
    if any(not np.isfinite(v).all() for v in output.values()):raise FloatingPointError('nonfinite context readout')
    return output,tokens


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--allow-smoke',action='store_true',help='Correctness/cost check only; cannot be evaluated as quality.')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    manifest=json.loads((a.model/'MANIFEST.json').read_text());state=json.loads((a.model/'RUN_STATE.json').read_text())
    if state['status'] not in ('TRAINED','SMOKE_COMPLETE'):raise ValueError('training has not completed')
    if manifest['smoke'] and not a.allow_smoke:raise ValueError('smoke model cannot produce a quality run')
    bundle=FeatureBundle(a.data,manifest['bank']);d=len(bundle.columns);method=manifest['method']
    if hashlib.sha256((a.data/'MANIFEST.json').read_bytes()).hexdigest()!=manifest['data_manifest_sha256']:raise ValueError('training/scoring data drift')
    train,validation,held=bundle.split(manifest['excluded_folds'])
    for name,ids in [('training',train),('validation',validation),('held',held)]:
        if sorted({bundle.metadata[i]['group_id'] for i in ids})!=manifest[name+'_groups']:raise ValueError('source-group split drift')
    if len(manifest['excluded_folds'])==2:held=[i for i in held if not bundle.metadata[i]['cell'].startswith('pb_')]
    if a.allow_smoke:held=held[:3]
    model=(TelemetryTCN(d) if method=='tcn' else ConditionalFlow(17*d+18,d)).to(a.device)
    best=torch.load(a.model/'BEST.pt',map_location=a.device,weights_only=False);model.load_state_dict(best['model']);model.eval()
    with np.load(a.baseline/'SCORES_FROZEN.npz',allow_pickle=False) as f:
        baseline=f['steps__'+(BASELINE if manifest['bank']=='original4' else 'append_innovation__H0lim')]
    own=dict(schema='temporal-context-scoring-v1',model_manifest=manifest,checkpoint_sha256=hashlib.sha256((a.model/'BEST.pt').read_bytes()).hexdigest(),
             expected_answers=len(held),smoke=a.allow_smoke,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             controls='Inference history-slot permutation and zero-past controls; not refitted null models.',
             flow_error_control='Observed innovation at token t uses endpoint generated at t-1; first token zero/missing history.')
    if (a.out/'MANIFEST.json').exists() and json.loads((a.out/'MANIFEST.json').read_text())!=own:raise ValueError('immutable scoring manifest changed')
    atomic_json(a.out/'MANIFEST.json',own);signal.signal(signal.SIGTERM,stop)
    db=sqlite3.connect(a.out/'SCORES.sqlite');db.execute('CREATE TABLE IF NOT EXISTS answers (idx INTEGER PRIMARY KEY, steps BLOB, tokens BLOB)');db.commit()
    done={r[0] for r in db.execute('SELECT idx FROM answers')};begin=time.perf_counter()
    for i in held:
        if i in done:continue
        m=bundle.metadata[i];b=baseline[m['step_start']:m['step_stop']]
        step,token=readouts(bundle,i,model,method,manifest['seed'],b,a.device)
        sb=io.BytesIO();tb=io.BytesIO();np.savez_compressed(sb,**step)
        if len(manifest['excluded_folds'])==1:np.savez_compressed(tb,**token)
        db.execute('INSERT INTO answers VALUES (?,?,?)',(i,sb.getvalue(),tb.getvalue()));db.commit();done.add(i)
        if len(done)%100==0:
            atomic_json(a.out/'RUN_STATE.json',dict(status='SCORING',answers=len(done),expected=len(held),seconds=time.perf_counter()-begin))
            print('[score]',len(done),len(held),flush=True)
        if STOP:
            atomic_json(a.out/'RUN_STATE.json',dict(status='STOPPED_CHECKPOINTED',answers=len(done)));db.close();return
    total=int(bundle.manifest['steps']);scores={}
    for i,blob in db.execute('SELECT idx,steps FROM answers ORDER BY idx'):
        m=bundle.metadata[i]
        with np.load(io.BytesIO(blob),allow_pickle=False) as f:
            for name in f.files:
                if name not in scores:scores[name]=np.full(total,np.nan)
                scores[name][m['step_start']:m['step_stop']]=f[name]
    db.close();np.savez_compressed(a.out/'STEP_SCORES.npz',**scores)
    atomic_json(a.out/'RUN_STATE.json',dict(status='SMOKE_COMPLETE' if a.allow_smoke else 'SCORED',answers=len(done),expected=len(held),seconds=time.perf_counter()-begin))


if __name__=='__main__':main()
