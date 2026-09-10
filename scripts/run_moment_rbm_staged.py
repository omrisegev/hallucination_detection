"""Continuous moment fusion with equal, IU-PCR, exact GRBM and B3, using the frozen evaluator."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import io
import json
from pathlib import Path
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_direct_probability_temporal as base
from spectral_utils.moment_rbm_fusion import METHODS as ALL_METHODS, fit_all
from spectral_utils import moment_rbm_fusion as core
METHODS=ALL_METHODS
from spectral_utils.varentropy_contribution_fusion import contributions
from scripts.run_varentropy_contribution_fusion import NAMES as VARENTROPY_NAMES
from spectral_utils.direct_probability_fusion import step_top_mean
from spectral_utils.answer_localization_v2 import STREAM_NAMES

PARENT_OUT=ROOT/'results/moment_rbm_fusion_v1'
STAGED_OUT=ROOT/'results/moment_rbm_fusion_v1_staged'
OUT=STAGED_OUT
FAST=('equal','iu','rbm_initial','rbm')
SLOW=('b3_initial','b3')
NAMES={'equal':'Moment mean','iu':'Moment IU-PCR','rbm_initial':'Gaussian RBM before learning','rbm':'Gaussian RBM exact likelihood','b3_initial':'B3 before learning','b3':'B3 continuous moment fusion'}
NAMES.update({'ref__'+k:v for k,v in VARENTROPY_NAMES.items()})


def worker(task):
    global METHODS
    i, uid, lp, chosen, entropy, spans, expected, active = task
    METHODS=tuple(active);core.METHODS=METHODS
    fits, failures, seconds = fit_all(lp,chosen,uid)
    replay=contributions(lp,50).sum(axis=1)
    np.testing.assert_allclose(replay, expected, atol=1e-12, rtol=1e-10,
                               err_msg=f'{uid}: frozen raw Varentropy50 mismatch')
    error = float(np.max(np.abs(replay-expected)))
    S=np.full((len(spans),len(METHODS)),np.nan);W=np.full((len(METHODS),6),np.nan)
    diag={};states={}
    for j,m in enumerate(METHODS):
        if m not in fits:continue
        f=fits[m];W[j]=f['weights'];S[:,j]=step_top_mean(f['score'],spans[:,0],spans[:,1],count=10)
        for key,value in f['state'].items():states[m+'::'+key]=value
        diag[m]=f['diagnostics']
    return i,base.packed(steps=S,weights=W,**states),base.dumps(dict(
        uid=uid,n_tokens=len(lp),failures=failures,seconds=seconds,diagnostics=diag,raw50_max_error=error))


def manifest_for(source,v2,temporal,varentropy):
    manifest=base.input_manifest(source,v2)
    from importlib import metadata
    manifest['packages']={n:metadata.version(n) for n in ('torch','numpy','scipy')}
    manifest.update(schema='moment-rbm-fusion-v1-staged',methods=list(METHODS))
    files=[Path(__file__),ROOT/'scripts/run_moment_rbm_fusion.py',ROOT/'spectral_utils/moment_rbm_fusion.py',ROOT/'spectral_utils/varentropy_contribution_fusion.py', ROOT/'spectral_utils/deem_adapter.py',ROOT/'spectral_utils/deem_b3_contract_ablation.py',ROOT/'spectral_utils/residual_graph_deem.py',
           ROOT/'scripts/run_varentropy_contribution_fusion.py',
           ROOT/'docs/experiments/MOMENT_RBM_FUSION_V1.md',
           temporal/'results/direct_probability_temporal_v3/SCORES.npz',
           temporal/'results/direct_probability_temporal_v3/METRICS.json',
           varentropy/'results/varentropy_contribution_fusion_v1/SCORES.npz',
           varentropy/'results/varentropy_contribution_fusion_v1/METRICS.json']
    for cell,_,_,_ in base.source_specs():
        files += [base.old.BENCH/'inputs'/cell/(name+'.npy') for name in ('raw','token_offsets','row_ids')]
    for path in files:manifest['hashes'][str(path)]=base.old.sha256_file(path)
    return manifest


def score(con,records,workers,smoke):
    done={r[0] for r in con.execute('SELECT idx FROM answers')};started=time.perf_counter()
    detector,_=base.old._gate_contract(records)
    with ProcessPoolExecutor(max_workers=workers,initializer=base.worker_init) as pool:
        for cell,path,kind,dataset in base.source_specs():
            all_indices=[i for i,r in enumerate(records) if r['cell']==cell]
            indices=[i for i in all_indices if i not in done]
            if not indices:continue
            print('[load]',cell,len(indices),flush=True)
            rows=base.old._source_row_map(base.old.load_pickle(path),kind=kind,dataset=dataset)
            d=base.old.BENCH/'inputs'/cell
            raw=np.load(d/'raw.npy',mmap_mode='r');to=np.load(d/'token_offsets.npy')
            lookup={str(v):j for j,v in enumerate(np.load(d/'row_ids.npy',allow_pickle=True))}
            if smoke:
                order=sorted(all_indices,key=lambda i:len(rows[records[i]['row_id']]['token_entropies']))
                indices=[order[j] for j in sorted({0,len(order)//2,min(len(order)-1,int(.95*len(order)))}) if order[j] not in done]
            for start in range(0,len(indices),4):
                batch=[]
                for i in indices[start:start+4]:
                    r=records[i];row=rows[r['row_id']]
                    lp=np.asarray(base.old._topk_payload(row)['logprobs'],float)
                    entropy=np.asarray(row['token_entropies'],float);spans=np.asarray(row['step_token_spans'],int)
                    if lp.shape!=(len(entropy),50):raise ValueError(f'{r["uid"]}: expected T x 50')
                    if spans.shape!=(r['steps'],2):raise ValueError('step count mismatch')
                    with np.load(base.old.BENCH/'scores'/f'{r["uid"]}.npz') as z:
                        np.testing.assert_array_equal(spans[:,0],z['step_starts'])
                        np.testing.assert_array_equal(spans[:,1],z['step_ends'])
                    if kind=='pb':np.testing.assert_allclose(entropy.mean(),detector[i],atol=1e-12,rtol=0)
                    j=lookup[r['row_id']];expected=np.asarray(raw[to[j]:to[j+1],STREAM_NAMES.index('topk_varentropy_series')],float)
                    if len(expected)!=len(lp):raise ValueError('frozen token count mismatch')
                    batch.append((i,r['uid'],lp,row['token_spilled_energies'],entropy,spans,expected,METHODS))
                for item in pool.map(worker,batch,chunksize=1):con.execute('INSERT INTO answers VALUES (?,?,?)',item)
                con.commit();done.update(item[0] for item in batch)
                state=dict(status='SMOKE' if smoke else 'RUNNING',completed=len(done),expected=len(records),
                    elapsed_seconds=time.perf_counter()-started)
                base.atomic_json(OUT/('SMOKE_STATE.json' if smoke else 'RUN_STATE.json'),state)
                print('[checkpoint]',len(done),'/',len(records),round(state['elapsed_seconds'],1),'seconds',flush=True)
            del rows,raw


def evaluate(con,records,joined,temporal,v2,varentropy):
    base.METHODS=METHODS
    scores,telemetry=base.load_scored(con,records,joined['offsets'])
    previous=json.loads((varentropy/'results/varentropy_contribution_fusion_v1/METRICS.json').read_text(encoding='utf8'))
    with np.load(varentropy/'results/varentropy_contribution_fusion_v1/SCORES.npz') as z:
        for m in previous['metrics']:scores['ref__'+m]=z['steps__'+m]
    metrics,per=base.evaluate_arrays(records,joined,scores)
    for name,x in previous['metrics'].items():
        for key in ('pb_all8','pb_q4','pb_q8','prm_within','prm_pooled','prmscore_q08'):
            np.testing.assert_allclose(metrics['ref__'+name][key],x[key],atol=1e-12,rtol=0,err_msg=name+' '+key)
    oldmetrics=json.loads((temporal/'results/direct_probability_temporal_v3/METRICS.json').read_text(encoding='utf8'))
    primary={('rbm','iu'),('b3','rbm')}
    pairs=sorted(primary)+[('iu','equal'),('rbm','equal'),('b3','equal'),('rbm','rbm_initial'),('b3','b3_initial')]
    pairs += [(m,'ref__'+ref) for m in METHODS for ref in ('k15__raw','k15__equal','k50__raw')]
    pairs=[(a,b) for a,b in pairs if a in scores and b in scores]
    primary={p for p in primary if p in pairs}
    print('[evaluate] all nine frozen references reproduced; bootstrap10000',flush=True)
    contrasts=base.paired_bootstrap(records,joined,per,pairs=pairs,primary_pairs=primary,draws=10000,primary_ci=.975)
    pb=np.array([r['cell'].startswith('pb_') for r in records]);target=joined['target'];cases={}
    for a,b in pairs:
        c=contrasts[a+'_minus_'+b];c['pb_delta']=metrics[a]['pb_all8']-metrics[b]['pb_all8']
        oldhit=pb & per[b]['decision_valid'] & (per[b]['prediction']==target)
        newhit=pb & per[a]['decision_valid'] & (per[a]['prediction']==target)
        c.update(gained=int((newhit & ~oldhit).sum()),lost=int((oldhit & ~newhit).sum()))
        cases[a+'_minus_'+b]=[dict(uid=records[i]['uid'],cell=records[i]['cell'],target=int(target[i]),
            before=int(per[b]['prediction'][i]),after=int(per[a]['prediction'][i]),
            change='gained' if newhit[i] else 'lost') for i in np.flatnonzero(oldhit ^ newhit)]
    weights={}
    for m,t in telemetry.items():
        saved=t.pop('weights',[]);t.pop('alphas',None)
        weights[m]=dict(note='Per-answer standardized weights and model states in checkpoint. B3 has no single fixed weight vector.')
    payload=dict(schema='moment-rbm-fusion-v1',n_answers=len(records),n_steps=int(joined['offsets'][-1]),
        scope='Full cached localization development; fusion answer-local, gate/calibration external; no historical24 result.',
        metrics=metrics,contrasts=contrasts,telemetry=telemetry,weights=weights,
        historical_references=oldmetrics['historical_references'],mind_gap_reference=oldmetrics['mind_gap_reference'])
    base.atomic_json(OUT/'METRICS.json',payload);base.atomic_json(OUT/'ERROR_CASES.json',cases)
    np.savez_compressed(OUT/'SCORES.npz',**{'steps__'+m:s for m,s in scores.items()},
        **{'prediction__'+m:p['prediction'] for m,p in per.items()},**{'valid__'+m:p['valid'] for m,p in per.items()})
    rows=[]
    for m,x in metrics.items():
        row=dict(method=NAMES[m],method_id=m,PB_macro_percent=100*x['pb_all8'],PB_Q4_percent=100*x['pb_q4'],
            PB_Q8_percent=100*x['pb_q8'],PRMB_within_AUC=x['prm_within'],PRMB_pooled_AUC=x['prm_pooled'],
            PRMScore=x['prmscore_q08'],valid_answers=x['valid_answers'],within_answers=x['prm_within_n'])
        row.update({cell:100*v['f1'] for cell,v in x['pb_cells'].items()});rows.append(row)
        print(m,x['pb_all8'],x['prm_within'],x['prm_pooled'],x['prmscore_q08'],flush=True)
    with (OUT/'SUMMARY.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    base.atomic_json(OUT/'RUN_STATE.json',dict(status='COMPLETE',completed=len(records),expected=len(records)))


def subset_row(blob,info,selected):
    """Project a committed original row without refitting or changing any value."""
    with np.load(io.BytesIO(blob),allow_pickle=False) as z:
        indices=[ALL_METHODS.index(m) for m in selected]
        states={k:z[k].copy() for k in z.files if '::' in k and k.split('::')[0] in selected}
        payload=base.packed(steps=z['steps'][:,indices],weights=z['weights'][indices],**states)
    data=json.loads(info)
    for key in ('failures','seconds','diagnostics'):
        data[key]={m:v for m,v in data[key].items() if m in selected}
    return payload,base.dumps(data)


def seed_original(con,records):
    import sqlite3
    path=PARENT_OUT/'CHECKPOINT.sqlite'
    if not path.exists():return
    original=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)
    assert original.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    if json.loads(original.execute('SELECT payload FROM manifest WHERE id=1').fetchone()[0])!=json.loads((PARENT_OUT/'MANIFEST.json').read_text()):
        raise ValueError('Original checkpoint manifest mismatch')
    done={x[0] for x in con.execute('SELECT idx FROM answers')};added=0
    for i,blob,info in original.execute('SELECT idx,payload,info FROM answers'):
        if i in done:continue
        if json.loads(info)['uid']!=records[i]['uid']:raise ValueError('original checkpoint ID drift')
        payload,projected=subset_row(blob,info,METHODS)
        con.execute('INSERT INTO answers VALUES (?,?,?)',(i,payload,projected));added+=1
    con.commit();original.close();print('[reuse original]',len(METHODS),'methods',added,'rows',flush=True)


def merge_rows(fast_blob,fast_info,slow_blob,slow_info):
    a=json.loads(fast_info);b=json.loads(slow_info)
    if a['uid']!=b['uid']:raise ValueError('phase ID mismatch')
    if a['n_tokens']!=b['n_tokens']:raise ValueError('phase token count mismatch')
    with np.load(io.BytesIO(fast_blob)) as f,np.load(io.BytesIO(slow_blob)) as s:
        states={k:z[k].copy() for z in (f,s) for k in z.files if '::' in k}
        payload=base.packed(steps=np.column_stack((f['steps'],s['steps'])),
            weights=np.vstack((f['weights'],s['weights'])),**states)
    for key in ('failures','seconds','diagnostics'):a[key].update(b[key])
    return payload,base.dumps(a)


def verify(args):
    import subprocess
    subprocess.run([sys.executable,str(ROOT/'scripts/verify_direct_probability_temporal_results.py'),
        '--source-root',str(args.source_root),'--result-dir',str(OUT)],check=True)
    state=json.loads((OUT/'RUN_STATE.json').read_text());state['review']='PASS'
    base.atomic_json(OUT/'RUN_STATE.json',state)


def main():
    global OUT,METHODS
    import sqlite3
    import ctypes
    p=argparse.ArgumentParser(description='Fast full benchmark first; B3 second; preserve frozen algorithms.')
    for name in ('source-root','v2-root','temporal-root','varentropy-root'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args();STAGED_OUT.mkdir(parents=True,exist_ok=True)
    # Refuse concurrent execution with the original benchmark. Never stop it here.
    launch=json.loads((PARENT_OUT/'LAUNCH.json').read_text(encoding='utf-8-sig'))
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.restype=ctypes.c_void_p
    kernel.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)]
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    handle=kernel.OpenProcess(0x1000,False,int(launch['pid']))
    if handle:
        code=ctypes.c_ulong();ok=kernel.GetExitCodeProcess(handle,ctypes.byref(code));kernel.CloseHandle(handle)
        if not ok or code.value==259:raise RuntimeError('Original benchmark still active; no phase started')
    elif ctypes.get_last_error()!=87:
        raise RuntimeError('Cannot verify original process stopped; no phase started')
    base.old.configure_source_root(args.source_root.resolve())
    # Verify every frozen input/code hash before reusing the original checkpoint.
    frozen=json.loads((PARENT_OUT/'MANIFEST.json').read_text())
    for name,expected in frozen['hashes'].items():
        if base.old.sha256_file(Path(name))!=expected:raise ValueError('Original source drift: '+name)
    records=json.loads((base.old.BENCH/'evaluation/JOINED.json').read_text(encoding='utf8'))['records']
    joined=np.load(base.old.BENCH/'evaluation/JOINED.npz')
    if len(records)!=13769 or len({r['uid'] for r in records})!=13769:raise ValueError('roster mismatch')
    connections={}
    with threadpool_limits(limits=1):
        for phase,active in (('fast',FAST),('b3',SLOW),('combined',ALL_METHODS)):
            OUT=STAGED_OUT/phase;OUT.mkdir(parents=True,exist_ok=True);METHODS=active
            manifest=manifest_for(args.source_root.resolve(),args.v2_root.resolve(),args.temporal_root.resolve(),args.varentropy_root.resolve())
            manifest['phase']=phase
            manifest['original_manifest_sha256']=base.old.sha256_file(PARENT_OUT/'MANIFEST.json')
            con=base.connect(OUT/'CHECKPOINT.sqlite',manifest);connections[phase]=con
            base.atomic_json(OUT/'MANIFEST.json',manifest)
            base.atomic_json(STAGED_OUT/'RUN_STATE.json',dict(status='RUNNING',phase=phase))
            if phase!='combined':
                seed_original(con,records);score(con,records,args.workers,False)
            else:
                done={r[0] for r in con.execute('SELECT idx FROM answers')}
                for i,blob,info in connections['fast'].execute('SELECT idx,payload,info FROM answers'):
                    if i in done:continue
                    row=connections['b3'].execute('SELECT payload,info FROM answers WHERE idx=?',(i,)).fetchone()
                    if row is None:raise ValueError('B3 phase incomplete')
                    payload,combined=merge_rows(blob,info,*row)
                    con.execute('INSERT INTO answers VALUES (?,?,?)',(i,payload,combined))
                con.commit()
            # B3 comparison against RBM is made after the combined phase.
            evaluate(con,records,joined,args.temporal_root,args.v2_root,args.varentropy_root);verify(args)
            print('[phase reviewed]',phase,flush=True)
        for con in connections.values():con.close()
    base.atomic_json(STAGED_OUT/'RUN_STATE.json',dict(status='COMPLETE',phase='combined',review='PASS'))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        STAGED_OUT.mkdir(parents=True,exist_ok=True)
        base.atomic_json(STAGED_OUT/'RUN_STATE.json',dict(status='FAILED',error=f'{type(error).__name__}: {error}'))
        raise

