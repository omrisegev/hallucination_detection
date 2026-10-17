"""Full eight-arm answer-local binary moment comparison, using the frozen evaluator."""
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
from spectral_utils.binary_moment_fusion import METHODS, fit_all
from spectral_utils.varentropy_contribution_fusion import contributions
from scripts.run_varentropy_contribution_fusion import NAMES as VARENTROPY_NAMES
from spectral_utils.direct_probability_fusion import step_top_mean
from spectral_utils.answer_localization_v2 import STREAM_NAMES

OUT = ROOT / 'results/binary_moment_fusion_v1'
NAMES={f'{b}__{m}': {'var18':'Varentropy contributions + H,V,chosen (18)', 'both33':'Entropy and varentropy contributions + H,V,chosen (33)'}[b]+' / '+{'continuous_equal':'Normalized continuous mean','binary_equal':'Equal median votes','sml':'Signed SML median votes','lsml':'Latent SML median votes'}[m] for b in ('var18','both33') for m in ('continuous_equal','binary_equal','sml','lsml')}
NAMES.update({'ref__'+k:v for k,v in VARENTROPY_NAMES.items()})


def worker(task):
    i, uid, lp, chosen, entropy, spans, expected = task
    fits, failures, seconds = fit_all(lp,chosen,entropy)
    replay=contributions(lp,50).sum(axis=1)
    np.testing.assert_allclose(replay, expected, atol=1e-12, rtol=1e-10,
                               err_msg=f'{uid}: frozen raw Varentropy50 mismatch')
    error = float(np.max(np.abs(replay-expected)))
    S=np.full((len(spans),len(METHODS)),np.nan);W=np.full((len(METHODS),33),np.nan)
    diag={}
    for j,m in enumerate(METHODS):
        if m not in fits:continue
        f=fits[m];S[:,j]=step_top_mean(f['score'],spans[:,0],spans[:,1],count=10)
        k=len(f['weights']);W[j,:k]=f['weights']
        diag[m]=f['diagnostics']
    return i,base.packed(steps=S,weights=W),base.dumps(dict(
        uid=uid,n_tokens=len(lp),failures=failures,seconds=seconds,diagnostics=diag,raw50_max_error=error))


def manifest_for(source,v2,temporal,varentropy):
    manifest=base.input_manifest(source,v2)
    manifest.update(schema='binary-moment-v1',methods=list(METHODS))
    files=[Path(__file__),ROOT/'spectral_utils/binary_moment_fusion.py',ROOT/'spectral_utils/varentropy_contribution_fusion.py', ROOT/'spectral_utils/fusion_utils.py',
           ROOT/'scripts/run_varentropy_contribution_fusion.py',
           ROOT/'docs/experiments/BINARY_MOMENT_FUSION_V1.md',
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
            indices=[i for i,r in enumerate(records) if r['cell']==cell and i not in done]
            if not indices:continue
            print('[load]',cell,len(indices),flush=True)
            rows=base.old._source_row_map(base.old.load_pickle(path),kind=kind,dataset=dataset)
            d=base.old.BENCH/'inputs'/cell
            raw=np.load(d/'raw.npy',mmap_mode='r');to=np.load(d/'token_offsets.npy')
            lookup={str(v):j for j,v in enumerate(np.load(d/'row_ids.npy',allow_pickle=True))}
            if smoke:
                order=sorted(indices,key=lambda i:len(rows[records[i]['row_id']]['token_entropies']))
                indices=[order[j] for j in sorted({0,len(order)//2,min(len(order)-1,int(.95*len(order)))})]
            for start in range(0,len(indices),32):
                batch=[]
                for i in indices[start:start+32]:
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
                    batch.append((i,r['uid'],lp,row['token_spilled_energies'],entropy,spans,expected))
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
    primary={('both33__sml','both33__binary_equal'),('both33__lsml','both33__sml')}
    pairs=sorted(primary)+[('var18__sml','var18__binary_equal'),('var18__lsml','var18__sml')]
    pairs += [('both33__'+m,'var18__'+m) for m in ('continuous_equal','binary_equal','sml','lsml')]
    pairs += [(b+'__binary_equal',b+'__continuous_equal') for b in ('var18','both33')]
    pairs += [(m,'ref__'+ref) for m in METHODS for ref in ('k15__raw','k15__equal','k50__raw')]
    print('[evaluate] all nine frozen references reproduced; bootstrap10000',flush=True)
    contrasts=base.paired_bootstrap(records,joined,per,pairs=pairs,primary_pairs=primary,draws=10000)
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
        W=np.stack(t.pop('weights')) if t['weights'] else np.zeros((0,33));k=18 if m.startswith('var18') else 33;W=W[:,:k]
        if m.endswith('__lsml'):
            weights[m]=dict(n=len(W),note='Nonlinear two-level votes; per-answer within/cross weights are stored in checkpoint diagnostics.')
            t.pop('alphas',None)
            continue
        total=np.abs(W).sum(axis=1,keepdims=True)
        share=np.divide(W,total,out=np.zeros_like(W),where=total>0)
        weights[m]=dict(n=len(W),mean_standardized_coefficients=W.mean(axis=0).tolist() if len(W) else None,
            mean_absolute_share=np.abs(share).mean(axis=0).tolist() if len(W) else None,
            mean_negative_share=np.maximum(-share,0).sum(axis=1).mean().item() if len(W) else None,
            note='Continuous weights act on standardized columns; binary weights act on oriented votes. L-SML is nonlinear: see per-answer within/cross weights, not this NaN vector.')
        t.pop('alphas',None)
    payload=dict(schema='binary-moment-fusion-v1',n_answers=len(records),n_steps=int(joined['offsets'][-1]),
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


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for a in ('source-root','v2-root','temporal-root','varentropy-root'):p.add_argument('--'+a,type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--smoke',action='store_true');p.add_argument('--evaluate-only',action='store_true')
    args=p.parse_args();base.old.configure_source_root(args.source_root.resolve());OUT.mkdir(parents=True,exist_ok=True)
    manifest=manifest_for(args.source_root.resolve(),args.v2_root.resolve(),args.temporal_root.resolve(),args.varentropy_root.resolve())
    con=base.connect(OUT/('SMOKE.sqlite' if args.smoke else 'CHECKPOINT.sqlite'),manifest)
    base.atomic_json(OUT/('SMOKE_MANIFEST.json' if args.smoke else 'MANIFEST.json'),manifest)
    records=json.loads((base.old.BENCH/'evaluation/JOINED.json').read_text(encoding='utf8'))['records']
    joined=np.load(base.old.BENCH/'evaluation/JOINED.npz')
    if len(records)!=13769 or len({r['uid'] for r in records})!=13769:raise ValueError('benchmark roster mismatch')
    with threadpool_limits(limits=1):
        if not args.evaluate_only:score(con,records,args.workers,args.smoke)
        if args.smoke:
            info=[json.loads(r[0]) for r in con.execute('SELECT info FROM answers')]
            base.atomic_json(OUT/'SMOKE.json',dict(scope='FEASIBILITY_ONLY',rows=info))
            print('[smoke]',len(info),'rows; no benchmark ranking; failures:',sum(len(r['failures']) for r in info),flush=True)
        else:evaluate(con,records,joined,args.temporal_root,args.v2_root,args.varentropy_root)
    con.close()


if __name__=='__main__':
    try:main()
    except BaseException as error:
        name='SMOKE_STATE.json' if '--smoke' in sys.argv else 'RUN_STATE.json'
        path=OUT/name;state=json.loads(path.read_text()) if path.exists() else {}
        state.update(status='FAILED',error=f'{type(error).__name__}: {error}');base.atomic_json(path,state)
        raise
