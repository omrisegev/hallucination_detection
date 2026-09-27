"""Independent external metric replay. Execute only after explicit parent GO.

Imports no project scoring/evaluation helpers and never reads metric summaries.
An all-cell prediction seal is verified before any evaluator annotation is opened.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse, csv, hashlib, json, sys, time
from pathlib import Path
import numpy as np

CELLS={'hard2verify_qwen3_8b':'hard2verify',
       'socratic_qwen3_8b':'socratic','socratic_qwq32b':'socratic'}
ARMS=('frozen_lsml','frozen_equal','frozen_partition_equal',
      'local_lsml','local_equal','local_partition_equal','ct7')

def load(path):
    return json.loads(Path(path).read_text(encoding='utf8'))

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def canonical_sha(value):
    raw=json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()

def confusion(correct,predicted,include):
    y=np.asarray(correct,bool); p=np.asarray(predicted,bool); keep=np.asarray(include,bool)
    return np.array([np.count_nonzero(keep&y&p),np.count_nonzero(keep&~y&p),
                     np.count_nonzero(keep&~y&~p),np.count_nonzero(keep&y&~p)],dtype=np.int64)

def metrics(count,benchmark):
    tp,fp,tn,fn=map(int,count)
    rgood=tp/(tp+fn) if tp+fn else None
    rbad=tn/(tn+fp) if tn+fp else None
    fgood=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
    fbad=2*tn/(2*tn+fp+fn) if 2*tn+fp+fn else 0.
    if benchmark=='hard2verify':
        primary=(2*rgood*rbad/(rgood+rbad) if rgood+rbad else 0.) if rgood is not None and rbad is not None else None
    else:
        primary=(fgood+fbad)/2
    return {'metric':primary,'correct_recall':rgood,'error_recall':rbad,
            'valid_F1':fgood,'error_F1':fbad,
            'gold_error_fraction':(tn+fp)/(tp+fp+tn+fn),
            'predicted_error_fraction':(tn+fn)/(tp+fp+tn+fn)}

def pairwise_auc(correct,scores,valid):
    correct=np.asarray(correct,bool); s=np.asarray(scores,float); keep=np.asarray(valid,bool)
    bad=s[keep&~correct]; good=s[keep&correct]
    if not len(bad) or not len(good): return None
    comparisons=bad[:,None]-good[None,:]
    return float(((comparisons>0)+.5*(comparisons==0)).mean())

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--go',action='store_true',help='Only pass after explicit parent authorization')
    parser.add_argument('--evaluation',required=True,type=Path)
    parser.add_argument('--inputs',required=True,type=Path)
    args=parser.parse_args()
    if not args.go: raise RuntimeError('Explicit GO is required; no artifacts or labels opened')
    started=time.perf_counter(); root=args.evaluation.resolve(); inputs=args.inputs.resolve()
    out=Path(__file__).resolve().parent
    if (out/'INDEPENDENT_EXTERNAL.json').exists(): raise FileExistsError('Preserve completed independent audit')
    all_seal_path=root/'ALL_CELLS_SEALED.json'
    if not all_seal_path.is_file(): raise RuntimeError('ALL_CELLS_SEALED is required before opening labels')
    seal_all=load(all_seal_path); bundle_path=root/'source/BUNDLE.json'; bundle_hash=sha(bundle_path)
    if seal_all['bundle_sha256']!=bundle_hash or set(seal_all['seals'])!=set(CELLS):
        raise ValueError('Incomplete or mismatched all-cell seal')
    artifacts={str(all_seal_path):sha(all_seal_path),str(bundle_path):bundle_hash}
    # Verify all predictions and unlabeled roster alignment BEFORE opening any gold.
    predictions={}; answers={}
    for benchmark in sorted(set(CELLS.values())):
        path=inputs/benchmark/'answers.json'; rows=load(path)
        if len({row['uid'] for row in rows})!=len(rows): raise ValueError('Duplicate input UID')
        answers[benchmark]=rows;artifacts[str(path)]=sha(path)
    total_answers=total_steps=0
    for cell,benchmark in CELLS.items():
        path=root/cell/'PREDICTIONS.json'; sp=root/cell/'SEAL.json'
        rows=load(path); seal=load(sp)
        if seal!=seal_all['seals'][cell] or seal['lock_sha256']!=bundle_hash:
            raise ValueError('Cell seal differs from all-cell seal')
        if canonical_sha(rows)!=seal['prediction_sha256']:
            raise ValueError('Prediction payload changed after sealing')
        expected={row['uid']:row for row in answers[benchmark]}
        if set(rows)!=set(expected) or seal['answers']!=len(rows) or set(seal['arms'])!=set(ARMS):
            raise ValueError('Population or arm accounting mismatch')
        for uid,row in rows.items():
            steps=len(expected[uid]['steps'])
            if set(row['scores'])!=set(ARMS) or set(row['predictions'])!=set(ARMS):
                raise ValueError('Wrong method roster')
            if len(row['nonempty'])!=steps: raise ValueError('Nonempty mask misalignment')
            nonempty=np.asarray(row['nonempty'],bool)
            for arm in ARMS:
                pred=row['predictions'][arm]; score=row['scores'][arm]
                if len(pred)!=steps or len(score)!=steps or not set(pred)<={0,1}:
                    raise ValueError('Invalid predictions or step alignment')
                values=np.array([np.nan if x is None else x for x in score],float)
                if not np.isfinite(values[nonempty]).all() or np.isfinite(values[~nonempty]).any():
                    raise ValueError('Invalid score missingness')
                if np.asarray(pred)[~nonempty].any(): raise ValueError('Empty step is not conservatively invalid')
        total_answers+=len(rows);total_steps+=sum(len(v['nonempty']) for v in rows.values())
        predictions[cell]=rows;artifacts[str(path)]=sha(path);artifacts[str(sp)]=sha(sp)
    assert total_answers==6190 and total_steps==53970,(total_answers,total_steps)
    # Label access begins only below this line, after every cell passed its seal.
    gold={}
    for benchmark in sorted(set(CELLS.values())):
        path=inputs/'evaluator_only'/(benchmark+'.json'); rows=load(path)
        if len({row['uid'] for row in rows})!=len(rows): raise ValueError('Duplicate annotation UID')
        gold[benchmark]={row['uid']:row for row in rows};artifacts[str(path)]=sha(path)
        assert set(gold[benchmark])=={a['uid'] for a in answers[benchmark]}
    result={};table=[];archive={}
    for cell,benchmark in CELLS.items():
        rows=predictions[cell]; uids=[a['uid'] for a in answers[benchmark]]
        counts={arm:[] for arm in ARMS}; native_counts={arm:[] for arm in ARMS}
        aucs={arm:[] for arm in ARMS};native=[]; included_steps=empty_steps=0
        for uid in uids:
            row=rows[uid]; annotation=gold[benchmark][uid]; correct=annotation['correct'];include=annotation['include']
            assert len(correct)==len(include)==len(row['nonempty'])
            assert set(correct)<={False,True} and set(include)<={False,True}
            nonempty=np.asarray(row['nonempty'],bool);include=np.asarray(include,bool)
            valid=nonempty&include;is_native=bool(row['local']['native']);native.append(is_native)
            included_steps+=int(include.sum());empty_steps+=int((~nonempty).sum())
            for arm in ARMS:
                pred=row['predictions'][arm]
                score=np.array([np.nan if x is None else x for x in row['scores'][arm]],float)
                counts[arm].append(confusion(correct,pred,include))
                native_counts[arm].append(confusion(correct,pred,valid&is_native))
                aucs[arm].append(pairwise_auc(correct,score,valid))
        panel={'benchmark':benchmark,'n_checked':len(uids),'n_total':len(answers[benchmark]),
               'steps':included_steps,'empty_steps':empty_steps,'local_native_answers':sum(native),
               'coverage_flag':'FULL_REGISTERED_POPULATION','arms':{}}
        for arm in ARMS:
            array=np.asarray(counts[arm],np.int64);count=array.sum(axis=0)
            assert int(count.sum())==included_steps
            tp,fp,tn,fn=count
            assert tp+fn>=10 and tn+fp>=10,'Too few correct or error labels'
            auc=np.array([np.nan if v is None else v for v in aucs[arm]])
            native_sum=np.asarray(native_counts[arm]).sum(axis=0)
            values={'confusion_TP_FP_TN_FN':count.tolist(),**metrics(count,benchmark),
                    'within_auc':float(np.nanmean(auc)) if np.isfinite(auc).any() else None,
                    'within_auc_answers':int(np.isfinite(auc).sum()),
                    'matched_native_metric':metrics(native_sum,benchmark)['metric'],
                    'label_flag':'BINARY_CORRECTNESS_VALIDATED;BOTH_CLASSES_GE10'}
            panel['arms'][arm]=values
            table.append({'cell':cell,'benchmark':benchmark,'arm':arm,'answers':len(uids),'steps':included_steps,
                          'TP':int(tp),'FP':int(fp),'TN':int(tn),'FN':int(fn),
                          **{k:v for k,v in values.items() if k!='confusion_TP_FP_TN_FN'}})
            archive[cell+'__'+arm+'__counts']=array
            archive[cell+'__'+arm+'__within_auc']=auc
        archive[cell+'__uids']=np.asarray(uids,dtype=str)
        archive[cell+'__native']=np.asarray(native,bool)
        result[cell]=panel
    assert len(table)==21
    summary={'status':'PASS','n_checked':total_answers,'n_total':6190,'step_slots_checked':total_steps,
      'confusion_rows':len(table),'coverage_flag':'FULL_REGISTERED_POPULATION',
      'metric_definitions':{'hard2verify':'Harmonic mean of pooled correct-step recall and error-step recall',
                            'socratic':'Arithmetic mean of pooled correct-step F1 and error-step F1',
                            'within_auc':'Unweighted mean over answers containing both label classes and finite nonempty included steps; risk high predicts error; ties count0.5',
                            'confusion':'TP=predicted valid on correct; FP=predicted valid on error; TN=predicted invalid on error; FN=predicted invalid on correct'},
      'empty_steps':'Retained in primary confusion as invalid; excluded from ranking because score is undefined',
      'panels':result,'disjoint_scope':'Population agent independently audits observed-disjoint membership; UID-aligned per-answer contributions provided for its panel',
      'artifact_sha256':artifacts,'script_sha256':sha(__file__),
      'command':[sys.executable,*sys.argv],'seconds':time.perf_counter()-started,
      'independence':'No project scoring/evaluation helper imported; no METRICS/CONTRASTS/report or other agent result read'}
    np.savez_compressed(out/'PER_ANSWER_CONTRIBUTIONS.npz',**archive)
    with (out/'METRIC_ROWS.csv').open('w',newline='',encoding='utf8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    (out/'INDEPENDENT_EXTERNAL.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps({'status':'PASS','answers':total_answers,'steps':total_steps,'rows':21,
                      'metrics':{c:{a:v['metric'] for a,v in p['arms'].items()} for c,p in result.items()}}),flush=True)

if __name__=='__main__': main()
