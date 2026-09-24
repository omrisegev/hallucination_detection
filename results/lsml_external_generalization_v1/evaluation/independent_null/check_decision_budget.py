"""Independent post-evaluation budget-bound check; no root diagnostics imported."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
EVAL=OUT.parent
INPUTS=ROOT/'scratch/external_generalization_private/inputs/evaluator_only'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def counts(y,p):
    return np.array([sum(y&p),sum(~y&p),sum(~y&~p),sum(y&~p)],dtype=np.int64)
def metric(c,bench):
    tp,fp,tn,fn=map(float,c)
    if bench=='socratic':return (tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)+(tn/(2*tn+fp+fn) if 2*tn+fp+fn else 0)
    a=tp/(tp+fn);b=tn/(tn+fp)
    return 2*a*b/(a+b) if a+b else 0.
def oracle(c):
    tp,fp,tn,fn=map(int,c);n=tp+fp+tn+fn;goldpos=tp+fn;predpos=tp+fp
    besttp=min(goldpos,predpos)
    # Constructive maximum agreement under the fixed positive-decision budget.
    return np.array([besttp,predpos-besttp,n-goldpos-predpos+besttp,goldpos-besttp],np.int64)
cells={'hard2verify_qwen3_8b':'hard2verify','socratic_qwen3_8b':'socratic','socratic_qwq32b':'socratic'}
bundle=EVAL/'source/BUNDLE.json';sealed=read(EVAL/'ALL_CELLS_SEALED.json')
assert sealed['bundle_sha256']==sha(bundle)
predictions={};seals={}
for cell in cells:
    predictions[cell]=read(EVAL/cell/'PREDICTIONS.json');seals[cell]=read(EVAL/cell/'SEAL.json')
    assert seals[cell]==sealed['seals'][cell]
    assert seals[cell]['prediction_sha256']==digest(predictions[cell])
    assert seals[cell]['lock_sha256']==sha(bundle)
gold={b:{r['uid']:r for r in read(INPUTS/(b+'.json'))} for b in set(cells.values())}
result={}
for cell,bench in cells.items():
    rows=predictions[cell];assert set(rows)==set(gold[bench])
    arms={}
    for arm in rows[next(iter(rows))]['predictions']:
        observed=[];best=[];clean=cleanflagged=0
        for uid,row in rows.items():
            g=gold[bench][uid];keep=np.asarray(g['include'],bool)
            y=np.asarray(g['correct'],bool)[keep];p=np.asarray(row['predictions'][arm],bool)[keep]
            c=counts(y,p);observed.append(c);best.append(oracle(c))
            if len(y) and y.all():clean+=1;cleanflagged+=int((~p).any())
        observed=np.sum(observed,axis=0);best=np.sum(best,axis=0);globalbest=oracle(observed)
        arms[arm]={'observed_metric':metric(observed,bench),'per_answer_budget_ceiling':metric(best,bench),
                   'global_prediction_prevalence_ceiling':metric(globalbest,bench),'observed_counts':observed.tolist(),
                   'per_answer_oracle_counts':best.tolist(),'global_oracle_counts':globalbest.tolist(),
                   'entirely_correct_answers':clean,'entirely_correct_answers_flagged':cleanflagged}
        assert arms[arm]['observed_metric']<=arms[arm]['per_answer_budget_ceiling']+1e-12
        assert arms[arm]['per_answer_budget_ceiling']<=arms[arm]['global_prediction_prevalence_ceiling']+1e-12
    result[cell]={'answers_checked':len(rows),'steps_checked':int(sum(observed)),'arms':arms}
q=predictions['socratic_qwen3_8b'];w=predictions['socratic_qwq32b'];ids=sorted(q);assert set(ids)==set(w)
qa=[];wa=[];qp=[];wp=[];answer_pearson=[];answer_spearman=[]
for uid in ids:
    mask=np.asarray(gold['socratic'][uid]['include'],bool)
    qp.extend(np.asarray(q[uid]['predictions']['frozen_lsml'])[mask]);wp.extend(np.asarray(w[uid]['predictions']['frozen_lsml'])[mask])
    aq=np.asarray([np.nan if x is None else x for x in q[uid]['scores']['frozen_lsml']])[mask]
    aw=np.asarray([np.nan if x is None else x for x in w[uid]['scores']['frozen_lsml']])[mask]
    qa.extend(aq);wa.extend(aw)
    available=np.isfinite(aq)&np.isfinite(aw)
    if available.sum()>1 and aq[available].std()>1e-12 and aw[available].std()>1e-12:
        answer_pearson.append(float(np.corrcoef(aq[available],aw[available])[0,1]))
        answer_spearman.append(float(spearmanr(aq[available],aw[available]).statistic))
qa=np.asarray(qa);wa=np.asarray(wa);finite=np.isfinite(qa)&np.isfinite(wa)
cross={'arm':'frozen_lsml','answers_checked':len(ids),'decision_steps_checked':len(qp),
       'decisions_disagree':int(np.sum(np.asarray(qp)!=np.asarray(wp))),
       'score_steps_checked':int(finite.sum()),'score_pearson':float(np.corrcoef(qa[finite],wa[finite])[0,1]),
       'score_spearman':float(spearmanr(qa[finite],wa[finite]).statistic),
       'correlation_eligible_answers':len(answer_pearson),
       'equal_answer_mean_pearson':float(np.mean(answer_pearson)),
       'equal_answer_mean_spearman':float(np.mean(answer_spearman))}
payload={'status':'PASS','post_evaluation':True,'primary_contrast':False,'cells':result,'socratic_backbone_comparison':cross,
    'interpretation':'The per-answer ceiling permits oracle reassignment of correctness decisions only within each answer at its OBSERVED predicted-positive count. It is an attainable label-using upper bound under those fixed budgets; not a bound for other thresholds, other per-answer budgets, or all L-SML methods. The global ceiling relaxes per-answer budgets but fixes total predicted positives. These are descriptive oracle diagnostics, not deployable corrections.',
    'proof':'At fixed N, gold positives C, and predicted positives P, TP ranges through [max(0,C+P-N),min(C,P)]. FP=P-TP, FN=C-TP, TN=N-C-P+TP. Increasing TP simultaneously increases both recalls and both class F1 terms because their denominators stay fixed. Thus maximize TP separately per answer, pool resulting confusion counts, then apply the official metric.',
    'bundle_sha256':sha(bundle),'script_sha256':sha(Path(__file__)),'seals':seals,
    'gold_hashes':{b:sha(INPUTS/(b+'.json')) for b in gold}}
target=OUT/'DECISION_BUDGET_CHECK.json';temp=target.with_suffix('.tmp')
temp.write_text(json.dumps(payload,sort_keys=True,allow_nan=False)+'\n',encoding='utf8');os.replace(temp,target)
print(json.dumps({'frozen':{c:r['arms']['frozen_lsml'] for c,r in result.items()},'cross_backbone':cross},indent=2))
