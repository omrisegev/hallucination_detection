"""Replay pinned official evaluators on every sealed prediction (no inference)."""
import ast,hashlib,json
from pathlib import Path
from sklearn.metrics import recall_score
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/lsml_external_generalization_v1/evaluation'
PRIVATE=ROOT/'scratch/external_generalization_private'

def load(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def functions(path,names,ns):
    nodes=[n for n in ast.parse(path.read_text(encoding='utf8')).body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
    return ns

def main():
    assert (OUT/'ALL_CELLS_SEALED.json').exists()
    hp=PRIVATE/'sources/hard2verify/utils.py'
    sp=PRIVATE/'sources/prmeval_classified_task.py'
    hard=functions(hp,{'calculate_metrics'},{'recall_score':recall_score})
    soc=functions(sp,{'evaluate_function','eval_on_hallucination_step'},{})
    summary=load(OUT/'METRICS.json');result={'sources':{},'cells':{}}
    for path,revision in ((hp,'9cd1eca6fa662c3dcd4576ee5ae8668515add186'),(sp,'a1ae4eab803e14c4c6316bf89d1679b0486faa8c')):
        result['sources'][str(path.relative_to(ROOT))]={'sha256':sha(path),'revision':revision}
    for cell,m in summary.items():
        bench=m['benchmark'];predpath=OUT/cell/'PREDICTIONS.json';pred=load(predpath)
        goldpath=PRIVATE/'inputs/evaluator_only'/f'{bench}.json';gold=load(goldpath)
        assert {g['uid'] for g in gold}==set(pred)
        assert all(all(g['include']) for g in gold), 'official full-set adapter requires all steps'
        out={'answers':len(gold),'steps':sum(len(g['correct']) for g in gold),'prediction_sha256':sha(predpath),'gold_sha256':sha(goldpath),'arms':{}}
        for arm,v in m['arms'].items():
            if bench=='hard2verify':
                y=[int(x) for g in gold for x in g['correct']]
                p=[int(x) for g in gold for x in pred[g['uid']]['predictions'][arm]]
                official=hard['calculate_metrics'](p,y)
                value=official['balanced_f1_score'];assert value==round(100*v['metric'],2)
                out['arms'][arm]={'official_percent_rounded':value,'ours_percent_rounded':round(100*v['metric'],2),'pass':True}
            else:
                meta=[];rows=[];counts=[0]*4
                for g in gold:
                    errors=[i+1 for i,x in enumerate(g['correct']) if not x]+g.get('out_of_range_error_indices',[])
                    meta.append({'idx':g['uid'],'classification':g['category'],'error_steps':errors})
                    p=pred[g['uid']]['predictions'][arm]
                    rows.append({'idx':g['uid'],'scores':{'step_level_validity_labels':p}})
                    cm=soc['eval_on_hallucination_step'](errors,p)['f1_matrix']
                    counts=[a+cm[k] for a,k in zip(counts,('TP','FP','TN','FN'))]
                official=soc['evaluate_function'](rows,meta)['total_hallucination_results']
                value=(official['f1']+official['negative_f1'])/2
                assert counts==v['confusion'],(cell,arm,counts,v['confusion'])
                assert abs(value-v['metric'])<1e-12,(cell,arm,value,v['metric'])
                out['arms'][arm]={'official_prmscore':value,'ours_prmscore':v['metric'],'absolute_difference':abs(value-v['metric']),'confusion':counts,'pass':True}
        result['cells'][cell]=out
    result['passed']=True
    (OUT/'OFFICIAL_METRIC_REPLAY.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8',newline='\n')
    print(json.dumps({'passed':True,'cells':len(result['cells']),'arms':21}))
if __name__=='__main__':main()
