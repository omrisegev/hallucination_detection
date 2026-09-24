"""Descriptive prevalence ceilings after sealed evaluation; never chooses thresholds."""
import json
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/lsml_external_generalization_v1/evaluation'
def value(tp,fp,tn,fn,benchmark):
    if benchmark=='hard2verify':
        a=tp/(tp+fn);b=tn/(tn+fp)
        return 2*a*b/(a+b) if a+b else 0.
    return .5*(2*tp/(2*tp+fp+fn)+2*tn/(2*tn+fp+fn))
def main():
    assert (OUT/'ALL_CELLS_SEALED.json').exists()
    metrics=json.loads((OUT/'METRICS.json').read_text(encoding='utf8'))
    result={'interpretation':'Descriptive oracle assignment ceiling at each observed prediction prevalence. Uses evaluator labels after sealing; not a deployable arm, threshold search, target calibration, or causal diagnosis. The global bound ignores answer/score constraints; the per-answer bound preserves per-answer predicted class counts but freely permutes assignments within each answer. Neither is attainable evidence or causal attribution.','cells':{}}
    for cell,m in metrics.items():
        rows={};per_answer=np.load(OUT/cell/'COUNTS.npz')
        for arm,v in m['arms'].items():
            tp,fp,tn,fn=v['confusion'];gold_correct=tp+fn;gold_error=tn+fp;pred_correct=tp+fp;pred_error=tn+fn
            best=(min(gold_correct,pred_correct),max(pred_correct-gold_correct,0),min(gold_error,pred_error),max(gold_correct-pred_correct,0))
            ceiling=value(*best,m['benchmark']);actual=value(tp,fp,tn,fn,m['benchmark'])
            assert actual<=ceiling+1e-12 and abs(actual-v['metric'])<1e-12
            c=per_answer[arm];gc=c[:,0]+c[:,3];ge=c[:,2]+c[:,1];pc=c[:,0]+c[:,1];pe=c[:,2]+c[:,3]
            per_best=np.stack([np.minimum(gc,pc),np.maximum(pc-gc,0),np.minimum(ge,pe),np.maximum(gc-pc,0)],axis=1).sum(0)
            per_ceiling=value(*per_best,m['benchmark']);assert actual<=per_ceiling+1e-12 and per_ceiling<=ceiling+1e-12
            all_correct=ge==0
            rows[arm]={'actual':actual,'per_answer_prediction_count_ceiling':float(per_ceiling),'per_answer_ceiling_confusion':per_best.tolist(),'all_correct_answers':int(all_correct.sum()),'all_correct_answers_flagged':int(np.sum(all_correct&(pe>0))),'false_error_steps_in_all_correct_answers':int(pe[all_correct].sum()),'optimistic_fixed_prevalence_ceiling':ceiling,'gap_to_bound':ceiling-actual,'ceiling_confusion':best,'gold_error_fraction':gold_error/(gold_correct+gold_error),'predicted_error_fraction':pred_error/(gold_correct+gold_error)}
        result['cells'][cell]=rows
    (OUT/'CALIBRATION_DIAGNOSTICS.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8',newline='\n')
    print('descriptive ceiling diagnostics saved; no parameter or prediction changes')
if __name__=='__main__':main()
