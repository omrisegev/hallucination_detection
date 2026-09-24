"""Descriptive cross-backbone diversity from sealed Socratic predictions, no fusion."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/lsml_external_generalization_v1/evaluation'
def main():
    assert (OUT/'ALL_CELLS_SEALED.json').exists()
    a=json.loads((OUT/'socratic_qwen3_8b/PREDICTIONS.json').read_text(encoding='utf8'))
    b=json.loads((OUT/'socratic_qwq32b/PREDICTIONS.json').read_text(encoding='utf8'))
    labels=json.loads((ROOT/'scratch/external_generalization_private/inputs/evaluator_only/socratic.json').read_text(encoding='utf8'))
    assert set(a)==set(b)=={g['uid'] for g in labels}
    result={'answers':len(a),'steps':sum(sum(g['include']) for g in labels),'interpretation':'Descriptive post-evaluation complementarity, not a registered contrast, new fusion method, or usable oracle. Identical-repeat telemetry audit is separate.','arms':{}}
    for arm in ('frozen_lsml','local_lsml','ct7'):
        events=np.zeros(4,dtype=int);rho=[];disagreements=0
        for g in labels:
            x=a[g['uid']];z=b[g['uid']];mask=np.asarray(g['include'],bool);y=np.asarray(g['correct'],int)[mask]
            p=np.asarray(x['predictions'][arm])[mask];q=np.asarray(z['predictions'][arm])[mask]
            e=p!=y;f=q!=y;events+=np.array([np.sum(~e&~f),np.sum(e&~f),np.sum(~e&f),np.sum(e&f)])
            disagreements+=int(np.sum(p!=q))
            s=np.asarray([np.nan if v is None else v for v in x['scores'][arm]]);t=np.asarray([np.nan if v is None else v for v in z['scores'][arm]])
            valid=mask&np.isfinite(s)&np.isfinite(t)
            if valid.sum()>1 and np.std(s[valid])>1e-12 and np.std(t[valid])>1e-12:rho.append(float(spearmanr(s[valid],t[valid]).statistic))
        total=int(events.sum());assert total==result['steps']
        result['arms'][arm]={'both_right':int(events[0]),'qwen3_wrong_only':int(events[1]),'qwq_wrong_only':int(events[2]),'both_wrong':int(events[3]),'decision_disagreement_steps':disagreements,'decision_disagreement_fraction':disagreements/total,'mean_within_answer_score_spearman':float(np.mean(rho)),'score_spearman_answers':len(rho),'oracle_select_correct_backbone_step_accuracy':1-float(events[3])/total,'oracle_warning':'uses gold for every step; descriptive upper bound only, not official PRMScore or a deployable selector'}
    (OUT/'BACKBONE_COMPLEMENTARITY.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8',newline='\n');print('cross-backbone diversity saved; no new fusion or parameter fitting')
if __name__=='__main__':main()
