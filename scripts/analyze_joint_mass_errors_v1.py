"""Full PB gate/locator attribution from frozen predictions; no parameter choice."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'
import json,sys,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_joint_fixed_readout_v1 import inputs
OUT=ROOT/'results/joint_mass_membership_v1'
ARMS=('base__mass','near_copies__mass','base__previous','base__feasible_reference','innovation5','historical_bocpd')

def main():
    data,_,uids=inputs();result=json.loads((OUT/'RESULTS.json').read_text())
    with np.load(OUT/'SCORES.npz') as z:
        np.testing.assert_array_equal(z['gate'],data['gate'])
        scores={k:z[k] for k in ARMS}
    gate=data['gate'].astype(bool);target=data['target'];cells=data['cells'].astype(str)
    pb=np.array([c.startswith('pb_') for c in cells]);error=pb&(target>=0);clean=pb&(target<0)
    assert pb.sum()==6800 and error.sum()==4442 and clean.sum()==2358
    correct={};summary={}
    for name,s in scores.items():
        peak=np.array([int(np.argmax(s[a:b])) for a,b in zip(data['offsets'][:-1],data['offsets'][1:])])
        located=peak==target;correct[name]=np.where(gate,peak,-1)==target
        row=dict(error_hit=int(np.sum(error&gate&located)),
            gate_only_miss=int(np.sum(error&~gate&located)),
            locator_only_miss=int(np.sum(error&gate&~located)),
            both_miss=int(np.sum(error&~gate&~located)),
            clean_correct=int(np.sum(clean&~gate)),clean_false_alarm=int(np.sum(clean&gate)))
        assert sum(row[k] for k in ('error_hit','gate_only_miss','locator_only_miss','both_miss'))==4442
        offsets=peak[error&~located]-target[error&~located]
        row['wrong_peak_before_target']=int(np.sum(offsets<0));row['wrong_peak_after_target']=int(np.sum(offsets>0))
        row['wrong_peak_step_offset_quantiles']=np.quantile(offsets,[0,.25,.5,.75,1]).tolist()
        per_cell=[]
        for cell in sorted(set(cells[pb])):
            ca=float(np.mean(correct[name][clean&(cells==cell)]))
            ea=float(np.mean(correct[name][error&(cells==cell)]))
            per_cell.append(2*ca*ea/(ca+ea) if ca+ea>0 else 0.)
        np.testing.assert_allclose(np.mean(per_cell),result['metrics'][name]['pb'],atol=1e-12,rtol=0)
        summary[name]=row
    changes={}
    for left,right in [('base__mass','base__previous'),('base__mass','base__feasible_reference'),
                       ('near_copies__mass','base__mass'),('base__mass','innovation5')]:
        wins=int(np.sum(error&correct[left]&~correct[right]));losses=int(np.sum(error&~correct[left]&correct[right]))
        assert wins-losses==summary[left]['error_hit']-summary[right]['error_hit']
        changes[left+' minus '+right]=dict(error_exact_gains=wins,error_exact_losses=losses,
            clean_decisions_changed=int(np.sum(clean&(correct[left]!=correct[right]))))
    assert all(v['clean_decisions_changed']==0 for v in changes.values())
    output=dict(status='PASS',scope='full development PB post-evaluation attribution; no feature or threshold selected',
        answers=len(uids),pb_answers=6800,pb_error=4442,pb_clean=2358,
        score_sha256=hashlib.sha256((OUT/'SCORES.npz').read_bytes()).hexdigest(),
        target_sha256=hashlib.sha256(np.asarray(target).tobytes()).hexdigest(),
        interpretation='Gate-only and both denote misses on erroneous answers; clean false alarms are reported separately. Same gate in all arms.',
        arms=summary,paired_changes=changes)
    (OUT/'GATE_LOCATOR_DIAGNOSTIC.json').write_bytes((json.dumps(output,indent=2)+'\n').encode())
    print(json.dumps(output,indent=2))

if __name__=='__main__':main()
