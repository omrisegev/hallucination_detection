"""Replay failed provisional discovery; never relax final identification guards."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_joint_mass_membership_v1 import OUT,inputs,bank,training_inputs
from spectral_utils.joint_mass_groups import discover_mass_groups
from spectral_utils.joint_sparse_membership import alias_coordinates

def main():
    data,base,uids=inputs();rows=[]
    for kind in ('base','duplicates','noise','near_copies','structured_noise'):
        files=[OUT/f'{kind}_fold{f}.json' for f in range(5)]
        failures=[(f,p,json.loads(p.read_text())) for f,p in enumerate(files) if p.exists() and not json.loads(p.read_text())['model']['valid']]
        if not failures:continue
        x=bank(data,base,uids,kind)
        for outer,path,d in failures:
            m=d['model'];_,args=training_inputs(data,x,outer)
            z=alias_coordinates(args[0],m['aliases']);folds=np.repeat(args[3],np.diff(args[1]))
            discovery=discover_mass_groups(z,folds,seed=m['seed']+100)
            candidates=[]
            for c in discovery['candidates']:
                row={k:v for k,v in c.items() if k not in ('labels','parts')}
                if 'parts' in c:
                    row['fold_group_sizes']=[np.bincount(p).tolist() for p in c['parts']]
                    row['singleton_original_ids']=[m['aliases'][i][0] for i,l in enumerate(c['labels']) if sum(c['labels']==l)==1]
                candidates.append(row)
            rows.append(dict(kind=kind,outer=outer,failure=m['failure'],initial_status=discovery['status'],
                candidate_summary=candidates,positive_grouping_mass=int(np.sum(discovery['mass_audit']['mass']>0))))
    result=dict(status='COMPLETE',scope='post-fit structural failure diagnosis; no parameter or guard changes',failures=rows)
    # Convert NumPy scalar summaries without changing fitted artifacts.
    text=json.dumps(result,indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item())
    (OUT/'FAILURE_DIAGNOSTIC.json').write_bytes((text+'\n').encode());print(text,flush=True)

if __name__=='__main__':main()
