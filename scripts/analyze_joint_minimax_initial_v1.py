"""Compare saved initial references on explicitly matched original features."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'
import json,hashlib
from pathlib import Path
import numpy as np
from sklearn.metrics import adjusted_rand_score
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/joint_minimax_membership_v1'
OLD=ROOT/'results/joint_feasible_membership_v1'

def original_positions(model):
    # Sparse membership may have removed an ORIGINAL feature, not an added
    # copy. Never align by slicing the first51 retained rows.
    return {original:row for row,canonical in enumerate(model['initial_active'])
            for original in model['aliases'][canonical] if original<51}

def main():
    rows=[];hashes={}
    for outer in range(5):
        models=[]
        for kind in ('base','near_copies'):
            path=OUT/f'{kind}_fold{outer}.json';priorpath=OLD/path.name
            m=json.loads(path.read_text())['model'];prior=json.loads(priorpath.read_text())['model']
            for p in (path,priorpath):hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
            for key in ('initial_factor_matrix','initial_factor_information'):
                np.testing.assert_array_equal(m['refinement']['selection'][key],prior['refinement']['selection'][key])
            np.testing.assert_array_equal(m['membership']['observed_covariance'],prior['membership']['observed_covariance'])
            np.testing.assert_array_equal(m['initial_active'],prior['initial_active']);models.append(m)
        b,n=models;bp=original_positions(b);np_=original_positions(n);common=sorted(bp.keys()&np_.keys())
        bi=[bp[i] for i in common];ni=[np_[i] for i in common]
        bc=np.asarray(b['membership']['observed_covariance']);nc=np.asarray(n['membership']['observed_covariance'])
        np.testing.assert_allclose(bc[np.ix_(bi,bi)],nc[np.ix_(ni,ni)],atol=1e-12,rtol=0)
        bs=b['refinement']['selection'];ns=n['refinement']['selection']
        bv=np.asarray(bs['initial_factor_matrix'])[bi,0];nv=np.asarray(ns['initial_factor_matrix'])[ni,0]
        bl=np.asarray(bs['path'][0]['labels'])[bi];nl=np.asarray(ns['path'][0]['labels'])[ni]
        rows.append(dict(outer=outer,common_original_ids=common,
            base_removed_original=sorted(set(range(51))-bp.keys()),near_removed_original=sorted(set(range(51))-np_.keys()),
            base_group_sizes=bs['path'][0]['group_sizes'],near_group_sizes=ns['path'][0]['group_sizes'],
            original_group_ari=float(adjusted_rand_score(bl,nl)),
            original_global_loading_cosine=float(abs(bv@nv)/(np.linalg.norm(bv)*np.linalg.norm(nv))),
            identical_to_step410=True))
    result=dict(status='PASS',scope='post-fit descriptive initial-reference comparison; not proof of which component causes quality loss',
        initial_fits_matching_step410=10,models=rows,model_hashes=hashes)
    (OUT/'INITIAL_REFERENCE_DIAGNOSTIC.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
    for row in rows:print({k:v for k,v in row.items() if k!='common_original_ids'})

if __name__=='__main__':main()
