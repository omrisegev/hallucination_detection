"""Descriptive source validation, labels used only after separated predictions."""
import json,pickle,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spectral_utils.external_generalization.artifacts import atomic_json,file_hash
from spectral_utils.external_generalization.evaluation import confusion,metric,within_auc
from spectral_utils.external_generalization.scoring import ALL_ARMS


def main():
    out=ROOT/'results/lsml_external_generalization_v1/evaluation/source'
    records=json.loads((ROOT/'results/localization_full_benchmark_v3/evaluation/JOINED.json').read_text())['records']
    jp=ROOT/'results/localization_full_benchmark_v3/evaluation/JOINED.npz';data=np.load(jp);off=data['offsets'];y=~data['labels'].astype(bool)
    mp=ROOT/'dataset_cache/four_localization/prmbench_qwen25math7b_full/prmbench_prm.pkl'
    meta={r['idx']:r for r in pickle.loads(mp.read_bytes()).values()}
    eligible=np.array([r['cell'].startswith('prm') and meta[r['row_id']]['classification']!='correct' for r in records])
    for i,r in enumerate(records):
        if r['cell'].startswith('prm'):
            m=meta[r['row_id']];a,b=off[i:i+2]
            np.testing.assert_array_equal(y[a:b],[j+1 not in m['error_steps'] for j in range(b-a)])
    z=np.load(out/'VALIDATION.npz');mask=np.repeat(eligible,np.diff(off));sf=np.repeat(z['folds'],np.diff(off))
    result={'source_answers_all':len(records),'prmb_noncontrol_answers':int(eligible.sum()),'prmb_noncontrol_steps':int(mask.sum()),'local_native_answers_all':int(z['native'].sum()),'arms':{},'validation_sha256':file_hash(out/'VALIDATION.npz'),'labels_sha256':file_hash(jp),'metadata_sha256':file_hash(mp),'access':'3 fit folds /1 calibration fold /1 test fold; all development evidence'}
    for arm in ALL_ARMS:
        p=z[arm+'_pred'];s=z[arm+'_score'];c=confusion(y,p,mask);aucs=[]
        for i,(a,b) in enumerate(zip(off[:-1],off[1:])):
            if eligible[i]:
                v=within_auc(y[a:b],s[a:b],np.ones(b-a,bool))
                if v is not None:aucs.append(v)
        result['arms'][arm]={'prmscore':float(metric(c,'socratic')),'confusion':c.tolist(),'within_auc':float(np.mean(aucs)),'within_auc_answers':len(aucs),'fold_prmscore':[float(metric(confusion(y,p,mask&(sf==f)),'socratic')) for f in range(5)]}
    atomic_json(out/'VALIDATION_METRICS.json',result)
    print('source validation summarized',int(eligible.sum()),flush=True)

if __name__=='__main__':main()
