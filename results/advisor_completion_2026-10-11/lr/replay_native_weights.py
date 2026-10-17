"""Replay the original transductive label-free bank scores, separately from OOF."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from spectral_utils.advisor_lr import fit_labelfree,as_dict
INP=ROOT/'results/advisor_completion_2026-10-11/inputs'
OUT=Path(__file__).resolve().parent
BANKS=['N11','N13','N22','N40','N53','N14','N16','N25','N43','N56']

def main():
    rows=[]; models={}
    for bank in BANKS:
        z=np.load(INP/f'bank_V_{bank}.npz'); original=np.load(INP/f'bank_{bank}.npz')
        names=z['columns'].astype(str).tolist(); X=z['V'].astype(float)
        for method in ('LSML_CONT','IU_PCR'):
            refkey='S__'+method
            try:
                model=fit_labelfree(X,method);score=model.predict(X)
                reference=original[refkey]
                # Export hook stores answer-standardized scores, not raw
                # affine outputs. Match that representation before replay.
                standardized=np.zeros(len(score))
                for start,end in zip(original['offsets'][:-1],original['offsets'][1:]):
                    value=score[start:end];sd=value.std()
                    if sd>1e-12:standardized[start:end]=(value-value.mean())/sd
                error=float(np.max(np.abs(standardized-reference)))
                rows.append({'bank':bank,'method':method,'status':'PASS' if error<=1e-9 else 'FAIL',
                    'n_steps':len(X),'max_abs_score_difference':error,
                    'affine_max_abs':model.diagnostic['affine_max_abs'],
                    'raw_unstandardized_max_abs_vs_export':float(np.max(np.abs(score-reference))),
                    'export_representation':'per-answer z-score of native affine output'})
                models[bank+'__'+method]={'names':names,'model':as_dict(model),
                    'interpretation':'Original all-population transductive unsupervised fit; not new group OOF'}
                print(rows[-1],flush=True)
            except Exception as e:
                rows.append({'bank':bank,'method':method,'status':'EXPLICIT_FAILURE',
                    'n_steps':len(X),'reason':f'{type(e).__name__}: {e}'})
                print(rows[-1],flush=True)
    (OUT/'NATIVE_WEIGHT_REPLAY.json').write_text(json.dumps({'rows':rows,'models':models,
      'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2,allow_nan=False))
    with (OUT/'NATIVE_WEIGHT_REPLAY.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
    if any(r['status']=='FAIL' for r in rows):raise SystemExit('Native score replay parity failed')

if __name__=='__main__':main()
