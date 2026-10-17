"""Describe trained feature membership; never select a method from labels."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import json,sys,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spectral_utils.digitfree_broad50 import NAMES
OUT=ROOT/'results/joint_mass_membership_v1'
KINDS=('base','duplicates','noise','near_copies','structured_noise')
ORIGINAL=list(NAMES)+['bocpd_signed_residual']

def family(index):
    if index<15:return 'probability_ranks'
    if index<19:return 'provided_token_confidence'
    if index<21:return 'tail_mass'
    if index<34:return 'distribution_shape'
    if index==34:return 'top2_ratio'
    if index<48:return 'prefix_innovation'
    if index==48:return 'top15_turnover'
    if index==49:return 'top50_truncated_js'
    if index==50:return 'bocpd'
    return 'added'

def main():
    rows=[];summary={};hashes={}
    for kind in KINDS:
        names=ORIGINAL+([] if kind=='base' else [f'{kind}_{i+1}' for i in range(15)])
        counts=np.zeros(51,dtype=int);weights=[];kept=[]
        for outer in range(5):
            path=OUT/f'{kind}_fold{outer}.json'
            hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
            meta=json.loads(path.read_text());m=meta['model']
            if not m['valid']:
                kept.append(None);rows.append(dict(kind=kind,outer=outer,valid=False,failure=m['failure']));continue
            selected=m['active_original'];w=np.asarray(m['expanded_weights'])
            assert len(w)==len(names);np.testing.assert_allclose(np.abs(w).sum(),1.,atol=1e-12)
            counts[[i for i in selected if i<51]]+=1;weights.append(w);kept.append(len(m['active']))
            row=dict(kind=kind,outer=outer,valid=True,selected_canonical=len(m['active']),
                selected_original=meta['selected_original'],selected_added=meta['selected_added'],
                bocpd_retained=meta['bocpd_retained'],
                selected=[dict(index=i,name=names[i],weight=float(w[i])) for i in selected],
                omitted_original=[names[i] for i in range(51) if i not in selected])
            rows.append(row)
        average=np.mean(np.abs(np.asarray(weights)),axis=0) if weights else np.zeros(len(names));mass={}
        for i,value in enumerate(average):mass[family(i)]=mass.get(family(i),0.)+float(value)
        np.testing.assert_allclose(sum(mass.values()),1. if weights else 0.,atol=1e-12)
        summary[kind]=dict(native_folds=len(weights),canonical_counts=kept,original_retained_folds=dict(zip(ORIGINAL,map(int,counts))),
            mean_absolute_expanded_weight_by_family=mass)
    result=dict(status='COMPLETE',scope='descriptive trained membership, not quality attribution',
        note='Exact aliases count once canonically; expanded weights split their mass. Active membership can include zero readout weights. Frequencies and mean weights describe NATIVE folds only; failed folds are explicit and never counted as a selected roster.',
        model_hashes=hashes,summary=summary,models=rows)
    (OUT/'FEATURE_ROSTERS.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
    lines=['# Learned rosters and weight allocation','',result['note'],'',
        'Weights describe this fit, not a causal contribution or feature quality test.','',
        'Native fit counts: '+str({k:v['native_folds'] for k,v in summary.items()}),'',
        '| Original feature | Base retained folds | Near-copy retained folds |','|---|---:|---:|']
    for name in ORIGINAL:
        lines.append(f"| {name} | {summary['base']['original_retained_folds'][name]}/{summary['base']['native_folds']} | {summary['near_copies']['original_retained_folds'][name]}/{summary['near_copies']['native_folds']} |")
    lines+=['','| Family | Base mean absolute weight | Near-copy mean absolute weight |','|---|---:|---:|']
    for group,v in summary['near_copies']['mean_absolute_expanded_weight_by_family'].items():
        lines.append(f"| {group} | {summary['base']['mean_absolute_expanded_weight_by_family'].get(group,0):.6f} | {v:.6f} |")
    (OUT/'FEATURE_ROSTERS.md').write_bytes(('\n'.join(lines)+'\n').encode())
    print(json.dumps({k:dict(canonical_counts=v['canonical_counts'],bocpd_folds=v['original_retained_folds']['bocpd_signed_residual']) for k,v in summary.items()}))

if __name__=='__main__':main()
