"""Independent post-run metric/score replay; does not select or refit methods."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json
from pathlib import Path
import sys
import numpy as np
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_digitfree_broad50_v1 import OUT, ARMS, load_data, sha
from scripts.run_lsml_gate_locator_research_v1 import dump


def independent_metrics(score,gate,data,eligible=None):
    offsets=data['offsets']; target=data['target']; cells=data['cells'].astype(str)
    eligible=np.ones(len(target),bool) if eligible is None else eligible
    peaks=[]; within=[]; within_index=[]
    for i,(a,b) in enumerate(zip(offsets[:-1],offsets[1:])):
        peaks.append(int(np.argmax(score[a:b])))
        if cells[i].startswith('prmbench') and eligible[i]:
            y=data['labels'][a:b]; ok=y>=0; y=y[ok]==1
            if y.any() and (~y).any():
                within.append(float(roc_auc_score(y,score[a:b][ok]))); within_index.append(i)
    pred=np.where(gate,np.asarray(peaks),-1); cellmetrics={}
    for cell in sorted(set(cells)):
        if not cell.startswith('pb_'): continue
        clean=(cells==cell)&(target==-1); error=(cells==cell)&(target>=0)
        ca=float(np.mean((pred[clean]==target[clean])&eligible[clean]))
        ea=float(np.mean((pred[error]==target[error])&eligible[error]))
        cellmetrics[cell]=2*ca*ea/(ca+ea) if ca+ea else 0.
    return dict(pb=float(np.mean(list(cellmetrics.values()))),within=float(np.mean(within)) if within else None,
        within_n=len(within),pb_cells=cellmetrics,prediction=pred,within_indexes=within_index)


def main():
    data=load_data(); results=json.loads((OUT/'RESULTS.json').read_text()); manifest=json.loads((OUT/'MANIFEST.json').read_text())
    for path,digest in manifest['hashes'].items():
        assert sha(path)==digest,f'implementation/input drift {path}'
    with np.load(OUT/'SCORES.npz') as z: scores={k:z[k] for k in z.files}
    np.testing.assert_array_equal(scores.pop('gate'),data['gate'])
    rowfolds=np.repeat(data['folds'],np.diff(data['offsets'])); maximum=0.; native={}; replay={}
    folds=[json.loads((OUT/f'fold_{f}.json').read_text()) for f in range(5)]
    for arm,score in scores.items():
        assert np.isfinite(score).all()
        if arm in ARMS:
            for f in folds:
                mask=rowfolds==f['outer']; w=np.asarray(f['weights'][arm])
                expected=data['x'][mask]@w
                maximum=max(maximum,float(np.max(np.abs(expected-score[mask]))))
                np.testing.assert_allclose(score[mask],expected,atol=1e-12,rtol=1e-12)
        m=independent_metrics(score,data['gate'],data)
        np.testing.assert_allclose([m['pb'],m['within']],[results['metrics'][arm]['pb'],results['metrics'][arm]['within']],atol=1e-12)
        replay[arm]={k:m[k] for k in ('pb','within','within_n')}
        if arm in ARMS:
            eligible=np.ones(len(data['target']),bool)
            for f in folds:
                if not f['arms'].get(arm,{}).get('valid',arm in ('entropy_H1','equal50')):
                    eligible[data['folds']==f['outer']]=False
            pure=independent_metrics(score,data['gate'],data,eligible)
            native[arm]=dict(answers=int(eligible.sum()),
                pb_counting_invalid_as_failures=pure['pb'],within_available=pure['within'],within_n=pure['within_n'])
    # Matched complete-population historical anchors must replay the old tail gate.
    np.testing.assert_allclose(replay['original4']['pb'],.374749,atol=5e-7)
    np.testing.assert_allclose(replay['innovation5']['pb'],.398314,atol=5e-7)
    report=dict(status='PASS',score_replay_max_absolute_difference=maximum,independent_metrics=replay,
        native_failure_accounting=native,score_sha256=sha(OUT/'SCORES.npz'),manifest_sha256=sha(OUT/'MANIFEST.json'),
        input_shape=list(data['x'].shape),missing_entries=int((~data['available']).sum()),
        checks=['all implementation and compact source hashes unchanged','all fold weight maps replay',
                'independent sklearn within-answer AUC','independent eight-cell PB formula',
                'both historical non-digit PB anchors replay','native coverage / invalid-as-failure accounting'])
    dump(OUT/'AUDIT.json',report); print(json.dumps(report,indent=2))


if __name__=='__main__':main()
