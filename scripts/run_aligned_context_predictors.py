"""Answer-resumable full-data prediction study. No correctness labels loaded."""
from pathlib import Path
import sys, json, hashlib, io, sqlite3, time
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from spectral_utils.context_training import FeatureBundle
from spectral_utils.temporal_context_models import RidgePredictor, history_windows
from spectral_utils.context_readout import prediction_readouts
from spectral_utils.aligned_context_predictors import bocpd_mean, noreset_mean, past_mean, agreement, remove_current

OUT = ROOT/'results/aligned_context_predictors_v1'
METHODS = ('ridge', 'mean16', 'bocpd', 'noreset', 'zero')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path); tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf8'); tmp.replace(path)


def ridge_prediction(z, model):
    n = len(z); windows, mask = history_windows(z, np.arange(n), history=16)
    return model.predict(np.column_stack((windows.reshape(n,-1),mask,(np.arange(n)+.5)/n)))


def models(bundle):
    result = {}; audit = {}
    for f in range(5):
        path = ROOT/f'results/temporal_linear_context_v1/innovation5__exclude_{f}_model.npz'
        spec = json.loads(path.with_suffix('.json').read_text())
        for name, ids in zip(('training', 'validation', 'held'), bundle.split((f,))):
            expected = {bundle.metadata[i]['group_id'] for i in ids}
            if set(spec[name+'_groups']) != expected: raise ValueError('Ridge group mismatch')
        with np.load(path, allow_pickle=False) as p:
            result[f] = RidgePredictor(p['coefficient'], p['mean'], p['scale'])
        audit[str(f)] = dict(model_sha256=sha(path), split_sha256=sha(path.with_suffix('.json')),
                            training_groups=spec['training_groups'], held_groups=spec['held_groups'])
    return result, audit


def score_one(bundle, i, model, base):
    m = bundle.metadata[i]; start = m['offset']; n = m['tokens']
    raw = np.asarray(bundle.features[start:start+n], float)[:,bundle.columns]
    z = (raw-bundle.mean[i])/bundle.scale[i]
    predictions = dict(ridge=ridge_prediction(z, model), mean16=past_mean(z),
                       bocpd=bocpd_mean(z), noreset=noreset_mean(z), zero=np.zeros_like(z))
    spans = np.asarray(bundle.spans[m['step_start']:m['step_stop']])-start
    scores = {}; diagnostics = {}; readout_delta = 0.
    current = z.mean(1); rr = current-predictions['ridge'].mean(1)
    for name, pred in predictions.items():
        if pred.shape != z.shape or not np.isfinite(pred).all(): raise FloatingPointError(name)
        rows = prediction_readouts(raw, z, pred, spans, base)
        scores[name] = rows['signed_residual_0.25']
        residual = z-pred; signed = residual.mean(1)
        # Independent scalar/sort readout check for every answer and predictor.
        aux = np.array([np.sort(signed[a:b])[-min(10,b-a):].mean() for a,b in spans])
        expected = base.copy()
        if aux.std() > 1e-12: expected = base+.25*base.std()*(aux-aux.mean())/aux.std()
        delta = float(np.max(np.abs(expected-scores[name]))); readout_delta = max(readout_delta, delta)
        np.testing.assert_allclose(expected,scores[name],atol=2e-12,rtol=0)
        diagnostics[name] = dict(mse=np.square(residual).mean(0).tolist(),
            mse_after16=np.square(residual[16:]).mean(0).tolist() if n>16 else None,
            prediction_correlation_ridge=agreement(pred.mean(1),predictions['ridge'].mean(1)),
            signed_residual_correlation_ridge=agreement(signed,rr),
            residual_correlation_after_removing_current=agreement(remove_current(signed,current),remove_current(rr,current)),
            signed_residual_correlation_current=agreement(signed,current))
    return scores, dict(uid=m['uid'],tokens=n,readout_delta=readout_delta,methods=diagnostics)


def run():
    OUT.mkdir(exist_ok=True); bundle = FeatureBundle(ROOT/'results/temporal_context_data_v1','innovation5')
    fitted, fit_audit = models(bundle)
    paths = [Path(__file__), ROOT/'spectral_utils/aligned_context_predictors.py',
        ROOT/'spectral_utils/context_readout.py', ROOT/'spectral_utils/temporal_context_models.py',
        ROOT/'docs/experiments/ALIGNED_CONTEXT_PREDICTORS_20260915.md',
        ROOT/'results/temporal_context_data_v1/MANIFEST.json',
        ROOT/'results/temporal_context_data_v1/METADATA.json',
        ROOT/'results/temporal_context_data_v1/features.npy',
        ROOT/'results/temporal_research_baseline_v1/SCORES_FROZEN.npz',
        ROOT/'results/temporal_linear_context_v1/SCORES_FROZEN.npz']
    manifest = dict(schema='aligned-context-predictors-v1', methods=METHODS,
        answers=len(bundle.metadata),tokens=int(bundle.length.sum()),steps=int(bundle.manifest['steps']),
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in paths},ridge_fits=fit_audit,
        correctness_labels_used=False,normalization='whole-answer, offline',budget_seconds=7200)
    manifest=json.loads(json.dumps(manifest))
    mp = OUT/'MANIFEST.json'
    if mp.exists() and json.loads(mp.read_text())!=manifest: raise ValueError('immutable manifest changed')
    write(mp, manifest)
    with np.load(ROOT/'results/temporal_research_baseline_v1/SCORES_FROZEN.npz') as f:
        original = f['steps__append_innovation__H0lim']
    with np.load(ROOT/'results/temporal_linear_context_v1/SCORES_FROZEN.npz') as f:
        ridge_reference = f['steps__innovation5__real__signed_residual_0.25']
    db=sqlite3.connect(OUT/'ANSWERS.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS answers (idx INTEGER PRIMARY KEY, scores BLOB, diagnostic TEXT)');db.commit()
    done={r[0] for r in db.execute('SELECT idx FROM answers')}; start=time.perf_counter()
    elapsed_before=json.loads((OUT/'RUN_STATE.json').read_text()).get('scoring_seconds',0) if (OUT/'RUN_STATE.json').exists() else 0
    def state(status,**kwargs):
        write(OUT/'RUN_STATE.json',dict(status=status,answers=len(done),expected=len(bundle.metadata),
            scoring_seconds=elapsed_before+time.perf_counter()-start,**kwargs))
    state('SCORING')
    try:
        for i,m in enumerate(bundle.metadata):
            if i in done: continue
            sl=slice(m['step_start'],m['step_stop'])
            scores,diag=score_one(bundle,i,fitted[m['fold']],original[sl])
            np.testing.assert_allclose(scores['ridge'],ridge_reference[sl],atol=2e-12,rtol=0)
            buf=io.BytesIO();np.savez_compressed(buf,**scores)
            db.execute('INSERT INTO answers VALUES (?,?,?)',(i,buf.getvalue(),json.dumps(diag,allow_nan=False)))
            done.add(i)
            if len(done)%100==0:
                db.commit();state('SCORING');print('[predictors]',len(done),len(bundle.metadata),'seconds',round(time.perf_counter()-start,1),flush=True)
            if time.perf_counter()-start>7200:
                db.commit();state('BUDGET_PAUSED');return
        db.commit()
        scores={name:np.full(len(original),np.nan) for name in METHODS};diagnostics=[]
        for i,blob,diag in db.execute('SELECT idx,scores,diagnostic FROM answers ORDER BY idx'):
            m=bundle.metadata[i];sl=slice(m['step_start'],m['step_stop'])
            with np.load(io.BytesIO(blob),allow_pickle=False) as f:
                for name in METHODS:scores[name][sl]=f[name]
            diagnostics.append(json.loads(diag))
        if any(not np.isfinite(a).all() for a in scores.values()): raise ValueError('incomplete scores')
        np.savez_compressed(OUT/'SCORES_FROZEN.npz',**scores)
        write(OUT/'DIAGNOSTICS.json',diagnostics)
        write(OUT/'SCORING_AUDIT.json',dict(status='PASS',answers=len(done),methods=len(METHODS),
            max_readout_delta=max(d['readout_delta'] for d in diagnostics),
            max_ridge_score_delta=float(np.max(np.abs(scores['ridge']-ridge_reference))),
            scores_sha256=sha(OUT/'SCORES_FROZEN.npz'),diagnostics_sha256=sha(OUT/'DIAGNOSTICS.json')))
        state('SCORING_COMPLETE_PENDING_EVALUATION')
    except BaseException as error:
        db.commit();state('FAILED',error=repr(error));raise
    finally:db.close()


if __name__=='__main__':
    with threadpool_limits(limits=1):run()
