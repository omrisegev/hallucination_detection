"""Read-only saved-model audit, independent of fitting; write review only."""
import io
import json
import sqlite3
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spectral_utils.higher_moment_fusion import METHODS, feature_names
OUT=ROOT/'results/higher_moment_fusion_v1'


def replay_real_examples():
    from scipy.special import expit
    from scripts.run_direct_probability_temporal import old, source_specs
    source=ROOT.parents[1]
    old.configure_source_root(source)
    records=json.loads((old.BENCH/'evaluation/JOINED.json').read_text())['records']
    smoke=json.loads((OUT/'SMOKE.json').read_text())['rows']
    chosen_ids={r['uid'] for r in smoke}
    con=sqlite3.connect('file:'+str(OUT/'CHECKPOINT.sqlite')+'?mode=ro',uri=True)
    checks=0;max_error=0.
    for cell,path,kind,dataset in source_specs():
        selected=[(i,r) for i,r in enumerate(records) if r['cell']==cell and r['uid'] in chosen_ids]
        rows=old._source_row_map(old.load_pickle(path),kind=kind,dataset=dataset)
        for i,r in selected:
            payload,info=con.execute('SELECT payload,info FROM answers WHERE idx=?',(i,)).fetchone()
            d=json.loads(info);raw=rows[r['row_id']]
            lp=np.asarray(old._topk_payload(raw)['logprobs'],float)[:,:15]
            p=np.exp(lp);q=p/(p.sum(axis=1,keepdims=True)+1e-12)
            y=-np.log(q+1e-12);h=np.sum(q*y,axis=1)
            v=np.sum(q*(y-h[:,None])**2,axis=1)
            a=np.maximum(np.asarray(raw['token_spilled_energies'],float),0.)
            columns=[h,v,np.sum(q*y**3,axis=1),a,a*a,a*a*a]
            for degree in (4,5,6):columns += [np.sum(q*y**degree,axis=1),a**degree]
            X=np.column_stack(columns)
            with np.load(io.BytesIO(payload),allow_pickle=False) as z:
                for j,m in enumerate(METHODS):
                    if m in d['failures']:continue
                    degree=int(m[1]);diag=d['diagnostics'][m];cols=diag['columns']
                    mean=np.asarray(diag['normalization_mean']);scale=np.asarray(diag['normalization_scale'])
                    np.testing.assert_allclose(X[:,:2*degree].mean(axis=0),mean,rtol=1e-12,atol=1e-10)
                    np.testing.assert_allclose(X[:,:2*degree].std(axis=0),scale,rtol=1e-12,atol=1e-10)
                    normalized=(X[:,cols]-mean[cols])/scale[cols]
                    score=normalized@z[m+'::w']
                    if m.split('__')[1].startswith('rbm'):score=expit(score+z[m+'::b'])
                    if diag['orientation']<0:
                        score=1-score if m.split('__')[1].startswith('rbm') else -score
                    steps=np.array([np.sort(score[u:v])[-10:].mean() for u,v in raw['step_token_spans']])
                    np.testing.assert_allclose(steps,z['steps'][:,j],rtol=1e-10,atol=1e-10)
                    max_error=max(max_error,float(np.max(np.abs(steps-z['steps'][:,j]))));checks+=1
        del rows
    con.close()
    return dict(answers=len(chosen_ids),model_step_replays=checks,max_abs_error=max_error)


def main():
    status=json.loads((OUT/'RUN_STATE.json').read_text())
    assert status['status']=='COMPLETE' and status['review']=='PASS'
    metrics=json.loads((OUT/'METRICS.json').read_text())
    manifest=json.loads((OUT/'MANIFEST.json').read_text())
    from scripts.run_direct_probability_temporal import old
    for path,digest in manifest['hashes'].items():
        assert old.sha256_file(Path(path))==digest, path
    conn=sqlite3.connect('file:'+str(OUT/'CHECKPOINT.sqlite')+'?mode=ro',uri=True)
    diagnostics={m:dict(valid=0,failed=0,nonconverged=0,collapsed=0,orientation_flips=0,
                       distance=[],weight_norm=[],iteration=[]) for m in METHODS}
    checked=0
    for start in range(0,13769,128):
        # Materialize each batch before inspecting arrays; no lingering SELECT cursor.
        rows=conn.execute('SELECT idx,payload,info FROM answers WHERE idx>=? AND idx<? ORDER BY idx',
                          (start,min(start+128,13769))).fetchall()
        assert [r[0] for r in rows]==list(range(start,min(start+128,13769)))
        for i,payload,info in rows:
            d=json.loads(info)
            with np.load(io.BytesIO(payload),allow_pickle=False) as z:
                for j,m in enumerate(METHODS):
                    deg=int(m[1]);solver=m.split('__')[1];tel=diagnostics[m]
                    if m in d['failures']:
                        tel['failed']+=1
                        assert np.isnan(z['steps'][:,j]).all()
                        continue
                    diag=d['diagnostics'][m];w=z[m+'::w'];cols=diag['columns']
                    assert tuple(diag['feature_names'])==feature_names(deg)
                    assert len(w)==diag['active_columns']==len(cols)
                    assert diag['order']==deg and diag['orientation'] in (-1,1)
                    assert np.isfinite(w).all() and np.isfinite(z['steps'][:,j]).all()
                    expected=np.zeros(2*deg);expected[cols]=diag['orientation']*w
                    np.testing.assert_array_equal(expected,z['weights'][j,:2*deg])
                    if solver in ('equal','rbm_initial'):
                        np.testing.assert_array_equal(w,np.full(len(w),(2. if solver=='rbm_initial' else 1.)/len(w)))
                    if solver.startswith('rbm'):
                        a=z[m+'::a'];b=z[m+'::b']
                        assert len(a)==len(w) and np.isfinite(a).all() and np.isfinite(b)
                        assert np.all((z['steps'][:,j]>=-1e-15)&(z['steps'][:,j]<=1+1e-15))
                        if solver=='rbm_initial':
                            np.testing.assert_array_equal(a,0);assert b==0
                        else:
                            assert diag['nll_final']<=diag['nll_initial']+1e-8
                            tel['nonconverged']+=int(not diag['converged'])
                            tel['iteration'].append(diag['iterations'])
                        tel['distance'].append(float(np.linalg.norm(w-2/len(w))))
                    tel['valid']+=1;tel['collapsed']+=diag['collapsed']
                    tel['orientation_flips']+=diag['orientation']<0
                    tel['weight_norm'].append(float(np.linalg.norm(w)))
                    checked+=1
    conn.close()
    for m,t in diagnostics.items():
        assert t['valid']+t['failed']==13769
        assert t['failed']==len(metrics['telemetry'][m]['failures'])
        for key in ('distance','weight_norm','iteration'):
            values=t.pop(key)
            t[key+'_quantiles']=np.quantile(values,[0,.25,.5,.75,1]).tolist() if values else []
    result=dict(status='PASS',answers=13769,model_records_checked=checked,
                scope='Saved-state dimensions, coefficients, finite outputs, likelihood descent, fixed initial controls and input hashes. Separate metric arithmetic review required. Not external scientific validation.',
                telemetry=diagnostics,independent_real_token_replay=replay_real_examples())
    (OUT/'STATE_REVIEW.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print('PASS saved state audit:',checked,'models',flush=True)


if __name__=='__main__':main()
