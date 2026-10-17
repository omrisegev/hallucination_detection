"""Post-fit state audit, separate from the frozen scoring/evaluation implementation."""
import io
import json
from pathlib import Path
import sqlite3
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/rbm_weight_shrinkage_v1'


def main():
    status=json.loads((OUT/'RUN_STATE.json').read_text())
    assert status['status']=='COMPLETE' and status['review']=='PASS'
    review=json.loads((OUT/'RESULT_REVIEW.json').read_text())
    assert review['status']=='PASS'
    conn=sqlite3.connect('file:'+str(OUT/'CHECKPOINT.sqlite')+'?mode=ro',uri=True)
    rows=conn.execute('SELECT idx,payload,info FROM answers ORDER BY idx').fetchall()
    conn.close()  # Release the read transaction before decoding/analysis.
    assert [r[0] for r in rows]==list(range(13769))
    methods=('rbm','rbm_initial','rbm_shrinkage')
    distances={m:[] for m in methods};converged={m:0 for m in methods}
    errors={m:0 for m in methods};flips={m:0 for m in methods};iteration={m:[] for m in methods}
    checks=0;changed_direction=0;paired_distances=[]
    for i,payload,info in rows:
        d=json.loads(info)
        with np.load(io.BytesIO(payload)) as z:
            for j,m in enumerate(methods):
                if m in d['failures']:
                    errors[m]+=1;continue
                diag=d['diagnostics'][m];w=z[m+'::w'];b=z[m+'::b'];a=z[m+'::a']
                assert len(w)==len(a)==diag['active_columns']
                assert np.isfinite(w).all() and np.isfinite(a).all() and np.isfinite(b)
                expected=np.zeros(6);expected[diag['columns']]=diag['orientation']*w
                np.testing.assert_array_equal(expected,z['weights'][j,:6])
                distance=float(np.sum((w-2/len(w))**2));distances[m].append(distance**.5)
                np.testing.assert_allclose(distance**.5,diag['raw_weight_distance_from_initial'],atol=1e-12)
                if m=='rbm_shrinkage':
                    np.testing.assert_allclose(.1*distance,diag['penalty'],atol=1e-12)
                    np.testing.assert_allclose(diag['objective_final'],diag['nll_final']+.1*distance,atol=1e-12)
                    assert diag['objective_final']<=diag['objective_initial']+1e-8
                    assert diag['regularization']==.1
                if m=='rbm_initial':
                    np.testing.assert_array_equal(w,np.full(len(w),2/len(w)))
                    np.testing.assert_array_equal(a,0);assert b==0
                else:
                    converged[m]+=diag['converged'];iteration[m].append(diag['iterations'])
                flips[m]+=diag['orientation']<0
                checks+=1
            if all(m in d['diagnostics'] for m in ('rbm','rbm_shrinkage')):
                changed_direction+=d['diagnostics']['rbm']['orientation']!=d['diagnostics']['rbm_shrinkage']['orientation']
                paired_distances.append(tuple(d['diagnostics'][m]['raw_weight_distance_from_initial'] for m in ('rbm','rbm_shrinkage')))
    metrics=json.loads((OUT/'METRICS.json').read_text())
    result=dict(status='PASS',scope='Post-fit algebra/state audit; separate metric arithmetic review also PASS, no external reviewer.',
        answers=len(rows),fit_state_checks=checks,failures=errors,converged=converged,
        orientation_flips=flips,changed_latent_direction=changed_direction,
        raw_distance_median={m:float(np.median(v)) for m,v in distances.items()},
        paired_distance_answers=len(paired_distances),
        raw_distance_reduced_answers=sum(after<before for before,after in paired_distances),
        iterations_median={m:float(np.median(v)) for m,v in iteration.items() if v},
        mean_oriented_weights=metrics['weights'])
    (OUT/'STATE_REVIEW.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='mean_oriented_weights'},indent=2))


if __name__=='__main__':main()
