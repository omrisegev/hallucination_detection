"""Separate arithmetic of saved RBM states, selection and raw token replays."""
import io,json,sqlite3,sys
from pathlib import Path
import numpy as np
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spectral_utils.dufs_moment_selection import METHODS
from spectral_utils.higher_moment_fusion import feature_names
OUT=ROOT/'results/dufs_moment_selection_v1'


def replay_real_examples():
    from scripts.run_direct_probability_temporal import old,source_specs
    old.configure_source_root(ROOT.parents[1])
    records=json.loads((old.BENCH/'evaluation/JOINED.json').read_text())['records']
    ids={r['uid'] for r in json.loads((OUT/'SMOKE.json').read_text())['rows']}
    con=sqlite3.connect('file:'+str(OUT/'CHECKPOINT.sqlite')+'?mode=ro',uri=True)
    checks=0;max_error=0.;selectors=0
    for cell,path,kind,dataset in source_specs():
        selected=[(i,r) for i,r in enumerate(records) if r['cell']==cell and r['uid'] in ids]
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
            for n in (4,5,6):columns += [np.sum(q*y**n,axis=1),a**n]
            X=np.column_stack(columns);mu=X.mean(0);sd=X.std(0);active=np.flatnonzero(sd>1e-10)
            Z=(X[:,active]-mu[active])/sd[active]
            # Independent greedy implementation, distinct from core helper.
            C=np.corrcoef(Z,rowvar=False)**2;np.fill_diagonal(C,0)
            order=[min(range(len(active)),key=lambda k:(sum(C[:,k]),k))]
            while len(order)<6:
                order.append(min(set(range(len(active)))-set(order),key=lambda k:(sum(C[k,j] for j in order),k)))
            expected=sorted(active[order].tolist())
            with np.load(io.BytesIO(payload),allow_pickle=False) as z:
                for j,m in enumerate(METHODS):
                    if m in d['failures']:continue
                    diag=d['diagnostics'][m];cols=diag['columns']
                    np.testing.assert_allclose(diag['normalization_mean'],mu,rtol=1e-12,atol=1e-10)
                    np.testing.assert_allclose(diag['normalization_scale'],sd,rtol=1e-12,atol=1e-10)
                    if m.startswith('correlation6'):
                        assert cols==expected;selectors+=1
                    normalized=(X[:,cols]-mu[cols])/sd[cols]
                    w=z[m+'::w'];shift=z[m+'::a'];b=z[m+'::b']
                    # Recover posterior via two Gaussian components and prior,
                    # independently of the original sigmoid scoring expression.
                    prior_logit=float(b+shift@w+.5*(w@w))
                    l0=-.5*np.sum((normalized-shift)**2,axis=1)-np.logaddexp(0,prior_logit)
                    l1=-.5*np.sum((normalized-shift-w)**2,axis=1)-np.logaddexp(0,-prior_logit)
                    score=np.exp(l1-np.logaddexp(l0,l1))
                    if diag['orientation']<0:score=1-score
                    steps=np.array([np.sort(score[u:v])[-10:].mean() for u,v in raw['step_token_spans']])
                    np.testing.assert_allclose(steps,z['steps'][:,j],rtol=1e-9,atol=1e-10)
                    max_error=max(max_error,float(np.max(np.abs(steps-z['steps'][:,j]))));checks+=1
        del rows
    con.close()
    return dict(answers=len(ids),model_step_replays=checks,correlation_selection_replays=selectors,max_abs_error=max_error)


def main():
    state=json.loads((OUT/'RUN_STATE.json').read_text());assert state['status']=='COMPLETE' and state['review']=='PASS'
    metrics=json.loads((OUT/'METRICS.json').read_text());manifest=json.loads((OUT/'MANIFEST.json').read_text())
    from scripts.run_direct_probability_temporal import old
    for path,digest in manifest['hashes'].items():assert old.sha256_file(Path(path))==digest,path
    con=sqlite3.connect('file:'+str(OUT/'CHECKPOINT.sqlite')+'?mode=ro',uri=True)
    stats={m:dict(valid=0,failed=0,nonconverged=0,flips=0,collapsed=0,distance=[],selected=np.zeros(12,int)) for m in METHODS}
    checked=0
    for start in range(0,13769,128):
        rows=con.execute('SELECT idx,payload,info FROM answers WHERE idx>=? AND idx<? ORDER BY idx',(start,min(start+128,13769))).fetchall()
        assert [r[0] for r in rows]==list(range(start,min(start+128,13769)))
        for i,blob,info in rows:
            d=json.loads(info)
            with np.load(io.BytesIO(blob),allow_pickle=False) as packed:
                z={k:packed[k] for k in packed.files}
            for j,m in enumerate(METHODS):
                t=stats[m]
                if m in d['failures']:
                    t['failed']+=1;assert np.isnan(z['steps'][:,j]).all();continue
                diag=d['diagnostics'][m];cols=np.asarray(diag['columns']);p=len(cols)
                assert tuple(diag['feature_names'])==feature_names(6)
                assert p==diag['active_columns'] and np.all(np.diff(cols)>0)
                assert p==6 if not m.startswith('all12') else p<=12
                assert diag['orientation'] in (-1,1)
                w=z[m+'::w'];a=z[m+'::a'];b=z[m+'::b']
                assert len(w)==len(a)==p and np.isfinite(w).all() and np.isfinite(a).all() and np.isfinite(b)
                mapped=np.zeros(12);mapped[cols]=w*diag['orientation']
                np.testing.assert_array_equal(mapped,z['weights'][j,:12])
                assert np.isfinite(z['steps'][:,j]).all() and np.all((z['steps'][:,j]>=-1e-14)&(z['steps'][:,j]<=1+1e-14))
                if m.endswith('initial'):
                    np.testing.assert_array_equal(w,np.full(p,2./p));np.testing.assert_array_equal(a,0);assert b==0
                else:
                    assert diag['nll_final']<=diag['nll_initial']+1e-8
                    t['nonconverged']+=int(not diag['converged'])
                other=m.split('__')[0]+'__'+('rbm' if m.endswith('initial') else 'rbm_initial')
                if other not in d['failures']:assert diag['columns']==d['diagnostics'][other]['columns']
                if m.startswith('dufs6'):
                    sel=diag['selection'];per=np.asarray(sel['per_seed_probabilities']);raw=np.asarray(sel['raw_probabilities'])
                    assert per.shape[0]==3 and np.isfinite(per).all() and np.all((per>=0)&(per<=1))
                    np.testing.assert_array_equal(raw,per.mean(0))
                    active=np.flatnonzero(np.asarray(diag['normalization_scale'])>1e-10)
                    chosen=sorted(active[sorted(range(len(raw)),key=lambda k:(-raw[k],k))[:6]].tolist())
                    assert cols.tolist()==chosen
                t['valid']+=1;t['collapsed']+=diag['collapsed'];t['flips']+=diag['orientation']<0
                t['selected'][cols]+=1;t['distance'].append(float(np.linalg.norm(w-2./p)));checked+=1
    con.close()
    for m,t in stats.items():
        assert t['valid']+t['failed']==13769
        assert t['failed']==len(metrics['telemetry'][m]['failures'])
        assert t['selected'].tolist()==metrics['telemetry'][m]['selected_column_counts']
        t['selected']=t['selected'].tolist();t['distance_quantiles']=np.quantile(t.pop('distance'),[0,.25,.5,.75,1]).tolist()
    out=dict(status='PASS',answers=13769,model_records_checked=checked,telemetry=stats,
        independent_real_token_replay=replay_real_examples(),
        scope='Code/model/arithmetic checks, not untouched scientific confirmation. DUFS hard-selection probabilities replay; no independent reimplementation of its optimizer.')
    (OUT/'STATE_REVIEW.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
    print('PASS saved states:',checked,flush=True)


if __name__=='__main__':main()
